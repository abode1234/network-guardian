// proxy.cpp  –  single-threaded demo reverse proxy with inline AI call
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <iostream>

namespace beast  = boost::beast;   // from <boost/beast.hpp>
namespace http   = beast::http;
namespace net    = boost::asio;    // from <boost/asio.hpp>
using tcp        = net::ip::tcp;
using json       = nlohmann::json;

struct Args {
    std::string listen_host = "0.0.0.0";
    unsigned    listen_port = 8080;
    std::string upstream_host = "localhost";
    unsigned    upstream_port = 9000;
    std::string detector_url = "http://127.0.0.1:5000/score";
};

Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string v = argv[i];
        if (v == "--listen" && i + 1 < argc) {
            auto pos = std::string(argv[++i]).find(':');
            a.listen_host = argv[i].substr(0, pos);
            a.listen_port = std::stoi(argv[i].substr(pos + 1));
        } else if (v == "--upstream" && i + 1 < argc) {
            auto pos = std::string(argv[++i]).find(':');
            a.upstream_host = argv[i].substr(0, pos);
            a.upstream_port = std::stoi(argv[i].substr(pos + 1));
        } else if (v == "--detector" && i + 1 < argc) {
            a.detector_url = argv[++i];
        }
    }
    return a;
}

// very small helper that posts JSON to the detector and returns {allow:bool}
bool call_detector(const Args& cfg, const http::request<http::string_body>& req,
                   json& enriched_alert) {
    // Serialize raw request into JSON
    json jreq{
        {"method", req.method_string()},
        {"target", req.target()},
        {"body",   req.body()}
    };
    // Build HTTP POST to detector
    auto const pos = cfg.detector_url.find("://");
    auto const host_end = cfg.detector_url.find('/', pos + 3);
    std::string host = cfg.detector_url.substr(pos + 3,
                       host_end - (pos + 3));
    std::string target = cfg.detector_url.substr(host_end);

    net::io_context ioc;
    tcp::resolver resolver{ioc};
    beast::tcp_stream stream{ioc};
    auto const results = resolver.resolve(host, "80");
    stream.connect(results);
    http::request<http::string_body> detreq{http::verb::post, target, 11};
    detreq.set(http::field::host, host);
    detreq.set(http::field::content_type, "application/json");
    detreq.body() = jreq.dump();
    detreq.prepare_payload();
    http::write(stream, detreq);

    beast::flat_buffer buffer;
    http::response<http::string_body> detres;
    http::read(stream, buffer, detres);
    stream.socket().shutdown(tcp::socket::shutdown_both);

    enriched_alert = json::parse(detres.body());
    return enriched_alert.value("allow", true);
}

int main(int argc, char* argv[]) {
    Args cfg = parse_args(argc, argv);
    try {
        net::io_context ioc{1};
        tcp::acceptor acceptor{ioc,
           {net::ip::make_address(cfg.listen_host), cfg.listen_port}};
        for (;;) {
            tcp::socket client = acceptor.accept();

            beast::flat_buffer buffer;
            http::request<http::string_body> creq;
            http::read(client, buffer, creq);

            json alert;
            bool allow = call_detector(cfg, creq, alert);

            if (!allow) {
                http::response<http::string_body> deny{http::status::forbidden,
                                                       creq.version()};
                deny.set(http::field::content_type, "text/plain");
                deny.body() = "Blocked by NetGuardian";
                deny.prepare_payload();
                http::write(client, deny);
                std::cerr << "[ALERT] " << alert.dump() << std::endl;
                continue;
            }

            // Forward to upstream
            tcp::resolver r{ioc};
            beast::tcp_stream upstream{ioc};
            upstream.connect(r.resolve(cfg.upstream_host,
                                       std::to_string(cfg.upstream_port)));
            http::write(upstream, creq);

            http::response<http::string_body> ures;
            beast::flat_buffer buf2;
            http::read(upstream, buf2, ures);
            upstream.socket().shutdown(tcp::socket::shutdown_both);

            http::write(client, ures);
        }
    } catch (const std::exception& e) {
        std::cerr << "Proxy error: " << e.what() << '\n';
    }
}
