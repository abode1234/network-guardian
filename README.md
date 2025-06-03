## NetGuardian Detector Service (AI-Driven HTTP Inspection)

**NetGuardian** integrates an AI-powered detector to inspect incoming HTTP requests in real time, pinpoint exploitable parameters, and emit alerts. This post explains every part of the **`detector.py`** service, provides the requested ASCII flowchart, and shows how to integrate it with your C++ proxy.

---

### ASCII Flowchart (Pipeline Overview)

```
┌────────┐     1) Receive JSON         ┌──────────┐
│  Proxy │ ─────────────────────────────▶│ Feature  │  ← 80-field schema 
│  (C++) │                                 │Extractor │
└────────┘                                 └──────────┘
      │2) Vectorized features                        ▲
      ▼                                              │
┌────────┐ 3) Predict & SHAP        ┌──────────┐ 4) (prob, param) ───┘
│ XGBoost │────────────────────────▶│ Enricher │─ Map top feature to param
│ Model   │    (attack probability) │          │  (vulnerable_param)
└────────┘                          └──────────┘
      │5) allow/block decision  
      ▼
┌────────┐ 6) Return JSON 
│ Detector│     (with allow, prob, vuln_param)
└────────┘
      │
      ▼
┌────────┐ 7) Proxy forwards or blocks request
│  Proxy │  and logs alert if blocked
└────────┘
      │
      ▼
┌──────────┐
│ Alert    │  Optional: resend JSON to webhook
│  Sink    │  (Slack/Teams/Syslog)
└──────────┘
```

**Pipeline Steps:**

1. **Proxy (C++)** serializes the HTTP request into JSON and posts to `/score`.
2. Detector’s **Feature Extractor** builds an 80-dimensional vector.
3. **XGBoost Model** predicts attack probability; **SHAP** computes feature importances.
4. **Enricher** maps the highest-impact feature back to the original parameter.
5. Detector decides **allow/block** based on a threshold.
6. Detector returns a JSON containing `allow`, `probability`, and `vulnerable_param`.
7. Proxy uses `allow` to forward or block and logs the alert. Optionally, the detector **webhooks** the alert JSON to an external sink.

---

## 1. Prerequisites & Installation

* Python 3.8+
* Install dependencies:

  ```bash
  pip install flask xgboost shap pyyaml requests
  ```
* A trained XGBoost model file: `netguardian_model.json`
* A feature schema file: `schema.yaml` (exactly 80 entries)
* (Optional) A Slack/Teams webhook URL for alerts.

---

## 2. `schema.yaml` Format

Your **`schema.yaml`** must list exactly 80 entries, each defining how to extract one feature. Order must match model training.

Example:

```yaml
- name: len_target
  type: len_target
  param_hint: null

- name: num_special_chars
  type: num_special
  param_hint: null

- name: arg_entropy_user
  type: entropy
  param_hint: user

- name: arg_entropy_pass
  type: entropy
  param_hint: pass

- name: num_params
  type: num_params
  param_hint: null

# ... continue until 80 total ...
```

**Extraction Types:**

* `len_target`: `len(target)`
* `num_special`: Count of non-alphanumeric characters in `(target + body)`
* `entropy`: Shannon entropy of the value for `param_hint` key
* `num_params`: Count of `=` in `(target + body)`
* Extend for any custom features you trained on.

---

## 3. `detector.py` – Full Source Code

```python
# detector.py  – XGBoost + SHAP HTTP Detector & Alert Service
#
# Usage:
#   python detector.py \
#     --model netguardian_model.json \
#     --features schema.yaml \
#     --threshold 0.8 \
#     --webhook https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXX

import argparse
import json
import time
import yaml
import shap
import xgboost as xgb
import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1) Argument parsing

def parse_args():
    p = argparse.ArgumentParser(description="NetGuardian AI Detector")
    p.add_argument(
        "--model",
        required=True,
        help="Path to the XGBoost model JSON (e.g., netguardian_model.json)"
    )
    p.add_argument(
        "--features",
        required=True,
        help="Path to YAML schema (feature name + extraction rule)"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Probability threshold above which we block"
    )
    p.add_argument(
        "--webhook",
        help="Optional: HTTP endpoint to POST JSON alerts"
    )
    return p.parse_args()

args = parse_args()

# 2) Load model + schema
booster = xgb.Booster()
booster.load_model(args.model)

with open(args.features) as f:
    feature_defs = yaml.safe_load(f)

explainer = shap.TreeExplainer(booster)

# 3) Feature extraction

def extract_features(req_json):
    """
    Build an 80-dimensional vector from the incoming JSON.
    req_json = {"method": "GET", "target": "/login?user=admin&pass=123", "body": ""}
    feature_defs: list of {name, type, param_hint}
    """
    feats = []
    target = req_json.get("target", "")
    body   = req_json.get("body", "")

    for feat in feature_defs:
        ftype = feat["type"]
        if ftype == "len_target":
            feats.append(len(target))
        elif ftype == "num_special":
            combined = target + body
            feats.append(sum(1 for c in combined if not c.isalnum()))
        elif ftype == "entropy":
            key = feat.get("param_hint", "")
            val = ""
            if key and key + "=" in (target + "&" + body):
                part = (target + "&" + body).split(key + "=")[1]
                val = part.split("&")[0]
            if val:
                prob = []
                for ch in set(val):
                    p = val.count(ch) / len(val)
                    prob.append(-p * np.log2(p))
                feats.append(sum(prob))
            else:
                feats.append(0.0)
        elif ftype == "num_params":
            combined = target + "&" + body
            feats.append(combined.count("="))
        else:
            feats.append(0.0)
    return np.array([feats], dtype=np.float32)

# 4) SHAP mapping to parameter

def shap_to_param(req_json, shap_vals):
    """
    Map top SHAP feature back to its param key/value.
    """
    top_idx = int(np.argmax(np.abs(shap_vals)))
    feat = feature_defs[top_idx]
    key = feat.get("param_hint", "")
    sample_val = "<n/a>"

    combined = req_json.get("target", "") + "&" + req_json.get("body", "")
    if key and key + "=" in combined:
        sample_val = combined.split(key + "=")[1].split("&")[0][:100]
    return key, sample_val

# 5) Flask route

@app.route("/score", methods=["POST"])
def score():
    req_json = request.get_json(force=True)

    # 5.1) Build feature vector
    vec = extract_features(req_json)
    dmatrix = xgb.DMatrix(vec)

    # 5.2) Inference
    prob = float(booster.predict(dmatrix)[0])
    allow = prob < args.threshold

    # 5.3) SHAP explain
    shap_vals = explainer.shap_values(vec)[0]
    param_key, param_val = shap_to_param(req_json, shap_vals)

    # 5.4) Build alert JSON
    alert = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probability": prob,
        "allow": allow,
        "vulnerable_param": param_key,
        "sample_value": param_val,
        "method": req_json.get("method", ""),
        "target": req_json.get("target", "")
    }

    # 5.5) Send to webhook if blocked
    if not allow and args.webhook:
        try:
            requests.post(args.webhook, json=alert, timeout=2)
        except Exception as e:
            print(f"[Warning] Webhook POST failed: {e}")

    return jsonify(alert)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

---

## 4. Explanation of Key Sections

1. **Argument Parsing:**

   * `--model`: Path to exported XGBoost JSON.
   * `--features`: YAML file listing 80 features.
   * `--threshold`: Float; if probability ≥ threshold, we block.
   * `--webhook`: (Optional) URL to POST alert JSON.

2. **Model & SHAP Loading:**

   * `booster.load_model(args.model)` loads the trained XGBoost model.
   * `TreeExplainer(booster)` computes SHAP values at runtime for explainability.

3. **Feature Extraction (`extract_features`):**

   * Reads `req_json["target"]` (URI+query) and `req_json["body"]` (POST/PUT payload).
   * Loops through `feature_defs` in order, computes each feature type, and appends to a list.
   * Returns a NumPy array of shape `(1, 80)`, matching model input.

4. **SHAP Mapping (`shap_to_param`):**

   * Takes `shap_vals` (length 80) and finds the index with maximum absolute SHAP.
   * Retrieves `param_hint` (parameter name) from that feature.
   * Extracts the actual value of that parameter from `(target + "&" + body)`.

5. **Flask `/score` Endpoint:**

   * Receives POSTed JSON: `{ "method": "GET", "target": "/path?x=1&y=2", "body": "" }`
   * Builds features, runs inference, and computes SHAP.
   * Constructs `alert` JSON with keys: timestamp, probability, allow, vulnerable\_param, sample\_value, method, target.
   * If `allow == False` and `--webhook` is set, posts alert to the webhook.
   * Returns the same JSON to the caller (the C++ proxy).

---

## 5. Integration with the C++ Proxy

To enforce inline blocking and leverage the AI detector, the **C++ proxy** must:

1. **Serialize each HTTP request** into a JSON object matching `detector.py`’s expected format:

   ```json
   {
     "method": "GET",
     "target": "/login?user=admin&pass=123",
     "body": ""
   }
   ```
2. **POST** this JSON to `http://<detector-host>:5000/score`.
3. **Parse** the detector’s JSON response:

   ```json
   {
     "timestamp": "2025-06-03T12:00:00+0000",
     "probability": 0.92,
     "allow": false,
     "vulnerable_param": "pass",
     "sample_value": "123' OR '1'='1",
     "method": "GET",
     "target": "/login?user=admin&pass=123"
   }
   ```
4. If `allow == false`, the proxy returns **HTTP 403 Forbidden** to the client and logs the alert JSON to stderr (or a file).
5. If `allow == true`, the proxy **forwards** the original HTTP request to the upstream server and relays the response.

Below is a complete example of `proxy.cpp` with inline comments.

```cpp
// proxy.cpp  –  Inline AI-Driven SIEM Proxy
//
// Build:
//   g++ -std=c++17 proxy.cpp -lboost_system -lboost_thread -lpthread -o netguardian-proxy
// Run:
//   ./netguardian-proxy \
//     --listen 0.0.0.0:8080 \
//     --upstream localhost:9000 \
//     --detector http://127.0.0.1:5000/score

#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

namespace beast  = boost::beast;   // from <boost/beast.hpp>
namespace http   = beast::http;    // from <boost/beast/http.hpp>
namespace net    = boost::asio;    // from <boost/asio.hpp>
using tcp        = net::ip::tcp;
using json       = nlohmann::json;

// Holds command-line arguments
struct Args {
    std::string listen_host   = "0.0.0.0";
    unsigned    listen_port   = 8080;
    std::string upstream_host = "localhost";
    unsigned    upstream_port = 9000;
    std::string detector_url  = "http://127.0.0.1:5000/score";
};

// Parse flags: --listen host:port, --upstream host:port, --detector URL
Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string v = argv[i];
        if (v == "--listen" && i + 1 < argc) {
            std::string val = argv[++i];
            auto pos = val.find(':');
            args.listen_host = val.substr(0, pos);
            args.listen_port = std::stoi(val.substr(pos + 1));
        }
        else if (v == "--upstream" && i + 1 < argc) {
            std::string val = argv[++i];
            auto pos = val.find(':');
            args.upstream_host = val.substr(0, pos);
            args.upstream_port = std::stoi(val.substr(pos + 1));
        }
        else if (v == "--detector" && i + 1 < argc) {
            args.detector_url = argv[++i];
        }
    }
    return args;
}

// POST the HTTP request as JSON to the detector; return "allow" and fill alert_json
bool call_detector(const Args& cfg,
                   const http::request<http::string_body>& req,
                   json&                 alert_json)
{
    // 1) Serialize incoming request to JSON
    json jreq;
    jreq["method"] = req.method_string().to_string();
    jreq["target"] = req.target().to_string();
    jreq["body"]   = req.body();  // raw body if any

    // 2) Parse detector_url (e.g., "http://127.0.0.1:5000/score")
    auto scheme_sep = cfg.detector_url.find("://");
    auto host_start = scheme_sep + 3;
    auto path_start = cfg.detector_url.find('/', host_start);
    std::string host = cfg.detector_url.substr(host_start,
                                                path_start - host_start);
    std::string target = cfg.detector_url.substr(path_start);
    std::string port = "80";
    if (host.find(':') != std::string::npos) {
        auto colonPos = host.find(':');
        port = host.substr(colonPos + 1);
        host = host.substr(0, colonPos);
    }

    // 3) Connect to detector
    net::io_context          ioc;
    tcp::resolver            resolver{ioc};
    beast::tcp_stream        stream{ioc};
    auto const results = resolver.resolve(host, port);
    stream.connect(results);

    // 4) Build HTTP POST
    http::request<http::string_body> detreq{http::verb::post, target, 11};
    detreq.set(http::field::host, host);
    detreq.set(http::field::content_type, "application/json");
    detreq.body() = jreq.dump();
    detreq.prepare_payload();

    http::write(stream, detreq);

    // 5) Read response
    beast::flat_buffer                 buffer;
    http::response<http::string_body>  detres;
    http::read(stream, buffer, detres);
    stream.socket().shutdown(tcp::socket::shutdown_both);

    // 6) Parse JSON
    alert_json = json::parse(detres.body());
    return alert_json.value("allow", true);
}

int main(int argc, char* argv[]) {
    auto cfg = parse_args(argc, argv);

    try {
        net::io_context ioc{1};

        // 7) Listen for incoming client connections
        tcp::acceptor acceptor{ioc,
            {net::ip::make_address(cfg.listen_host),
             cfg.listen_port}};

        std::cout << "[NetGuardian] Listening on "
                  << cfg.listen_host << ":" << cfg.listen_port
                  << ", upstream=" << cfg.upstream_host
                  << ":" << cfg.upstream_port
                  << ", detector=" << cfg.detector_url << "\n";

        for (;;) {
            // 8) Accept a new client socket
            tcp::socket client{ioc};
            acceptor.accept(client);

            // 9) Read full HTTP request from client
            beast::flat_buffer                 buffer;
            http::request<http::string_body>   creq;
            http::read(client, buffer, creq);

            // 10) Call detector
            json alert_json;
            bool allow = call_detector(cfg, creq, alert_json);

            if (!allow) {
                // 11) Block and return 403
                http::response<http::string_body> deny{http::status::forbidden,
                                                       creq.version()};
                deny.set(http::field::content_type, "text/plain");
                deny.body() = "Blocked by NetGuardian";
                deny.prepare_payload();
                http::write(client, deny);

                // Log the JSON alert
                std::cerr << "[ALERT] " << alert_json.dump() << std::endl;
                continue;
            }

            // 12) If allowed, forward to upstream
            tcp::resolver            r{ioc};
            beast::tcp_stream        upstream{ioc};
            auto const upstream_endpoints =
                r.resolve(cfg.upstream_host,
                          std::to_string(cfg.upstream_port));
            upstream.connect(upstream_endpoints);

            // 13) Send original request to upstream
            http::write(upstream, creq);

            // 14) Read upstream response
            beast::flat_buffer                 buf2;
            http::response<http::string_body>  ures;
            http::read(upstream, buf2, ures);
            upstream.socket().shutdown(tcp::socket::shutdown_both);

            // 15) Relay upstream response to client
            http::write(client, ures);
        }

    } catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```

### Explanation of `proxy.cpp`

* **Parsing Arguments**: `--listen 0.0.0.0:8080`, `--upstream localhost:9000`, `--detector http://127.0.0.1:5000/score`.
* **`call_detector()`**:

  1. Serialize method, target, body into JSON.
  2. Connect to the detector’s host and port.
  3. POST JSON to `/score`.
  4. Read JSON response and parse into `alert_json`.
  5. Return `alert_json["allow"]`.
* In **`main`**:

  1. Listen on the configured host\:port for incoming client connections.
  2. Read the HTTP request.
  3. Call the detector; if `allow == false`, return **403 Forbidden** and log alert.
  4. Otherwise, connect to the upstream server, forward the request, read the response, and send it back to the client.

By integrating these two components, NetGuardian blocks malicious HTTP requests **inline** and provides explainable alerts about vulnerable parameters.

---

## 6. Testing & Validation

### A) Smoke Test

1. Run a simple HTTP server on port 9000:

   ```bash
   python -m http.server 9000
   ```
2. Start Detector:

   ```bash
   python detector.py --model netguardian_model.json --features schema.yaml --threshold 0.5
   ```
3. Start Proxy:

   ```bash
   ./netguardian-proxy --listen 127.0.0.1:8080 --upstream 127.0.0.1:9000 --detector http://127.0.0.1:5000/score
   ```
4. Send benign request:

   ```bash
   curl "http://127.0.0.1:8080/?q=test"
   # Expect HTTP 200 from the simple server
   ```
5. Send malicious request:

   ```bash
   curl "http://127.0.0.1:8080/?user=admin&pass=123' OR '1'='1"
   # Expect HTTP 403 Forbidden; alert logged in detector
   ```

### B) Unit Test Feature Extraction

Create `test_features.py`:

```python
import pytest
from detector import extract_features, feature_defs

def get_index(name):
    for i, f in enumerate(feature_defs):
        if f["name"] == name:
            return i
    return -1

def test_len_target_and_num_special():
    req = {"method": "GET", "target": "/search?q=test123", "body": ""}
    vec = extract_features(req)[0]
    assert vec[get_index("len_target")] == len("/search?q=test123")
    assert vec[get_index("num_special_chars")] >= 2  # '/' and '?' count

if __name__ == "__main__":
    pytest.main(["-v"])
```

Run:

```bash
pytest test_features.py
```

### C) Load Testing

Use **wrk** or **hey**:

```bash
wrk -t4 -c200 -d30s http://127.0.0.1:8080/?q=benchmark
```

* Monitor CPU/memory of proxy & detector.
* Tune `--threshold` or model complexity as needed.

---

## 7. Conclusion

You now have a complete **AI-Driven Detector Service** for NetGuardian that:

1. **Extracts 80 features** from each HTTP request in real time.
2. **Runs XGBoost inference** to detect malicious payloads.
3. **Uses SHAP** to identify the most suspect parameter in each request.
4. **Returns JSON** indicating `allow`/`block`, probability, and `vulnerable_param`.
5. **Optionally webhooks** alerts to Slack/Teams/other systems.

When combined with the **C++ proxy**, every request is scored and blocked inline if malicious. Developers receive **precise alerts** about vulnerable parameters, enabling rapid patching.


github link: https://github.com/abode1234/network-guardian
