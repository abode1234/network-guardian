# detector.py – loads model once & answers /score POSTs
import argparse, json, os, time, yaml, shap
import xgboost as xgb
import flask, requests
from flask import request, jsonify
import numpy as np

app = flask.Flask(__name__)

# ---- CLI / bootstrap -------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True,
                   help="YAML mapping: name -> extractor regex / func id")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--webhook")
    return p.parse_args()

args = parse_cli()
booster = xgb.Booster()
booster.load_model(args.model)
feature_defs = yaml.safe_load(open(args.features))
explainer = shap.TreeExplainer(booster)

def extract_features(req_json):
    """Produce vector in the same order as feature_defs."""
    feats = []
    for f in feature_defs:
        if f["type"] == "len_target":
            feats.append(len(req_json["target"]))
        elif f["type"] == "num_special":
            feats.append(sum(1 for c in req_json["body"] if not c.isalnum()))
        # ... add the rest
    return np.array([feats])

def shap_to_param(req_json, shap_vals):
    """Map top shap index back to parameter key/value."""
    top_idx = int(np.argmax(np.abs(shap_vals)))
    key = feature_defs[top_idx].get("param_hint", "<none>")
    val = "<too_long>"
    if "body" in req_json and key in req_json["body"]:
        val = req_json["body"].split(key + "=")[1].split("&")[0][:60]
    return key, val

# ---- Flask route -----------------------------------------------------
@app.route("/score", methods=["POST"])
def score():
    req_json = request.get_json(force=True)
    vec = extract_features(req_json)
    dmatrix = xgb.DMatrix(vec)
    prob = float(booster.predict(dmatrix)[0])
    allow = prob < args.threshold

    shap_vals = explainer.shap_values(vec)[0]
    pkey, pval  = shap_to_param(req_json, shap_vals)

    alert = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probability": prob,
        "allow": allow,
        "vulnerable_param": pkey,
        "sample_value": pval,
        "method": req_json["method"],
        "target": req_json["target"]
    }

    if not allow and args.webhook:
        requests.post(args.webhook, json=alert, timeout=2)

    return jsonify(alert)

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
