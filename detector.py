# detector.py  –  XGBoost + SHAP HTTP Detector & Alert Service
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
        help="Optional: HTTP endpoint to POST JSON alerts (Slack, Teams, etc.)"
    )
    return p.parse_args()

args = parse_args()

# 1) Load the XGBoost model
booster = xgb.Booster()
booster.load_model(args.model)

# 2) Load feature definitions and parameter hints
# schema.yaml should be a list of dicts, each with:
#   - name: "num_special_chars"
#   - type: "num_special"
#   - param_hint: "user"
with open(args.features) as f:
    feature_defs = yaml.safe_load(f)

explainer = shap.TreeExplainer(booster)

def extract_features(req_json):
    """
    Build an 80-dimensional vector from the incoming JSON.
    req_json = {
      "method": "POST",
      "target": "/login?user=admin&pass=123",
      "body": "csrf=abcd"
    }
    feature_defs is a list of dicts:
      [{"name": "len_target", "type": "len_target", "param_hint": null},
       {"name": "num_special_chars", "type": "num_special", "param_hint": null},
       {"name": "arg_entropy_user", "type": "entropy", "param_hint": "user"},
       ... 80 total ... ]
    """
    feats = []
    target = req_json.get("target", "")
    body   = req_json.get("body", "")

    for feat in feature_defs:
        ftype = feat["type"]
        if ftype == "len_target":
            feats.append(len(target))
        elif ftype == "num_special":
            # Count non-alphanumeric characters in (target + body)
            combined = target + body
            feats.append(sum(1 for c in combined if not c.isalnum()))
        elif ftype == "entropy":
            # Approximate by counting unique characters in param value
            key = feat.get("param_hint", "")
            val = ""
            if key and key + "=" in (target + "&" + body):
                # Extract up to next '&' or end
                part = (target + "&" + body).split(key + "=")[1]
                val = part.split("&")[0]
            # Shannon entropy on 'val'
            if val:
                prob = []
                for ch in set(val):
                    p = val.count(ch) / len(val)
                    prob.append(-p * np.log2(p))
                feats.append(sum(prob))
            else:
                feats.append(0.0)
        elif ftype == "num_params":
            # Count how many key=value pairs in query+body
            combined = target + "&" + body
            feats.append(combined.count("="))
        # ... add all other custom feature types per your schema ...
        else:
            feats.append(0.0)
    return np.array([feats], dtype=np.float32)

def shap_to_param(req_json, shap_vals):
    """
    Map the top SHAP feature back to its original parameter key/value.
    shap_vals is a length-80 array. We take argmax of abs(...) to find the
    feature that contributed most. Then use 'param_hint' to look up in target/body.
    """
    top_idx = int(np.argmax(np.abs(shap_vals)))
    feat = feature_defs[top_idx]
    key = feat.get("param_hint", "")
    sample_val = "<n/a>"

    # Try to extract the actual value of that parameter from the request
    combined = req_json.get("target", "") + "&" + req_json.get("body", "")
    if key and key + "=" in combined:
        sample_val = combined.split(key + "=")[1].split("&")[0][:100]
    return key, sample_val

@app.route("/score", methods=["POST"])
def score():
    req_json = request.get_json(force=True)

    # 1) Build feature vector
    vec = extract_features(req_json)
    dmatrix = xgb.DMatrix(vec)

    # 2) Run inference
    prob = float(booster.predict(dmatrix)[0])
    allow = prob < args.threshold

    # 3) Compute SHAP values
    shap_vals = explainer.shap_values(vec)[0]
    param_key, param_val = shap_to_param(req_json, shap_vals)

    # 4) Build alert JSON
    alert = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probability": prob,
        "allow": allow,
        "vulnerable_param": param_key,
        "sample_value": param_val,
        "method": req_json.get("method", ""),
        "target": req_json.get("target", "")
    }

    # 5) If malicious, POST to webhook
    if not allow and args.webhook:
        try:
            requests.post(args.webhook, json=alert, timeout=2)
        except Exception as e:
            # If webhook fails, log and continue
            print(f"[Warning] Webhook POST failed: {e}")

    return jsonify(alert)

if __name__ == "__main__":
    # Start Flask in production mode if needed (uwsgi/gunicorn recommended)
    app.run(host="0.0.0.0", port=5000)
