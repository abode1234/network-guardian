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
