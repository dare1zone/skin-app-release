
from functools import lru_cache
import os, json, numpy as np, tensorflow as tf

@lru_cache(maxsize=1)
def load_model_and_meta():
    mpath = "models/ntd/ntd_mbv2_roi.keras"
    if not os.path.exists(mpath):
        raise FileNotFoundError("Missing models/ntd/ntd_mbv2_roi.keras")
    model = tf.keras.models.load_model(mpath)
    class_names = ["buruli_ulcer","leishmaniasis","leprosy"]
    lpath = "models/ntd/ntd_labels.json"
    if os.path.exists(lpath):
        try:
            data = json.load(open(lpath, "r"))
            if isinstance(data, dict) and "class_names" in data:
                class_names = data["class_names"]
            elif isinstance(data, list):
                class_names = data
        except Exception:
            pass
    per_class_thr = [0.55, 0.55, 0.593]
    top2_margin   = 0.172
    return model, class_names, per_class_thr, top2_margin

def ntd_predict_with_guardrails(x):
    model, class_names, thr, margin = load_model_and_meta()
    p = model.predict(x, verbose=0)[0]
    order = np.argsort(p)[::-1]
    top = float(p[order[0]])
    sec = float(p[order[1]] if len(p) > 1 else 0.0)
    if top < thr[order[0]] or (top - sec) < margin:
        return p, order, True, "low confidence or borderline"
    return p, order, False, ""
