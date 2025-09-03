import numpy as np, cv2

def _entropy(gray):
    h = cv2.calcHist([gray],[0],None,[64],[0,256]).ravel()
    p = h / (h.sum() + 1e-6)
    nz = p[p>0]
    return float(-(nz*np.log2(nz)).sum())

def should_abstain_v2(probs, heatmap, rgb_img):
    p = np.asarray(probs, dtype=np.float32).ravel()
    conf = float(p.max()) if p.size else 0.0
    second = float(np.partition(p, -2)[-2]) if p.size>1 else 0.0
    margin = conf - second

    if rgb_img.dtype != np.uint8:
        rgb = (np.clip(rgb_img,0,1)*255).astype(np.uint8)
    else:
        rgb = rgb_img
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    edges = cv2.Canny(g, 50, 150)
    edge_ratio = float(edges.mean())

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[...,1]; v = hsv[...,2]
    sat_std = float(s.std())
    v_range = float(v.max() - v.min())
    ent = _entropy(g)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    skin_pct = float((skin>0).mean())

    is_plain_skin = (skin_pct > 0.40 and lap_var < 50.0 and edge_ratio < 0.006 and sat_std < 12.0 and v_range < 35.0 and ent < 4.5)
    if is_plain_skin:
        return True, "looks like normal skin (very uniform)"

    if (conf < 0.70 or margin < 0.20) and ((lap_var < 30.0 and edge_ratio < 0.010) or skin_pct < 0.05):
        return True, "weak signal or background"

    if heatmap is not None:
        h = heatmap.astype(np.float32)
        if h.ndim == 3:
            h = h.mean(axis=-1)
        rng = float(h.ptp())
        if rng > 0:
            h = (h - float(h.min())) / (rng + 1e-6)
        thr = float(np.quantile(h, 0.80))
        mask = (h >= thr).astype(np.uint8)
        area = mask.sum() / mask.size * 100.0
        if area < 1.0 or area > 45.0:
            return True, "unfocused explanation"

    return False, ""
