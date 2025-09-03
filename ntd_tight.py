import cv2, numpy as np
def tighten_cam(heat_bgr_u8, keep_q=0.85, min_area_frac=0.002):
    h, w = heat_bgr_u8.shape[:2]
    heat_gray = cv2.cvtColor(heat_bgr_u8, cv2.COLOR_BGR2GRAY)
    thr = np.percentile(heat_gray, 100*(1.0 - keep_q))
    bin_mask = (heat_gray >= thr).astype("uint8")
    num, labels = cv2.connectedComponents(bin_mask)
    if num <= 1:
        return heat_bgr_u8, 0.0
    iy, ix = np.unravel_index(int(np.argmax(heat_gray)), heat_gray.shape)
    peak_lbl = labels[iy, ix]
    tight = (labels == peak_lbl).astype("uint8")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    tight = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, k)
    if tight.sum() < min_area_frac * h * w:
        return heat_bgr_u8, float(tight.mean())
    mask3 = np.dstack([tight]*3).astype("float32")
    heat = heat_bgr_u8.astype("float32") * mask3
    m = heat.max()
    if m > 0:
        heat = (255.0 * heat / m).astype("uint8")
    else:
        heat = heat_bgr_u8
    return heat, float(tight.mean())
