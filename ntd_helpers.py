import numpy as np, cv2
from PIL import Image, ImageOps
def auto_roi_center(rgb):
    h, w = rgb.shape[:2]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g,(5,5),0)
    _, th1 = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    k = np.ones((5,5),np.uint8)
    cands=[]
    for th in (th1, th2):
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, 1)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            x,y,wc,hc = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            area=(wc*hc)/(w*h+1e-6); cands.append((abs(area-0.2),(x,y,wc,hc)))
    if not cands: return rgb
    _,(x,y,wc,hc)=sorted(cands,key=lambda t:t[0])[0]
    pad=int(0.10*max(wc,hc)); x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w,x+wc+pad); y1=min(h,y+hc+pad)
    crop=rgb[y0:y1, x0:x1]; return crop if crop.size else rgb
def preprocess_pil_ntd(pil_im, tone_mode, preprocess_fn, IMG_SIZE, tone_calibrate):
    im = ImageOps.exif_transpose(pil_im).convert("RGB")
    base = np.array(im); base = tone_calibrate(base, tone_mode); base = auto_roi_center(base)
    im224 = Image.fromarray(base).resize(IMG_SIZE, Image.BILINEAR)
    x = np.array(im224).astype("float32"); x = preprocess_fn(x); x = np.expand_dims(x, 0)
    return np.array(im224), x
