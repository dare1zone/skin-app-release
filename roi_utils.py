import numpy as np

def border_crop(img, pct=0.06):
    h, w = img.shape[:2]
    dy = int(h * pct)
    dx = int(w * pct)
    y0, y1 = dy, max(dy, h - dy)
    x0, x1 = dx, max(dx, w - dx)
    if y1 <= y0 or x1 <= x0:
        return img
    return img[y0:y1, x0:x1]

import cv2

def remove_edge_text(rgb_img, edge_pct=0.18):
    if rgb_img is None:
        return rgb_img
    if rgb_img.ndim == 2:
        rgb = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = rgb_img
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    my, mx = int(h*edge_pct), int(w*edge_pct)
    edge_mask = (np.zeros_like(gray, dtype=np.uint8))
    edge_mask[:my,:] = 255; edge_mask[-my:,:] = 255; edge_mask[:, :mx] = 255; edge_mask[:, -mx:] = 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k, iterations=2)
    bh = cv2.bitwise_and(bh, edge_mask)
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
    cleaned = cv2.inpaint(bgr, th, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

import cv2, numpy as np

def remove_edge_text2(rgb_img, edge_pct=0.28):
    if rgb_img is None:
        return rgb_img
    if rgb_img.ndim == 2:
        rgb = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = rgb_img

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    my, mx = int(h*edge_pct), int(w*edge_pct)

    edge_mask = np.zeros_like(gray, dtype=np.uint8)
    edge_mask[:my,:] = 255
    edge_mask[-my:,:] = 255
    edge_mask[:, :mx] = 255
    edge_mask[:, -mx:] = 255

   
    corner = np.zeros_like(gray, dtype=np.uint8)
    corner[-int(h*0.35):, :int(w*0.40)] = 255
    mask = cv2.bitwise_or(edge_mask, corner)

    k5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k5, iterations=2)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k5, iterations=2)
    cand = cv2.max(tophat, blackhat)
    cand = cv2.bitwise_and(cand, mask)

    _, th = cv2.threshold(cand, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)

    cleaned = cv2.inpaint(bgr, th, 4, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
import cv2, numpy as np

import cv2, numpy as np

def lesion_bbox(rgb):
    if rgb is None:
        return None
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

    # 1) Skin mask (YCrCb range) to ignore background/text
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
    if (skin>0).mean() < 0.10:
        skin = np.ones_like(skin)*255  # fallback if detector fails

    # 2) Lesion score = darker (low L) + redder (high 'a') inside skin
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[...,0].astype(np.float32)
    A = lab[...,1].astype(np.float32)
    Lb = cv2.GaussianBlur(L, (5,5), 0)
    Ab = cv2.GaussianBlur(A, (5,5), 0)
    L_dark = 255.0 - Lb
    A_centered = np.maximum(Ab - Ab[skin>0].mean(), 0.0)
    score = 0.6*L_dark + 0.4*A_centered
    score = (score * (skin>0)).astype(np.float32)

    # 3) Threshold + clean up
    sc8 = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(sc8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    area = (w*h) / float(rgb.shape[0]*rgb.shape[1])
    if area < 0.01 or area > 0.70:
        return None

    # 4) Small margin around the box
    H,W = rgb.shape[:2]
    pad = int(0.05*max(H,W))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return (y0, x0, y1, x1)

import cv2, numpy as np

def lesion_bbox_fallback(rgb):
    if rgb is None:
        return None
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[...,0]
    Lb = cv2.GaussianBlur(L, (5,5), 0)
    _, th = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    H,W = rgb.shape[:2]
    area = (w*h)/(H*W)
    if area < 0.003 or area > 0.90:
        return None
    pad = int(0.05*max(H,W))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return (y0, x0, y1, x1)


import cv2, numpy as np

def lesion_bbox_ranked(rgb):
    if rgb is None:
        return None
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
    if (skin>0).mean() < 0.10:
        skin = np.ones_like(skin)*255

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[...,0].astype(np.float32)
    A = lab[...,1].astype(np.float32)
    Lb = cv2.GaussianBlur(L, (5,5), 0)
    Ab = cv2.GaussianBlur(A, (5,5), 0)
    L_dark = 255.0 - Lb
    A_centered = np.maximum(Ab - Ab[skin>0].mean(), 0.0)
    score = 0.6*L_dark + 0.4*A_centered
    score = (score * (skin>0)).astype(np.float32)

    sc8 = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(sc8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H,W = rgb.shape[:2]
    cx, cy = W/2.0, H/2.0

    best = None
    best_rank = -1.0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = (w*h)/(H*W)
        if area < 0.005 or area > 0.70:
            continue
        mask = np.zeros((H,W), np.uint8); cv2.drawContours(mask,[c],-1,255,-1)
        mean_sc = float(score[mask>0].mean()) if (mask>0).any() else 0.0
        M = cv2.moments(c); ux = (M['m10']/M['m00']) if M['m00'] else x+w/2; uy = (M['m01']/M['m00']) if M['m00'] else y+h/2
        d = ((ux-cx)/W)**2 + ((uy-cy)/H)**2
        center_w = float(np.exp(-(d/0.20)))  # favour nearer the centre
        rank = (mean_sc) * (area**0.3) * center_w
        if rank > best_rank:
            best_rank = rank
            best = (y, x, y+h, x+w)

    if best is None:
        return None

    y0,x0,y1,x1 = best
    pad = int(0.05*max(H,W))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return (y0, x0, y1, x1)
