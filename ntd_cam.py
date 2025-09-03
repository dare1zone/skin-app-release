import numpy as np, cv2, tensorflow as tf
from tensorflow import keras

def build_grad_model(model, target_layer="block_13_expand_relu"):
    try:
        lyr = model.get_layer(target_layer)
        return keras.Model(model.input, [lyr.output, model.output])
    except Exception:
        pass
    inner = None
    for L in model.layers:
        if hasattr(L, "get_layer") and hasattr(L, "layers"):
            try:
                lyr = L.get_layer(target_layer)
                return keras.Model(model.input, [lyr.output, model.output])
            except Exception:
                if inner is None and len(getattr(L, "layers", [])) > 20:
                    inner = L
    for L in reversed(model.layers):
        try:
            shp = L.output.shape
            if len(shp) == 4:
                return keras.Model(model.input, [L.output, model.output])
        except Exception:
            pass
    raise RuntimeError("No 4D feature map found for Grad-CAM")

def gradcam_uint8(x_pre, grad_model, class_index, out_size):
    with tf.GradientTape() as tape:
        conv, preds = grad_model(x_pre, training=False)
        loss = preds[:, int(class_index)]
    grads = tape.gradient(loss, conv)
    w = tf.reduce_mean(grads, axis=(1,2), keepdims=True)
    cam = tf.reduce_sum(w * conv, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    m = cam.max()
    if m > 1e-6:
        cam = cam / m
    heat = (cam * 255.0).astype("uint8")
    heat = cv2.resize(heat, (int(out_size[1]), int(out_size[0])), interpolation=cv2.INTER_CUBIC)
    return heat

def overlay(rgb_uint8, heat_uint8, alpha):
    if heat_uint8.ndim == 2:
        heat_rgb = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    else:
        heat_rgb = heat_uint8
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    if heat_rgb.shape[:2] != rgb_uint8.shape[:2]:
        heat_rgb = cv2.resize(heat_rgb, (rgb_uint8.shape[1], rgb_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(rgb_uint8, 1.0, heat_rgb, float(alpha), 0)
