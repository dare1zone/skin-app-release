import io, os, json, base64, numpy as np, streamlit as st, tensorflow as tf
from PIL import Image, ImageOps
import cv2
from ntd_tight import tighten_cam

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

st.set_page_config(page_title="Skin Lesion Assistant — General + NTD", page_icon="🩺", layout="wide")
st.markdown("""
<style>
:root{--card-bg:rgba(255,255,255,0.04);--border:rgba(255,255,255,0.10)}
div.block-container{padding-top:1rem}
.card{padding:1rem;border:1px solid var(--border);border-radius:16px;background:var(--card-bg);margin-bottom:1rem}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-weight:700;font-size:.9rem}
.badge-ok{background:#DCFCE7;color:#166534}
.badge-mid{background:#FEF9C3;color:#713F12}
.badge-low{background:#FEE2E2;color:#7F1D1D}
.small{opacity:.75;font-size:.9rem}
.kv{display:flex;gap:.5rem;align-items:center}
.kv b{min-width:120px;display:inline-block}
</style>
""", unsafe_allow_html=True)

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

IMG_SIZE = (224, 224)

def load_json_list(path, fallback):
    try:
        with open(path, "r") as f:
            obj = json.load(f)
            if isinstance(obj, dict): 
                return list(obj.values())
            return list(obj)
    except Exception:
        return fallback

GENERAL_LABELS = load_json_list("labels7.json", [
    "Melanocytic nevus (common mole)",
    "Melanoma",
    "Benign keratosis (solar lentigo / seborrheic keratosis)",
    "Basal cell carcinoma",
    "Actinic keratosis / Bowen’s disease",
    "Vascular lesion",
    "Dermatofibroma"
])

NTD_LABELS = load_json_list("ntd_labels.json", [
    "Buruli ulcer",
    "leishmaniasis",
    "leprosy"
])

def open_uploaded(up):
    b = up.getvalue()
    try:
        return Image.open(io.BytesIO(b))
    except Exception:
        pass
    try:
        arr = np.frombuffer(b, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("decode_failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as e:
        raise e

@st.cache_resource(show_spinner=False)
def load_general_artifacts():
    from tensorflow.keras.applications.efficientnet import preprocess_input as pp
    layer = tf.keras.layers.TFSMLayer("ham10000_effnet7_cam", call_endpoint="serving_default")
    inp = tf.keras.Input(shape=IMG_SIZE+(3,))
    outs = layer(inp)
    conv_t = next(v for v in outs.values() if len(v.shape)==4)
    probs_t = next(v for v in outs.values() if len(v.shape)==2)
    cam_model = tf.keras.Model(inp, [conv_t, probs_t])
    return cam_model, GENERAL_LABELS, pp, "general"

@st.cache_resource(show_spinner=False)
def load_ntd_artifacts():
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp
    m = tf.keras.models.load_model("ntd_mbv2_roi.keras", compile=False)

    def _build_grad_model(model, target="block_13_expand_relu"):
        inner = None
        for L in model.layers:
            if hasattr(L, "layers") and len(getattr(L, "layers", [])) > 20:
                inner = L
                break
        if inner is None:
            inner = model
        feat = None
        try:
            feat = inner.get_layer(target).output
        except Exception:
            for L in reversed(getattr(inner, "layers", [])):
                try:
                    shp = getattr(L.output, "shape", None)
                    if hasattr(shp, "__len__") and len(shp) == 4:
                        feat = L.output
                        break
                except Exception:
                    pass
        if feat is None:
            for L in reversed(model.layers):
                try:
                    shp = getattr(L.output, "shape", None)
                    if hasattr(shp, "__len__") and len(shp) == 4:
                        feat = L.output
                        break
                except Exception:
                    pass
        if feat is None:
            raise RuntimeError("No 4D conv layer found for CAM.")
        return tf.keras.Model(model.input, [feat, model.output])

    grad_model = _build_grad_model(m, "block_13_expand_relu")
    return m, grad_model, NTD_LABELS, pp, "ntd"

def tone_calibrate(arr_uint8, mode):
    if mode == "none":
        return arr_uint8
    lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    if mode == "darker":
        L2 = np.clip(L2 * 0.85, 0, 255).astype(np.uint8)
    if mode == "brighter":
        L2 = np.clip(L2 * 1.10, 0, 255).astype(np.uint8)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def preprocess_pil(pil_im, tone_mode, preprocess_fn):
    im = ImageOps.exif_transpose(pil_im).convert("RGB")
    base = np.array(im)
    base = tone_calibrate(base, tone_mode)
    im224 = Image.fromarray(base).resize(IMG_SIZE, Image.BILINEAR)
    x = np.array(im224).astype("float32")
    x = preprocess_fn(x)
    x = np.expand_dims(x, 0)
    return np.array(im224), x

def gradcam_uint8(img_tensor, grad_model, class_index, out_size):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, int(class_index)]
    grads = tape.gradient(loss, conv_out)
    w = tf.reduce_mean(grads, axis=(1,2), keepdims=True)
    cam = tf.reduce_sum(w * conv_out, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-6)
    cam = cv2.resize(cam, out_size[::-1], interpolation=cv2.INTER_CUBIC)
    heat = (255*cam).astype("uint8")
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)

def scorecam_uint8(x_pre, cam_model, class_index, out_size, k=16):
    conv, probs = cam_model.predict(x_pre, verbose=0)
    idx = int(class_index)
    conv = conv[0]
    h, w, c = conv.shape
    means = conv.reshape(-1, c).mean(0)
    order = np.argsort(means)[::-1][:min(k, c)]
    heat = np.zeros((out_size[0], out_size[1]), dtype="float32")
    for ci in order:
        a = conv[..., int(ci)]
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
        mask = cv2.resize(a.astype("float32"), out_size[::-1], interpolation=cv2.INTER_CUBIC)
        x_mask = x_pre.copy()
        x_mask[0] = x_mask[0] * mask[..., None]
        _, p2 = cam_model.predict(x_mask, verbose=0)
        heat += float(p2[0, idx]) * mask
    heat -= heat.min()
    heat /= (heat.max() + 1e-6)
    heat = (255*heat).astype("uint8")
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)

def overlay_heat(rgb_uint8, heat_uint8, alpha):
    heat_rgb = cv2.cvtColor(heat_uint8, cv2.COLOR_BGR2RGB)
    if heat_rgb.shape[:2] != rgb_uint8.shape[:2]:
        heat_rgb = cv2.resize(heat_rgb, (rgb_uint8.shape[1], rgb_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(rgb_uint8, 1.0, heat_rgb, float(alpha), 0)

def conf_stability(pvec):
    p = np.asarray(pvec, dtype="float32")
    top = float(np.max(p))
    order = np.argsort(p)[::-1]
    top2 = float(p[order[1]]) if p.size > 1 else 0.0
    margin = top - top2
    ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    if top >= 0.75:
        conf_html = f'<span class="badge badge-ok">Confidence: {top*100:.1f}%</span>'
    elif top >= 0.5:
        conf_html = f'<span class="badge badge-mid">Confidence: {top*100:.1f}%</span>'
    else:
        conf_html = f'<span class="badge badge-low">Confidence: {top*100:.1f}%</span>'
    if margin >= 0.25 and ent <= 1.2:
        stab_html = '<span class="badge badge-ok">Stability: Stable</span>'
    elif margin >= 0.12:
        stab_html = '<span class="badge badge-mid">Stability: Borderline</span>'
    else:
        stab_html = '<span class="badge badge-low">Stability: Uncertain</span>'
    return conf_html, stab_html, order

def top_table(labels, pvec, k=5):
    order = np.argsort(pvec)[::-1][:min(k, len(labels))]
    names = [labels[i] for i in order]
    vals = [round(float(pvec[i])*100.0, 1) for i in order]
    return names, vals

st.title("Skin Lesion Assistant")
st.caption("General dermatology (7 classes) and Neglected Tropical Diseases (3 classes). Educational tool, not a medical diagnosis.")
model_choice = st.selectbox("Model", ["General — 7 classes", "NTD — 3 classes"], index=1, key="model_select")

if model_choice.startswith("General"):
    cam_model, CLASS_NAMES, preprocess_fn, mode_key = load_general_artifacts()
    model = None
    grad_model = None
else:
    model, grad_model, CLASS_NAMES, preprocess_fn, mode_key = load_ntd_artifacts()
    cam_model = None

nav = st.radio("Sections", ["Upload & Predict","Grad-CAM","LIME explanation","Helpbot","Find care","Message a doctor"], horizontal=True, key="nav_top")

if nav == "Upload & Predict":
    colL, colR = st.columns([1,1])
    with colL:
        st.subheader("Upload a dermatoscopic image")
        up = st.file_uploader("Choose image", type=["jpg","jpeg","png","heic","heif"], key="up_main")
        tone = st.selectbox("Skin tone calibration", ["none","darker","brighter"], index=0, key="tone_main")
        alpha = st.slider("CAM overlay strength", 0.0, 1.0, 0.35, 0.01, key="alpha_main")
    with colR:
        st.markdown("<div class='card'><b>How to read the result</b><ul><li>The model predicts a class and confidence.</li><li>Grad-CAM/Score-CAM highlights what influenced the prediction.</li><li>The table shows top alternatives.</li><li>Educational guidance only.</li></ul></div>", unsafe_allow_html=True)
    if up is not None:
        rgb224, x = preprocess_pil(open_uploaded(up), tone, preprocess_fn)
        if mode_key == "ntd":
            g = cv2.cvtColor(rgb224, cv2.COLOR_RGB2GRAY)
            lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
            edge_ratio = float((cv2.Canny(g, 30, 90) > 0).mean())
            if lap_var < 35.0 and edge_ratio < 0.020:
                st.info(f"Inconclusive (background gate) · lap_var={lap_var:.1f} thr<15.0 · edge_ratio={edge_ratio:.3f} thr<0.010")
                st.stop()
            p = model.predict(x, verbose=0)[0]
            order_tmp = np.argsort(p)[::-1]
            conf = float(p[order_tmp[0]])
            margin = float(conf - (p[order_tmp[1]] if len(p) > 1 else 0.0))
            if conf < 0.50 or margin < 0.10:
                st.info(f"Inconclusive (decision gate) · conf={conf:.2f} thr>=0.50 · margin={margin:.2f} thr>=0.10")
                st.stop()
            conf_html, stab_html, order = conf_stability(p)
            pred_idx = int(order[0])
            heat = gradcam_uint8(x, grad_model, pred_idx, IMG_SIZE)
            blend = overlay_heat(rgb224, heat, alpha)
            st.markdown(conf_html + " " + stab_html, unsafe_allow_html=True)
            c1, c2 = st.columns([1,1])
            with c1:
                st.image(rgb224, caption="Input (224×224)", width=320, use_container_width=False)
            with c2:
                st.image(blend, caption=f"Grad-CAM — {CLASS_NAMES[pred_idx]}", width=320, use_container_width=False)
                buf = io.BytesIO()
                Image.fromarray(blend).save(buf, format="PNG")
                fname = "gradcam_" + CLASS_NAMES[pred_idx].replace(" ", "_") + ".png"
                st.download_button("Download CAM", data=buf.getvalue(), file_name=fname, mime="image/png", key="dl_cam_ntd")
            names, vals = top_table(CLASS_NAMES, p, k=len(CLASS_NAMES))
            st.dataframe({"Class": names, "Confidence %": vals}, hide_index=True, use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(blend).save(buf, format="PNG"); buf.seek(0)
            st.download_button("Download CAM (PNG)", data=buf, file_name="ntd_gradcam.png", mime="image/png")
        else:
            conv, probs = cam_model.predict(x, verbose=0)
            p = probs[0]
            conf_html, stab_html, order = conf_stability(p)
            pred_idx = int(order[0])
            heat = scorecam_uint8(x, cam_model, pred_idx, IMG_SIZE, k=16)
            blend = overlay_heat(rgb224, heat, alpha)
            st.markdown(conf_html + " " + stab_html, unsafe_allow_html=True)
            c1, c2 = st.columns([1,1])
            with c1:
                st.image(rgb224, caption="Input (224×224)", width=320, use_container_width=False)
            with c2:
                st.image(blend, caption=f"Score-CAM — {CLASS_NAMES[pred_idx]}", width=320, use_container_width=False)
                buf = io.BytesIO()
                Image.fromarray(blend).save(buf, format="PNG")
                fname = "scorecam_" + CLASS_NAMES[pred_idx].replace(" ", "_") + ".png"
                st.download_button("Download CAM", data=buf.getvalue(), file_name=fname, mime="image/png", key="dl_cam_general")
            names, vals = top_table(CLASS_NAMES, p, k=len(CLASS_NAMES))
            st.dataframe({"Class": names, "Confidence %": vals}, hide_index=True, use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(blend).save(buf, format="PNG"); buf.seek(0)
            st.download_button("Download CAM (PNG)", data=buf, file_name="general_scorecam.png", mime="image/png")

elif nav == "Grad-CAM":
    st.subheader("Grad-CAM / Score-CAM viewer")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png","heic","heif"], key="up_cam")
    tone = st.selectbox("Skin tone calibration", ["none","darker","brighter"], index=0, key="tone_cam")
    alpha = st.slider("Overlay strength", 0.0, 1.0, 0.35, 0.01, key="alpha_cam")
    target = st.selectbox("Class to visualize", CLASS_NAMES, key="cam_target")
    if up is not None:
        rgb224, x = preprocess_pil(open_uploaded(up), tone, preprocess_fn)
        t_idx = CLASS_NAMES.index(target)
        if mode_key == "ntd":
            heat = gradcam_uint8(x, grad_model, t_idx, IMG_SIZE)
            blend = overlay_heat(rgb224, heat, alpha)
            st.image(blend, caption=f"Grad-CAM — {target}", width=360, use_container_width=False)
        else:
            heat = scorecam_uint8(x, cam_model, t_idx, IMG_SIZE, k=16)
            blend = overlay_heat(rgb224, heat, alpha)
            st.image(blend, caption=f"Score-CAM — {target}", width=360, use_container_width=False)
        buf = io.BytesIO()
        Image.fromarray(blend).save(buf, format="PNG"); buf.seek(0)
        st.download_button("Download CAM (PNG)", data=buf, file_name="cam.png", mime="image/png")

elif nav == "LIME explanation":
    st.subheader("LIME explanation")
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png","heic","heif"], key="up_lime")
    tone = st.selectbox("Skin tone calibration", ["none","darker","brighter"], index=0, key="tone_lime")
    label_to_explain = st.selectbox("Class to explain", CLASS_NAMES, key="lime_target")
    num_feat = st.slider("Number of regions", 4, 20, 8, 1, key="lime_num")
    if up is not None:
        try:
            from lime import lime_image
            from skimage.segmentation import slic, mark_boundaries
            base, x = preprocess_pil(open_uploaded(up), tone, preprocess_fn)
            base_arr = base.astype(np.uint8)
            def classifier_fn(imgs):
                arrs = []
                for a in imgs:
                    a_uint8 = (a*255 if a.max()<=1.0 else a).astype(np.uint8)
                    arrs.append(np.array(Image.fromarray(a_uint8).resize(IMG_SIZE)))
                X = np.stack(arrs, 0).astype("float32")
                X = preprocess_fn(X)
                if mode_key == "ntd":
                    return model.predict(X, verbose=0)
                else:
                    _, probs = cam_model.predict(X, verbose=0)
                    return probs
            segments = slic(base_arr, n_segments=180, compactness=12, sigma=1, start_label=1)
            explainer = lime_image.LimeImageExplainer()
            exp = explainer.explain_instance(base_arr, classifier_fn=classifier_fn, top_labels=1, num_samples=800, hide_color=0, segmentation_fn=lambda x: segments)
            lab = CLASS_NAMES.index(label_to_explain)
            _, mask = exp.get_image_and_mask(lab, positive_only=True, num_features=int(num_feat), hide_rest=False)
            outlined = mark_boundaries(base_arr/255.0, mask, color=(1,1,0), mode="thick")
            st.image((outlined*255).astype(np.uint8), caption=f"LIME — {label_to_explain}", width=360, use_container_width=False)
        except Exception as e:
            st.error("Install extras: lime, scikit-image")
elif nav == "Helpbot":
    st.subheader("Helpbot")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h3>🚩 When to seek care</h3><ul><li><b>Urgent:</b> rapid growth, bleeding, very dark/irregular</li><li><b>Soon:</b> changing spot, new symptoms, or you are worried</li><li><b>Routine:</b> stable and harmless-appearing</li></ul></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>📸 Great photos</h3><ul><li>Clean the lens</li><li>Bright, even lighting</li><li>Focus sharply</li><li>Fill the frame</li></ul></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>🔍 Dermatoscopic image</h3><p>A polarized close-up that reduces glare and shows sub-surface structures.</p></div>", unsafe_allow_html=True)
    with st.expander("More tips"):
        st.markdown("<div class='card'><div class='kv'><b>Distance</b><span>Take one close-up and one mid-range view</span></div><div class='kv'><b>Background</b><span>Plain, non-reflective background</span></div><div class='kv'><b>Stability</b><span>If changing over weeks, arrange a review</span></div></div>", unsafe_allow_html=True)
    faq = st.selectbox("Common questions", ["What is this tool for?","Does this diagnose skin cancer?","How accurate is it?","What if I’m concerned right now?"], index=0, key="helpbot_faq")
    if faq == "What is this tool for?":
        st.info("Education and triage support. It suggests possibilities and highlights image regions (Grad-CAM/Score-CAM). It is not a diagnosis.")
    elif faq == "Does this diagnose skin cancer?":
        st.warning("No. Only a clinician can diagnose. Use this as guidance, not a final answer.")
    elif faq == "How accurate is it?":
        st.write("Accuracy varies by class and image quality. Confidence and stability badges summarize how decisive a prediction was.")
    else:
        st.write("If you are worried about a lesion that is changing, bleeding, painful, or rapidly growing, arrange an in-person evaluation soon. If severe or urgent, seek emergency care.")
elif nav == "Find care":
    st.subheader("Find care")
    st.caption("Search nearby services. External map links will open in a new tab.")

    with st.form("findcare"):
        col1, col2 = st.columns([2,1])
        with col1:
            city = st.text_input("City or region", key="care_city")
            specialty = st.selectbox(
                "Specialty",
                ["Dermatologist", "Infectious disease", "Tropical medicine", "Primary care clinic", "Skin clinic"],
                index=0, key="care_spec"
            )
        with col2:
            open_now = st.checkbox("Open now", value=False, key="care_open")
            walk_in  = st.checkbox("Walk-in",  value=False, key="care_walkin")
            pediatric= st.checkbox("Pediatric", value=False, key="care_ped")
        submit = st.form_submit_button("Build search links")

    if submit:
        q = specialty.lower()
        filters = []
        if open_now:  filters.append("open now")
        if walk_in:   filters.append("walk-in")
        if pediatric: filters.append("pediatric")
        if filters:
            q = q + " " + " ".join(filters)

        loc = city.strip()
        if not loc:
            loc = "near me"

        if loc == "near me":
            base_q = q + " " + loc
        else:
            base_q = q + " near " + loc

        gq = base_q.replace(" ", "+")
        gmaps_url = f"https://www.google.com/maps/search/{gq}"
        st.markdown(f"<div class='card'><b>Google Maps</b><br><a href='{gmaps_url}' target='_blank'>Open search</a></div>", unsafe_allow_html=True)

        apple_q = base_q.replace(" ", "+")
        apple_url = f"https://maps.apple.com/?q={apple_q}"
        st.markdown(f"<div class='card'><b>Apple Maps</b><br><a href='{apple_url}' target='_blank'>Open search</a></div>", unsafe_allow_html=True)

    st.markdown("<div class='card small'><b>Tips:</b> Try adding a neighborhood or hospital name to narrow results. Call ahead to confirm availability.</div>", unsafe_allow_html=True)

elif nav == "Message a doctor":
    st.subheader("Message a doctor")
    st.caption("Not for emergencies. If severe symptoms or rapid changes, seek urgent care.")
    with st.form("msg_doc"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", key="msg_name")
            contact = st.text_input("Email or phone", key="msg_contact")
            city = st.text_input("City / Region", key="msg_city")
        with c2:
            subj = st.text_input("Subject", key="msg_subj")
            up2 = st.file_uploader("Attach an image (optional)", type=["jpg","jpeg","png","heic","heif"], key="msg_up")
        msg = st.text_area("Message", height=120, key="msg_text")
        ok = st.checkbox("I consent to be contacted about this message.", key="msg_ok")
        sent = st.form_submit_button("Send message")
        if sent:
            if not ok or not contact.strip():
                st.error("Please provide contact info and consent.")
            else:
                import time
                rec = {"ts": int(time.time()), "name": name, "contact": contact, "city": city, "subject": subj, "message": msg}
                if up2 is not None:
                    try:
                        rec["image_name"] = up2.name
                        rec["image_mime"] = up2.type
                        rec["image_b64"] = base64.b64encode(up2.getvalue()).decode("ascii")
                    except Exception:
                        pass
                try:
                    with open("messages_outbox.jsonl","a",encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    st.success("Saved locally to messages_outbox.jsonl")
                except Exception as e:
                    st.warning(f"Could not write messages_outbox.jsonl: {e}")
                draft = f"""To: {contact}
Subject: {subj or 'Skin concern'}

Dear {name or 'Patient'},

Thank you for your message. A clinician will review your note from {city or 'your location'} and reply as soon as possible.

- Clinic"""
                st.code(draft, language="text")
