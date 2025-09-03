# Skin Lesion Assistant — General + NTD

Two educational image models wrapped in a Streamlit app:

- **General dermatology (7 classes)** — inference via a SavedModel wrapper with **Score-CAM**.
- **Neglected Tropical Diseases (3 classes: Buruli ulcer, leishmaniasis, leprosy)** — MobileNetV2 with **Grad-CAM**, guardrails, and **LIME**.

## Files needed in this folder
- app7.py
- labels7.json and ham10000_effnet7_cam/
- ntd_mbv2_mix_final.keras, ntd_labels.json
- requirements.lock.txt
- docs/  (optional screenshots)

## Quick start

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
streamlit run app7.py --server.port 8501


## How to use
1. Pick the model (7-class or NTD).
2. Upload an image (JPG/PNG; HEIC works if pillow-heif is installed).
3. See prediction, confidence/stability, CAM heatmap, and top-k table.
4. Optional LIME in the LIME tab.
5. Use Download CAM to save the overlay PNG.

## Troubleshooting
- HEIC support: `pip install pillow-heif`
- NumPy/TensorFlow mismatch: `pip install -r requirements.lock.txt`
- 7-class CAM wrapper present: `ham10000_effnet7_cam/saved_model.pb`

## Email setup 

To enable **Message a doctor** email sending, set these environment variables before running the app.  

```bash

export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="dareb516@gmail.com"
export SMTP_PASS="YOUR_16_CHAR_APP_PASSWORD"   # 16 chars from Google App Passwords
export SMTP_FROM="Skin Lesion Assistant <dareb516@gmail.com>"
export SMTP_TO="dareb516@gmail.com"            # or a comma-separated list

## Quick start

Python: use 3.11 .

```bash
# 1) Create & activate a Python 3.11 venv
# macOS (Homebrew python@3.11):
/opt/homebrew/bin/python3.11 -m venv .venv
# (Linux/Windows: just ensure `python3.11` is available)
source .venv/bin/activate

python -V             # should show 3.11.x
pip install -U pip setuptools wheel

# 2) Install deps
pip install -r requirements.lock.txt
# If you see a jaxlib error on 3.11:
#   pip install -r <(grep -v '^jax' requirements.lock.txt)
#   pip install "jax==0.4.34" "jaxlib==0.4.38"

# Apple Silicon only (if TF import errors):
# pip install "tensorflow-macos==2.16.1" "tensorflow-metal==1.1.0"

# 4)  LIME tab extras
pip install "scikit-image==0.24.0" "scikit-learn==1.4.2" "lime==0.2.0.1"

# 5) (Optional) Email sending for “Message a doctor”
# Create a Gmail App Password (16 chars) and export:
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your_gmail@gmail.com"
export SMTP_PASS="your_16_char_app_password"
export SMTP_FROM="Skin Lesion Assistant <your_gmail@gmail.com>"
export SMTP_TO="where_to_receive@example.com"

# 6) Run
streamlit run app7.py



## Disclaimer
Educational/triage support only; not a diagnosis.
