from contextlib import contextmanager
import pathlib, streamlit as st

def apply_theme():
    try:
        css = pathlib.Path("styles.css").read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        pass
    st.markdown(
        """
        <div class="hero">
          <div style="display:flex;align-items:center;">
            <div class="logo">🩺</div>
            <div>
              <div class="hero-title">Skin Lesion Assistant</div>
              <div class="hero-sub">Educational prototype • clear explanations • safe abstention</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

@contextmanager
def card(title:str, subtitle:str=""):
    st.markdown(f'<div class="card"><div class="card-h"><span class="ct">{title}</span><span class="cs">{subtitle}</span></div>', unsafe_allow_html=True)
    yield
    st.markdown('</div>', unsafe_allow_html=True)

def badge(text:str, kind:str="ok"):
    kind = kind if kind in {"ok","warn","inconclusive"} else "ok"
    st.markdown(f'<span class="badge {kind}">{text}</span>', unsafe_allow_html=True)
