"""Reusable Streamlit presentation components."""

from __future__ import annotations

import streamlit as st

STEPS = ["Source", "Profile", "Bronze STTM", "Bronze", "Silver STTM", "Silver", "Gold STTM", "Report"]


def inject_styles(light_mode: bool = False) -> None:
    if light_mode:
        colors = "--ink:#18343a;--muted:#547078;--panel:rgba(255,255,255,.83);--line:rgba(19,65,73,.16);--wash:#edf5f4;"
        background = "radial-gradient(ellipse at 52% -20%, #d0f1e9 0, transparent 45%), linear-gradient(145deg,#eef4f2,#f7f6ef 58%,#e9f1ef)"
    else:
        colors = "--ink:#e9f1f2;--muted:#9cb2b7;--panel:rgba(19,31,39,.78);--line:rgba(174,213,212,.15);--wash:#0a1117;"
        background = "radial-gradient(ellipse at 52% -20%, rgba(24,116,107,.34), transparent 47%), linear-gradient(145deg,#0a1117,#101c24 58%,#0b151c)"
    st.markdown(
        f"""
        <style>
        :root{{{colors}}}
        .stApp{{background:{background};color:var(--ink)}}
        [data-testid="stHeader"]{{background:transparent}}
        [data-testid="stSidebar"]{{background:var(--wash)}}
        .block-container{{max-width:1600px;padding-top:1.25rem;padding-bottom:3rem}}
        h1,h2,h3{{font-family:'Bahnschrift','Segoe UI Variable',sans-serif;letter-spacing:0;color:var(--ink)}}
        .idamp-eyebrow{{font:600 11px 'Consolas',monospace;letter-spacing:1px;color:#55d6be;text-transform:uppercase}}
        .hero{{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:10px;padding:26px 30px;margin:0 0 18px;background:linear-gradient(112deg,rgba(28,113,103,.28),rgba(19,31,39,.74) 54%,rgba(110,92,45,.18));box-shadow:0 18px 60px rgba(0,0,0,.15)}}
        .hero:after{{content:'';position:absolute;inset:0;background:repeating-linear-gradient(115deg,transparent 0 40px,rgba(255,255,255,.018) 41px 42px);pointer-events:none}}
        .hero h1{{font-size:34px;margin:4px 0 4px}}.hero p{{color:var(--muted);margin:0;max-width:720px}}
        .glass{{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:14px 16px;margin:0 0 12px;backdrop-filter:blur(14px)}}
        .feature{{height:100%;min-height:76px;border-left:2px solid #55d6be;padding:10px 12px;background:rgba(100,190,177,.06)}}
        .feature b{{display:block;color:var(--ink);font-size:13px}}.feature span{{color:var(--muted);font-size:12px}}
        .stepper{{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:5px;margin:10px 0 20px}}
        .step{{position:relative;border-top:2px solid var(--line);padding:8px 4px 0;color:var(--muted);font:11px 'Consolas',monospace;min-height:40px}}
        .step.done{{border-color:#55d6be;color:var(--ink)}}.step.active{{border-color:#f2bb63;color:var(--ink);animation:pulse 1.8s infinite}}
        @keyframes pulse{{50%{{filter:brightness(1.55)}}}}
        .kpi{{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:14px 16px;min-height:88px;animation:rise .4s ease both}}
        .kpi small{{color:var(--muted);text-transform:uppercase;font:10px 'Consolas',monospace}}.kpi strong{{display:block;font:25px 'Bahnschrift','Segoe UI Variable',sans-serif;margin-top:7px}}
        @keyframes rise{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
        .quote{{border-left:2px solid #55d6be;padding:10px 12px;color:var(--muted);font-style:italic;background:rgba(85,214,190,.05)}}
        .audit-event{{border-left:2px solid #55d6be;padding:7px 10px;margin:7px 0;background:rgba(255,255,255,.025);font-size:12px}}
        .muted{{color:var(--muted)}}
        [data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:7px}}
        div[data-testid="stButton"]>button{{border-radius:6px}}
        @media(max-width:900px){{.stepper{{grid-template-columns:repeat(4,minmax(0,1fr))}}.hero h1{{font-size:27px}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(active: int, completed: set[int] | None = None) -> None:
    completed = completed or set()
    columns = st.columns(len(STEPS), gap="small")
    for index, label in enumerate(STEPS):
        with columns[index]:
            if index in completed and index != active:
                if st.button(f"{index + 1:02d} {label}", key=f"step_rewind_{index}", help=f"Rewind to {label}"):
                    st.session_state.rewind_confirmation = index
            else:
                state = "active" if index == active else ""
                st.markdown(f'<div class="step {state}">{index + 1:02d} &nbsp;{label}</div>', unsafe_allow_html=True)
