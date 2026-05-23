"""Streamlit UI: upload files, pick mode, view structured + cited report."""
from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multimodal Analyst", layout="wide")
st.title("Multimodal Financial & Legal Analyst")
st.caption("Upload PDFs, Excel sheets, and chart images. Get a cited, guardrailed report.")

with st.sidebar:
    mode = st.radio("Mode", ["investment", "legal"], horizontal=True)
    topic = st.text_area("Topic / Question (optional)", height=100)
    submit = st.button("Analyze", type="primary", use_container_width=True)

uploads = st.file_uploader(
    "Upload artifacts",
    type=["pdf", "xlsx", "xls", "xlsm", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if submit:
    if not uploads:
        st.error("Please upload at least one file.")
        st.stop()

    files = [("files", (u.name, u.getvalue(), u.type or "application/octet-stream")) for u in uploads]
    data = {"mode": mode, "topic": topic}
    with st.spinner("Running multimodal analysis…"):
        try:
            r = requests.post(f"{API_URL}/v1/analyze", files=files, data=data, timeout=600)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            st.error(f"Request failed: {e}")
            st.stop()

    payload = r.json()
    report = payload["report"]
    guard = payload["guardrails"]

    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Guardrails")
        st.metric("Schema", "PASS" if guard["schema_ok"] else "FAIL")
        st.metric("Citations", "PASS" if guard["citations_ok"] else "FAIL")
        st.metric("Grounding", "PASS" if guard["grounding_ok"] else "FAIL")
        st.metric("Numeric", "PASS" if guard["numeric_ok"] else "FAIL")
        if payload.get("repaired"):
            st.info("Report was auto-repaired by the agent.")
        if guard["issues"]:
            with st.expander(f"{len(guard['issues'])} issue(s)"):
                for i in guard["issues"]:
                    st.write(f"- {i}")

    with col1:
        if mode == "investment":
            st.header(report.get("company") or "Investment Thesis")
            st.markdown(f"**Recommendation:** `{report.get('recommendation')}`  ·  "
                        f"Confidence: `{report.get('overall_confidence', 0):.2f}`")
            st.write(report.get("summary", ""))

            if report.get("key_metrics"):
                st.subheader("Key Metrics")
                for m in report["key_metrics"]:
                    cites = ", ".join(c["source_id"] for c in m["citations"])
                    st.markdown(f"- **{m['name']}**: {m['value']} "
                                f"({m.get('period') or 'n/a'}) — _cites: {cites}_")

            for label in ("strengths", "risks", "catalysts"):
                items = report.get(label) or []
                if items:
                    st.subheader(label.title())
                    for c in items:
                        gs = c.get("grounding_score")
                        flags = c.get("flags") or []
                        badge = f"🟢 {gs:.2f}" if gs and gs >= 0.6 else (f"🟡 {gs:.2f}" if gs else "—")
                        flag_txt = (" ⚠ " + ", ".join(flags)) if flags else ""
                        st.markdown(f"- {c['statement']}  \n  *{badge}{flag_txt}*")
                        with st.expander("citations"):
                            for cit in c["citations"]:
                                st.caption(f"`{cit['source_id']}`")
                                st.write(f"> {cit['quote']}")
        else:
            st.header(report.get("matter") or "Legal Risk Report")
            st.markdown(f"**Overall risk:** `{report.get('overall_risk')}`  ·  "
                        f"Confidence: `{report.get('overall_confidence', 0):.2f}`")
            st.write(report.get("summary", ""))

            if report.get("risks"):
                st.subheader("Risks")
                for r_ in report["risks"]:
                    st.markdown(f"### {r_['title']} — `{r_['severity']}`")
                    st.write(r_["description"])
                    if r_.get("mitigation"):
                        st.caption(f"Mitigation: {r_['mitigation']}")
                    with st.expander("citations"):
                        for cit in r_["citations"]:
                            st.caption(f"`{cit['source_id']}`")
                            st.write(f"> {cit['quote']}")

            if report.get("obligations"):
                st.subheader("Obligations")
                for c in report["obligations"]:
                    st.markdown(f"- {c['statement']}")

            if report.get("open_questions"):
                st.subheader("Open Questions")
                for q in report["open_questions"]:
                    st.markdown(f"- {q}")

    with st.expander("Raw JSON"):
        st.code(json.dumps(payload, indent=2), language="json")
