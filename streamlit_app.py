"""Streamlit demo for Clinical Note Intelligence.

Run:  streamlit run streamlit_app.py
Uses the same `app/` package as the FastAPI service (no API server needed).
"""
import asyncio
import os

import pandas as pd
import streamlit as st

from app.config import settings
from app.constants import GROUPS, LABEL_TO_GROUP, THRESHOLD
from app.llm import LLMClient
from app.local_model import LocalClassifier
from app.pipeline import Pipeline
from app.text import clean_note_text

st.set_page_config(page_title="Clinical Note Intelligence", page_icon="🩺", layout="wide")

SAMPLES = {
    "Operative note (knee arthroscopy)": (
        "PREOPERATIVE DIAGNOSIS: Right knee medial meniscus tear. PROCEDURE: Arthroscopic partial medial "
        "meniscectomy. The patient was brought to the operating room, general anesthesia was induced, and the knee "
        "was prepped and draped in the usual sterile fashion. A 52-year-old male."),
    "Radiology report (CT abdomen)": (
        "CT ABDOMEN AND PELVIS WITH CONTRAST. FINDINGS: The liver, spleen and pancreas are unremarkable. "
        "No free fluid. IMPRESSION: No acute intra-abdominal process."),
    "SOAP / progress note (hypertension)": (
        "SUBJECTIVE: The patient returns for follow-up of hypertension and reports no chest pain. "
        "OBJECTIVE: BP 138/84, heart rate 72. ASSESSMENT: Hypertension, controlled. "
        "PLAN: Continue lisinopril 10 mg daily, recheck in 3 months."),
    "Pediatric well visit": (
        "This 5-year-old female presents for a well-child check and school physical. Vision and hearing screening "
        "were normal. MMR, DTaP and IPV vaccines were given today. Development is appropriate for age."),
}


# ------------------------------- helpers ------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_local() -> LocalClassifier:
    return LocalClassifier(settings.model_dir)


def secret_key() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "") or settings.groq_api_key or ""
    except Exception:
        return os.getenv("GROQ_API_KEY", "") or settings.groq_api_key or ""


def make_pipeline(api_key: str, model: str) -> Pipeline:
    # A fresh pipeline/client per run: asyncio.run() creates a new event loop each time.
    llm = LLMClient(api_key, model) if api_key else None
    return Pipeline(get_local(), llm, settings.llm_concurrency)


def readable(note: str) -> bool:
    return any(ch.isalpha() for ch in clean_note_text(note))


async def _analyze(note, mode, do_extract, api_key, model):
    pipe = make_pipeline(api_key, model)
    cls = await pipe.classify(note, mode)
    ext = await pipe.extract(note) if (do_extract and pipe.llm) else None
    return cls, ext


async def _batch(notes, mode, api_key, model):
    return await make_pipeline(api_key, model).classify_batch(notes, mode)


# -------------------------------- sidebar -----------------------------------
with st.sidebar:
    st.title("🩺 Clinical Note Intelligence")
    st.caption("Demo on de-identified MTSamples notes. **Not for clinical use.**")
    st.divider()

    api_key = st.text_input("Groq API key (optional)", value=secret_key(), type="password",
                            help="Enables LLM classification and structured extraction.")
    llm_model = st.text_input("Groq model", value=settings.groq_model)
    has_llm = bool(api_key)

    mode = st.radio(
        "Classification mode",
        ["local", "llm"],
        format_func=lambda m: {"local": "Local model (TF-IDF + LogReg)",
                               "llm": "Groq LLM (falls back to local)"}[m],
        help="In the notebook the local model scored higher than the LLM (micro-F1 0.917 vs 0.885), "
             "so it is the default.",
    )
    if mode == "llm" and not has_llm:
        st.warning("No API key: the local model will be used.")
        mode = "local"

    do_extract = st.toggle("Extract structured fields (LLM)", value=has_llm, disabled=not has_llm)
    st.divider()
    st.caption(f"Decision threshold: **{THRESHOLD}** · Groups: **{len(GROUPS)}**")

tab_single, tab_batch, tab_about = st.tabs(["📝 Single note", "📂 Batch (CSV)", "ℹ️ About & results"])

# ------------------------------- single note --------------------------------
with tab_single:
    choice = st.selectbox("Load a sample note", ["(write your own)"] + list(SAMPLES))
    default = SAMPLES.get(choice, "")
    note = st.text_area("Clinical note", value=default, height=220, key=f"note_{choice}",
                        placeholder="Paste a clinical transcription here...")

    if st.button("Analyze", type="primary"):
        if not readable(note):
            st.error("The note is empty or has no readable text.")
        elif len(note) > settings.max_note_length:
            st.error(f"The note exceeds {settings.max_note_length:,} characters.")
        else:
            with st.spinner("Analyzing..."):
                cls, ext = asyncio.run(_analyze(note, mode, do_extract, api_key, llm_model))

            st.subheader("Classification")
            c1, c2, c3 = st.columns(3)
            c1.metric("Top group", cls.groups[0])
            c2.metric("Confidence", f"{cls.confidence:.0%}")
            c3.metric("Source", cls.source)
            if cls.source == "fallback":
                st.warning("The LLM failed after retries, so the local model answered.")
            st.write("**Predicted groups:** " + " · ".join(f"`{g}`" for g in cls.groups))

            if cls.scores:
                df = pd.DataFrame({"probability": cls.scores}).sort_values("probability")
                st.bar_chart(df, horizontal=True)
                st.caption(f"Groups with probability ≥ {THRESHOLD} are returned; the top group is always included.")

            if ext is not None:
                extraction, used_fallback = ext
                st.subheader("Structured extraction")
                if used_fallback:
                    st.warning("Extraction failed after retries; showing an empty result.")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Age", "-" if extraction.patient_age is None else f"{extraction.patient_age:g}")
                e2.metric("Sex", extraction.patient_sex)
                e3.metric("Note type", extraction.note_type)
                e4.metric("Anesthesia", {True: "Yes", False: "No", None: "Not stated"}[extraction.anesthesia_used])
                x1, x2, x3 = st.columns(3)
                for col, title, items in ((x1, "Procedures", extraction.procedures),
                                          (x2, "Diagnoses", extraction.diagnoses),
                                          (x3, "Medications", extraction.medications)):
                    col.markdown(f"**{title}**")
                    col.write("\n".join(f"- {i}" for i in items) if items else "_none_")
                with st.expander("Raw JSON"):
                    st.json(extraction.model_dump())
            elif do_extract is False and not has_llm:
                st.info("Add a Groq API key in the sidebar to also extract age, sex, procedures, diagnoses and medications.")

# --------------------------------- batch ------------------------------------
with tab_batch:
    st.write(f"Upload a CSV with a text column (up to {settings.max_batch_size} rows are processed).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        text_cols = [c for c in data.columns if pd.api.types.is_string_dtype(data[c])]
        if not text_cols:
            st.error("No text column found in this file.")
        else:
            guess = next((c for c in ("transcription", "note", "text") if c in text_cols), text_cols[0])
            col = st.selectbox("Column with the notes", text_cols, index=text_cols.index(guess))
            subset = data.head(settings.max_batch_size).copy()
            st.caption(f"{len(data):,} rows in file; processing the first {len(subset)}.")
            if st.button("Classify batch", type="primary"):
                notes = subset[col].fillna("").astype(str).tolist()
                ok = [i for i, n in enumerate(notes) if readable(n)]
                with st.spinner("Classifying..."):
                    res = asyncio.run(_batch([notes[i] for i in ok], mode, api_key, llm_model))
                out = subset.copy()
                out["predicted_groups"] = ""
                out["confidence"] = None
                out["source"] = ""
                for i, r in zip(ok, res):
                    out.loc[out.index[i], ["predicted_groups", "confidence", "source"]] = [
                        "; ".join(r.groups), round(r.confidence, 3), r.source]
                skipped = len(notes) - len(ok)
                if skipped:
                    st.warning(f"{skipped} empty/unreadable row(s) were skipped.")
                st.dataframe(out, width="stretch")
                top = out["predicted_groups"].str.split("; ").str[0].replace("", pd.NA).dropna().value_counts()
                if not top.empty:
                    st.bar_chart(top)
                st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                                   "classified_notes.csv", "text/csv")

# --------------------------------- about ------------------------------------
with tab_about:
    st.markdown("""
### What this does
Routes clinical transcriptions into **11 note groups** and extracts structured fields with an LLM.

**Why groups?** In MTSamples only 2,357 of 4,966 rows are unique notes, and **91%** of those carry 2+ labels
(2.11 on average). The 40 specialties were clustered into 11 groups (Ward clustering + a permutation test:
within-group similarity 0.089 vs 0.042 at the 99.9th percentile of random groupings), and the task is multi-label.

### Results (354 held-out unique notes)
""")
    st.table(pd.DataFrame({
        "Model": ["TF-IDF + Logistic Regression (deployed)"],
        "hit@1": [0.966], "hit@3": [1.000], "micro-F1": [0.896], "macro-F1": [0.855]}).set_index("Model"))
    st.markdown("""
On a shared 100-note sample the baseline also beat the LLM (micro-F1 **0.917** vs **0.885** async, 0.818 sequential),
so classification uses the local model by default and the LLM is reserved for structured extraction.
Async batching was 3.5x faster than sequential LLM calls.

### Reliability layers
Pydantic schema validation · retries with exponential backoff · fallback to the local model ·
prompt-injection hardening · input guardrails.

### The 11 groups
""")
    st.dataframe(pd.DataFrame({
        "Group": GROUPS,
        "Source labels": [", ".join(sorted(l for l, g in LABEL_TO_GROUP.items() if g == grp)) for grp in GROUPS],
    }), hide_index=True, width="stretch")
    st.caption("Code: github.com/abdodedo20005-hash/clinical-note-intelligence")
