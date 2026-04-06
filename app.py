"""
app.py — AI Data Science Agent  ·  Web UI

Run:  streamlit run app.py
"""

import os
import sys
import json
import base64
import shutil
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Data Science Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — glassmorphism dark theme ────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #060d1f 0%, #0b1a35 55%, #07111f 100%);
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(10,20,40,0.8); }
section[data-testid="stMain"] > div { padding-top: 1rem; }

/* ── typography ──────────────────────────────────────────────────────────── */
h1 { font-size: 2.4rem !important; font-weight: 700 !important;
     background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     background-clip: text; line-height: 1.2 !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; color: #94a3b8 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #cbd5e1 !important; }

/* ── glass card ──────────────────────────────────────────────────────────── */
.glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 28px; margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.glass-blue {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 18px; padding: 28px; margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(59,130,246,0.1);
}

/* ── metric cards ────────────────────────────────────────────────────────── */
.metric-row { display: flex; gap: 16px; margin: 12px 0; flex-wrap: wrap; }
.metric {
    flex: 1; min-width: 140px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 20px 16px; text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #60a5fa; line-height: 1.1; }
.metric-label { font-size: 0.78rem; color: #94a3b8; margin-top: 6px; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── stage progress list ─────────────────────────────────────────────────── */
.stage { display: flex; align-items: center; gap: 10px; padding: 7px 0;
         border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.93rem; }
.stage:last-child { border-bottom: none; }
.dot-ok   { width: 8px; height: 8px; border-radius: 50%; background: #10b981; flex-shrink:0; }
.dot-fail { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; flex-shrink:0; }
.dot-run  { width: 8px; height: 8px; border-radius: 50%; background: #f59e0b;
            animation: pulse 1s infinite; flex-shrink:0; }
.dot-wait { width: 8px; height: 8px; border-radius: 50%; background: #334155; flex-shrink:0; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.stage-label-ok   { color: #e2e8f0; }
.stage-label-fail { color: #fca5a5; }
.stage-label-run  { color: #fde68a; }
.stage-label-wait { color: #64748b; }

/* ── insight chips ───────────────────────────────────────────────────────── */
.insight {
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px; padding: 12px 16px; margin: 6px 0; font-size: 0.93rem; color: #c7d2fe;
}
.reco {
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25);
    border-radius: 10px; padding: 12px 16px; margin: 6px 0; font-size: 0.93rem; color: #6ee7b7;
}

/* ── buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.6rem 2.2rem !important;
    font-weight: 600 !important; font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.55) !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}

/* ── file uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(96,165,250,0.35) !important;
    border-radius: 14px !important; padding: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.65) !important;
}

/* ── text input / area ───────────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}

/* ── divider ─────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── status widget ───────────────────────────────────────────────────────── */
[data-testid="stStatusWidget"] { background: rgba(255,255,255,0.04) !important; }

/* ── image captions ──────────────────────────────────────────────────────── */
.stImage figcaption { color: #64748b !important; font-size: 0.78rem !important; }

/* ── tab strip ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; border-radius: 8px !important; }
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.25) !important; color: #e2e8f0 !important;
}

/* ── alert/info boxes ────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── pipeline stage definitions ────────────────────────────────────────────────
STAGES = [
    ("data_quality",                "Cleaning & validating data"),
    ("documentation",               "Documenting dataset"),
    ("eda",                         "Exploratory analysis"),
    ("feature_engineering",         "Engineering features"),
    ("statistical_analysis",        "Running statistical tests"),
    ("model_architecture",          "Selecting best model"),
    ("hyperparameter_optimization", "Optimising hyperparameters"),
    ("model_validation",            "Validating model"),
    ("insight_synthesis",           "Synthesising insights"),
    ("visualization",               "Creating visualisation charts"),
    ("final_report",                "Generating PDF report"),
]

ACCEPTED_TYPES = ["csv", "xlsx", "xls", "json", "parquet", "tsv"]
MAX_MB = 500


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in ["context", "data/raw", "data/cleaned", "data/processed",
              "output/reports", "output/pipeline_results", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _seed_context(question: str, raw_file: Path):
    from src.core.context_manager import write_context
    ctx = {
        "project_metadata": {
            "research_question": question,
            "created_at": datetime.now().isoformat(),
            "dataset_refs": [],
        },
        "context_chain": {
            "dataset_discovery": {"status": "skipped"},
            "data_acquisition": {
                "status": "success",
                "download_timestamp": datetime.now().isoformat(),
                "total_datasets": 1, "successful_downloads": 1, "failed_downloads": 0,
                "file_paths": [raw_file.name],  # relative to data/raw/
                "file_types": [raw_file.suffix.lstrip(".") or "csv"],
            },
            **{k: {} for k in ["data_quality","documentation","eda",
                               "feature_engineering","statistical_analysis",
                               "model_architecture","hyperparameter_optimization",
                               "model_validation","insight_synthesis",
                               "visualization","final_report"]},
        },
        "pipeline_log": [{
            "agent": "app.py",
            "timestamp": datetime.now().isoformat(),
            "message": f"Web UI analysis started: {question}",
        }],
    }
    write_context(ctx)


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _collect_charts() -> list[Path]:
    dirs = ["output/eda_01", "output/visualization_01"]
    charts = []
    for d in dirs:
        charts += sorted(Path(d).glob("*.png")) if Path(d).exists() else []
    return charts


def _read_context() -> dict:
    from src.core.context_manager import read_context
    return read_context()


def _generate_html_report(ctx: dict, charts: list[Path]) -> str:
    meta   = ctx.get("project_metadata", {})
    chain  = ctx.get("context_chain", {})
    q      = meta.get("research_question", "")
    ts     = meta.get("created_at", "")[:19].replace("T", " ")
    dq     = chain.get("data_quality", {})
    mv     = chain.get("model_validation", {})
    sa     = chain.get("statistical_analysis", {})
    ins    = chain.get("insight_synthesis", {})

    def _metric_section():
        metrics = ""
        for fp, m in mv.get("validation_metrics", {}).items():
            for k, v in m.items():
                if isinstance(v, float):
                    metrics += f"""
                    <div class="metric">
                        <div class="val">{v:.4f}</div>
                        <div class="lbl">{k.replace('_',' ').title()}</div>
                    </div>"""
        return f'<div class="metric-row">{metrics}</div>' if metrics else ""

    def _chart_section():
        html = '<div class="chart-grid">'
        for p in charts[:12]:
            b64 = _img_b64(p)
            caption = p.stem.replace("_", " ").title()
            html += f'<div class="chart-item"><img src="data:image/png;base64,{b64}" alt="{caption}"><p>{caption}</p></div>'
        html += "</div>"
        return html

    def _findings():
        items = ins.get("key_findings", [])
        return "".join(f"<li>{i}</li>" for i in items) if items else "<li>See attached charts for details.</li>"

    def _recos():
        items = ins.get("recommendations", [])
        return "".join(f"<li>{i}</li>" for i in items) if items else ""

    def _stat_table():
        rows = ""
        for t in sa.get("significant_tests", [])[:20]:
            v1 = t.get("variable1") or t.get("dependent_variable", "")
            v2 = t.get("variable2") or t.get("independent_variable", "")
            p  = t.get("p_value")
            ef = t.get("effect_size")
            p_s  = f"{p:.4f}"  if isinstance(p,  float) else str(p or "")
            ef_s = f"{ef:.4f}" if isinstance(ef, float) else str(ef or "N/A")
            rows += f"<tr><td>{t.get('test_type','')}</td><td>{v1}</td><td>{v2}</td><td>{p_s}</td><td>{ef_s}</td></tr>"
        if not rows:
            return ""
        return f"""<h2>Significant Statistical Findings</h2>
        <table><thead><tr><th>Test</th><th>Variable 1</th><th>Variable 2</th><th>p-value</th><th>Effect Size</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    cleaned = dq.get("successful_cleanings", 0)
    total   = dq.get("total_files", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Science Report — {q[:60]}</title>
<style>
:root{{--blue:#3b82f6;--indigo:#6366f1;--green:#10b981;--bg:#060d1f;--card:rgba(255,255,255,.04);--border:rgba(255,255,255,.09);}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter','Segoe UI',system-ui,sans-serif;background:linear-gradient(135deg,#060d1f,#0b1a35,#07111f);
  color:#e2e8f0;min-height:100vh;padding:40px 24px;}}
.container{{max-width:1100px;margin:0 auto}}
header{{text-align:center;padding:40px 0 32px}}
header h1{{font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#60a5fa,#a78bfa,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
header p{{color:#94a3b8;margin-top:10px;font-size:1.05rem}}
.badge{{display:inline-block;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);
  border-radius:20px;padding:4px 14px;font-size:.8rem;color:#93c5fd;margin-top:8px}}
section{{background:var(--card);backdrop-filter:blur(18px);border:1px solid var(--border);
  border-radius:18px;padding:28px;margin:20px 0;box-shadow:0 8px 32px rgba(0,0,0,.3)}}
h2{{font-size:1.2rem;font-weight:700;color:#cbd5e1;margin-bottom:16px;
  padding-bottom:8px;border-bottom:1px solid var(--border)}}
.metric-row{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.metric{{flex:1;min-width:130px;background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);
  border-radius:12px;padding:18px;text-align:center}}
.val{{font-size:1.9rem;font-weight:700;color:#60a5fa}}
.lbl{{font-size:.72rem;color:#94a3b8;margin-top:6px;text-transform:uppercase;letter-spacing:.06em}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;margin-top:16px}}
.chart-item{{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:12px;
  padding:12px;text-align:center}}
.chart-item img{{width:100%;border-radius:8px;display:block}}
.chart-item p{{font-size:.75rem;color:#64748b;margin-top:6px}}
ul{{padding-left:20px;line-height:2}}
li{{color:#c7d2fe;margin:4px 0}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{background:rgba(99,102,241,.2);color:#a5b4fc;padding:10px 12px;text-align:left;font-weight:600}}
td{{padding:9px 12px;border-bottom:1px solid var(--border);color:#cbd5e1}}
tr:hover td{{background:rgba(255,255,255,.03)}}
footer{{text-align:center;padding:32px 0;color:#334155;font-size:.8rem}}
</style>
</head>
<body>
<div class="container">
<header>
  <h1>🔬 AI Data Science Report</h1>
  <p>{q}</p>
  <span class="badge">Generated {ts}</span>
</header>

<section>
  <h2>📋 Dataset Summary</h2>
  <p>Files successfully cleaned: <strong>{cleaned} / {total}</strong></p>
</section>

<section>
  <h2>📊 Model Performance</h2>
  {_metric_section()}
</section>

{f'<section><h2>📈 Charts & Visualisations</h2>{_chart_section()}</section>' if charts else ''}

<section>
  <h2>💡 Key Findings</h2>
  <ul>{_findings()}</ul>
</section>

{f'<section><h2>✅ Recommendations</h2><ul>{_recos()}</ul></section>' if ins.get("recommendations") else ''}

<section>
  {_stat_table()}
</section>

<footer>Generated by AI Data Science Agent · {ts}</footer>
</div>
</body>
</html>"""


# ── pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(progress_placeholder, log_file: Path) -> tuple[bool, Optional[Path]]:
    """
    Run all 11 analysis stages. Updates the progress_placeholder in real-time.
    Returns (success, pdf_path).
    """
    from src.agents.data_quality_agent import DataQualityAgent
    from src.agents.documentation_agent import DocumentationAgent
    from src.agents.eda_agent_simple import EDAAgent
    from src.agents.feature_engineering_agent_simple import FeatureEngineeringAgent
    from src.agents.statistical_analysis_agent_simple import StatisticalAnalysisAgent
    from src.agents.model_architecture_agent_simple import ModelArchitectureAgent
    from src.agents.hyperparameter_optimization_agent_simple import HyperparameterOptimizationAgent
    from src.agents.model_validation_agent_simple import ModelValidationAgent
    from src.agents.insight_synthesis_agent_simple import InsightSynthesisAgent
    from src.agents.visualization_agent_simple import VisualizationAgent
    from src.agents.final_report_generator import FinalReportGenerator

    agent_classes = [
        DataQualityAgent, DocumentationAgent, EDAAgent,
        FeatureEngineeringAgent, StatisticalAnalysisAgent,
        ModelArchitectureAgent, HyperparameterOptimizationAgent,
        ModelValidationAgent, InsightSynthesisAgent,
        VisualizationAgent, FinalReportGenerator,
    ]

    statuses = ["wait"] * len(STAGES)   # wait | run | ok | fail
    pdf_path = None
    any_critical_fail = False

    def _render():
        html = '<div class="glass" style="padding:20px">'
        for i, (_, label) in enumerate(STAGES):
            s = statuses[i]
            dot   = f'<div class="dot-{s}"></div>'
            lbl   = f'<span class="stage-label-{s}">{label}</span>'
            extra = ""
            if s == "ok":   extra = '<span style="color:#10b981;font-size:.8rem;margin-left:auto">done</span>'
            if s == "fail": extra = '<span style="color:#ef4444;font-size:.8rem;margin-left:auto">failed</span>'
            if s == "run":  extra = '<span style="color:#f59e0b;font-size:.8rem;margin-left:auto">running…</span>'
            html += f'<div class="stage">{dot}{lbl}{extra}</div>'
        html += "</div>"
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    _render()

    for i, ((name, label), cls) in enumerate(zip(STAGES, agent_classes)):
        statuses[i] = "run"
        _render()
        try:
            result = cls().execute()
            statuses[i] = "ok"
            if name == "final_report" and isinstance(result, dict):
                rp = result.get("report_path")
                if rp:
                    pdf_path = Path(rp)
        except Exception as e:
            statuses[i] = "fail"
            with open(log_file, "a") as f:
                f.write(f"\n[{name}] FAILED: {e}\n{traceback.format_exc()}\n")
            # Non-critical stages: continue; critical early stages: stop
            if name in ("data_quality", "feature_engineering"):
                any_critical_fail = True
        _render()

    # Fallback: scan for any PDF produced
    if not pdf_path:
        pdfs = sorted(Path("output/reports").glob("*.pdf"))
        if pdfs:
            pdf_path = pdfs[-1]

    return (not any_critical_fail), pdf_path


# ── main UI ───────────────────────────────────────────────────────────────────

def main():
    _ensure_dirs()

    # silence loggers that spam stdout and pollute Streamlit
    for name in ["transformers","sentence_transformers","torch","PIL",
                 "matplotlib","kaggle","urllib3","httpx"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown("# 🔬 AI Data Science Agent")
    st.markdown("## Turn any dataset into a full research report — no code needed")
    st.markdown("---")

    # ── input section ─────────────────────────────────────────────────────────
    col_up, col_q = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("### 📂 Upload Dataset")
        st.caption("CSV, Excel, JSON, Parquet, TSV  ·  max 500 MB")
        uploaded = st.file_uploader(
            label="Drag & drop or click to browse",
            type=ACCEPTED_TYPES,
            label_visibility="collapsed",
        )
        if uploaded:
            size_mb = len(uploaded.getvalue()) / 1_048_576
            if size_mb > MAX_MB:
                st.error(f"File is {size_mb:.0f} MB — limit is {MAX_MB} MB.")
                uploaded = None
            else:
                st.success(f"✓ **{uploaded.name}**  ·  {size_mb:.1f} MB")

    with col_q:
        st.markdown("### ❓ Research Question")
        st.caption("Plain English — ask anything about your data")
        question = st.text_area(
            label="question",
            placeholder=(
                "e.g. What factors best predict survival?\n"
                "e.g. Is there a significant correlation between age and income?\n"
                "e.g. Which features drive customer churn?"
            ),
            height=148,
            label_visibility="collapsed",
        )

    st.markdown("")
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("▶  Run Analysis", use_container_width=True,
                            disabled=(not uploaded or not question or not question.strip()))

    # ── run pipeline ──────────────────────────────────────────────────────────
    if run_btn and uploaded and question.strip():

        # Save uploaded file to data/raw/
        raw_path = Path("data/raw") / uploaded.name
        raw_path.write_bytes(uploaded.getvalue())

        _seed_context(question.strip(), raw_path)

        log_file = Path("logs/app.log")
        log_file.parent.mkdir(exist_ok=True)
        log_file.write_text(f"Run started {datetime.now().isoformat()}\n"
                            f"File: {uploaded.name}\nQuestion: {question.strip()}\n\n")

        st.markdown("---")
        st.markdown("### ⚙️ Pipeline Progress")
        progress_ph = st.empty()

        success, pdf_path = run_pipeline(progress_ph, log_file)

        # ── results ───────────────────────────────────────────────────────────
        st.markdown("---")
        ctx    = _read_context()
        chain  = ctx.get("context_chain", {})
        mv     = chain.get("model_validation", {})
        sa     = chain.get("statistical_analysis", {})
        ins    = chain.get("insight_synthesis", {})
        charts = _collect_charts()

        if success:
            st.success("✅ Analysis complete!")
        else:
            st.warning("⚠️ Analysis finished with some errors — results may be partial.")

        # ── metric cards ──────────────────────────────────────────────────────
        val_metrics = mv.get("validation_metrics", {})
        flat_metrics: dict = {}
        for m in val_metrics.values():
            flat_metrics.update(m)

        sig_count = len(sa.get("significant_tests", []))

        metric_items = []
        for k in ["accuracy","f1_score","r2_score","mean_absolute_error"]:
            if k in flat_metrics:
                metric_items.append((k.replace("_"," ").title(), flat_metrics[k]))
        if sig_count:
            metric_items.append(("Significant Tests", sig_count))

        if metric_items:
            st.markdown("### 📊 Key Metrics")
            cols = st.columns(min(len(metric_items), 5))
            for col, (label, val) in zip(cols, metric_items):
                with col:
                    fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                    st.markdown(f"""
                    <div class="metric">
                        <div class="metric-value">{fmt}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("")

        # ── tabs: charts / insights / stats ───────────────────────────────────
        tab_charts, tab_insights, tab_stats = st.tabs(
            ["📈 Charts", "💡 Insights & Recommendations", "🔬 Statistical Findings"]
        )

        with tab_charts:
            if charts:
                cols = st.columns(2)
                for idx, p in enumerate(charts[:12]):
                    with cols[idx % 2]:
                        st.image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)
            else:
                st.info("No charts generated.")

        with tab_insights:
            findings = ins.get("key_findings", [])
            recos    = ins.get("recommendations", [])
            if findings:
                st.markdown("**Key Findings**")
                for f in findings:
                    st.markdown(f'<div class="insight">💡 {f}</div>', unsafe_allow_html=True)
            if recos:
                st.markdown("<br>**Recommendations**", unsafe_allow_html=True)
                for r in recos:
                    st.markdown(f'<div class="reco">✅ {r}</div>', unsafe_allow_html=True)
            if not findings and not recos:
                st.info("Insights not available — check the PDF report.")

        with tab_stats:
            sig_tests = sa.get("significant_tests", [])
            if sig_tests:
                import pandas as pd
                rows = []
                for t in sig_tests[:30]:
                    v1 = t.get("variable1") or t.get("dependent_variable", "")
                    v2 = t.get("variable2") or t.get("independent_variable", "")
                    p  = t.get("p_value")
                    ef = t.get("effect_size")
                    rows.append({
                        "Test Type":   t.get("test_type", ""),
                        "Variable 1":  v1,
                        "Variable 2":  v2,
                        "p-value":     f"{p:.4f}"  if isinstance(p,  float) else str(p or ""),
                        "Effect Size": f"{ef:.4f}" if isinstance(ef, float) else str(ef or "N/A"),
                        "Significant": "✓" if t.get("significant") else "✗",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No significant statistical tests recorded.")

        # ── download buttons ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⬇️ Download Report")
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 4])

        with dl_col1:
            if pdf_path and pdf_path.exists():
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.button("📄 PDF unavailable", disabled=True, use_container_width=True)

        with dl_col2:
            html_report = _generate_html_report(ctx, charts)
            fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            st.download_button(
                label="🌐 Download HTML",
                data=html_report.encode(),
                file_name=fname,
                mime="text/html",
                use_container_width=True,
            )

    # ── empty-state footer ────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="glass" style="text-align:center;padding:48px 24px;margin-top:24px">
            <div style="font-size:3rem;margin-bottom:12px">🔬</div>
            <div style="font-size:1.15rem;color:#94a3b8;line-height:1.7">
                Upload any dataset and ask a question in plain English.<br>
                The AI pipeline will clean, analyse, model, and report — automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
