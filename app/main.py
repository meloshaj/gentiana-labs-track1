"""DecoctionGraph Streamlit dashboard (Gentiana Labs | EthnoHack 2026 Track 1).

Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.extraction import extract_from_text, mock_extract_hypericum_sample
from src.graph_builder import DecoctionGraphBuilder
from src.lotus_retriever import enrich_taxon_compounds
from src.schemas import EthnoPreparationRecord, TaxonResolution
from src.taxonomy import resolve_plant_name

TITLE = "🌿 DecoctionGraph: Preparation-Aware Ethnobotanical Knowledge Mining"
TEAM_BADGE = "Gentiana Labs | EthnoHack 2026 Track 1"
DISCLAIMER = (
    "⚠️ Research & Hypothesis-Generation Prototype Only: Outputs reflect automated "
    "literature mining and first-order physicochemical estimations, not clinical "
    "advice or approved medical interventions."
)

MODE_PRESET = "Preloaded Case Studies"
MODE_RAW = "Raw Text Mining (Custom Text)"

LOTUS_PATH = ROOT / "data" / "reference" / "lotus_curated_sample.json"
DEMO_CITATION = "EthnoHack 2026 curated benchmark passage (illustrative demo text)"

STATUS_COLORS = {
    "High": ("#1B5E20", "#C8E6C9"),
    "Moderate": ("#7A5B00", "#FFF3C4"),
    "Low": ("#4E342E", "#EFEBE9"),
    "Excluded": ("#424242", "#E0E0E0"),
    "TOXIC ALERT": ("#FFFFFF", "#C62828"),
}


# ==============================================================================
# Case study definitions
# ==============================================================================

def _rec(**kw: Any) -> Dict[str, Any]:
    return EthnoPreparationRecord(**kw).model_dump(mode="json")


def _case_records(case: str) -> Tuple[List[Dict[str, Any]], str]:
    """Return (records, citation) for a preloaded case study."""
    if case == "Hypericum perforatum":
        batch = mock_extract_hypericum_sample(DEMO_CITATION, "Balkans / Mediterranean")
        return [r.model_dump(mode="json") for r in batch.records], DEMO_CITATION

    if case == "Gentiana lutea":
        return [
            _rec(
                verbatim_plant_name="Gentiana lutea", plant_part="root", preparation_type="decoction",
                solvent_class="water", thermal_regime="boiling_prolonged",
                verbatim_indication="stomach weakness, loss of appetite and intermittent fever",
                administration_route="oral",
                verbatim_quote="The dried yellow gentian root is boiled in spring water for an hour and the bitter decoction drunk before meals to strengthen a weak stomach and break intermittent fever.",
                cultural_or_geographic_origin="Alpine Herbalism", extraction_confidence=0.96,
                ambiguity_notes="Boiling for one hour mapped to boiling_prolonged.",
            ),
            _rec(
                verbatim_plant_name="Gentiana lutea", plant_part="root", preparation_type="tincture_alcohol",
                solvent_class="hydroalcoholic", thermal_regime="cold_ambient",
                verbatim_indication="stomach weakness, loss of appetite and intermittent fever",
                administration_route="oral",
                verbatim_quote="[No source quote: modelled comparator preparation for contrast with the decoction.]",
                cultural_or_geographic_origin="Alpine Herbalism", extraction_confidence=0.30,
                ambiguity_notes="Hypothetical comparator added by the dashboard; not extracted from the source text.",
            ),
        ], DEMO_CITATION

    if case == "Artemisia annua":
        ind = "intermittent fevers"
        return [
            _rec(
                verbatim_plant_name="Artemisia annua", plant_part="aerial_parts", preparation_type="juice_wringing",
                solvent_class="water", thermal_regime="cold_ambient", verbatim_indication=ind,
                administration_route="oral",
                verbatim_quote="The fresh green herb is pounded and wrung through cloth without heat, and the raw cold juice is drunk to drive out fevers.",
                cultural_or_geographic_origin="Traditional Chinese Medicine", extraction_confidence=0.94,
                ambiguity_notes="Cold wringing of fresh herb mapped to juice_wringing / cold_ambient.",
            ),
            _rec(
                verbatim_plant_name="Artemisia annua", plant_part="aerial_parts", preparation_type="decoction",
                solvent_class="water", thermal_regime="boiling_prolonged", verbatim_indication=ind,
                administration_route="oral",
                verbatim_quote="Later practice simmered the dried herb in water for a long while, but the physicians noted the strong boiled draught served the fever poorly.",
                cultural_or_geographic_origin="Traditional Chinese Medicine", extraction_confidence=0.90,
                ambiguity_notes="Prolonged simmering mapped to boiling_prolonged.",
            ),
        ], DEMO_CITATION

    if case == "Symphytum officinale":
        return [
            _rec(
                verbatim_plant_name="Symphytum officinale", plant_part="root", preparation_type="poultice",
                solvent_class="none", thermal_regime="none",
                verbatim_indication="bruises, sprains and broken bones", administration_route="topical",
                verbatim_quote="The fresh grated comfrey root is bound over bruises and sprains as a poultice, to knit the bone and ease swelling.",
                cultural_or_geographic_origin="European folk medicine", extraction_confidence=0.95,
                ambiguity_notes="Poultice applies plant matter directly; no solvent filter is modelled.",
            ),
            _rec(
                verbatim_plant_name="Symphytum officinale", plant_part="leaf", preparation_type="infusion",
                solvent_class="water", thermal_regime="warm_steep",
                verbatim_indication="stomach ulcers and coughs", administration_route="oral",
                verbatim_quote="The leaves are steeped in hot water and the tea drunk for ulcers of the stomach and for coughs.",
                cultural_or_geographic_origin="European folk medicine", extraction_confidence=0.92,
                ambiguity_notes="Internal use of comfrey carries a documented pyrrolizidine alkaloid toxicity concern.",
            ),
        ], DEMO_CITATION
    raise KeyError(case)


CASE_STUDIES: Dict[str, str] = {
    "Hypericum perforatum (Oil Maceration vs. Water Tea)": "Hypericum perforatum",
    "Gentiana lutea (Boiling Root Decoction)": "Gentiana lutea",
    "Artemisia annua (Cold Wringing vs. Prolonged Boiling)": "Artemisia annua",
    "Symphytum officinale (Topical Poultice vs. Internal Tea & PA Toxicity Warning)": "Symphytum officinale",
}


# ==============================================================================
# Cached computation
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_case(case: str) -> Dict[str, Any]:
    records, citation = _case_records(case)
    return {"records": records, "citation": citation}


@st.cache_data(show_spinner=False)
def known_taxa() -> List[str]:
    return list(json.loads(LOTUS_PATH.read_text(encoding="utf-8")).keys())


def _heuristic_extract(text: str, citation: str, region: str) -> List[Dict[str, Any]]:
    """Offline keyword fallback used only when no LLM key is configured (low confidence)."""
    low = text.lower()
    plant = next((t for t in known_taxa() if t.lower() in low), None)
    if plant is None:
        plant = next((t for t in known_taxa() if t.split()[0].lower() in low), None)
    if plant is None:
        return []
    if re.search(r"\b(oil|olive)\b", low):
        solvent, prep = "lipid_oil", "oil_maceration"
    elif re.search(r"\b(alcohol|wine|brandy|spirit|tincture)\b", low):
        solvent, prep = "hydroalcoholic", "tincture_alcohol"
    elif re.search(r"\bvinegar\b", low):
        solvent, prep = "vinegar_acidic", "infusion"
    elif re.search(r"\bpoultice\b|\bbound over\b|\bcompress\b", low):
        solvent, prep = "none", "poultice"
    elif re.search(r"\b(water|tea|boil|decoction|infus|steep)", low):
        solvent, prep = "water", "decoction" if "boil" in low or "decoct" in low else "infusion"
    else:
        solvent, prep = "unknown", "unknown"
    if re.search(r"\b(boil|simmer|decoct)", low):
        thermal = "boiling_prolonged"
    elif re.search(r"\b(steep|warm|sun)", low):
        thermal = "warm_steep"
    elif re.search(r"\b(cold|raw|ambient)\b", low):
        thermal = "cold_ambient"
    else:
        thermal = "none" if solvent == "none" else "unknown"
    route = "topical" if re.search(r"\b(topical|applied|poultice|wound|burn|skin|bound)", low) else "oral"
    part = next((p for p in ("root", "bark", "leaf", "flower", "seed", "fruit", "rhizome") if p in low or p + "s" in low), "unknown")
    quote = text.strip().split(".")[0].strip() + "."
    return [_rec(
        verbatim_plant_name=plant, plant_part=part, preparation_type=prep, solvent_class=solvent,
        thermal_regime=thermal, verbatim_indication="Not determined (offline keyword mode)",
        administration_route=route, verbatim_quote=quote, cultural_or_geographic_origin=region or "Unspecified",
        extraction_confidence=0.35,
        ambiguity_notes="Offline keyword heuristic; configure LLM_API_KEY for structured LLM extraction.",
    )]


@st.cache_data(show_spinner="Mining preparation records from text...")
def mine_text(text: str, citation: str, region: str, use_llm: bool) -> Dict[str, Any]:
    """Extract records from custom text. Returns records plus the method used."""
    if use_llm:
        try:
            batch = extract_from_text(text, citation, region or "Unspecified")
            return {"records": [r.model_dump(mode="json") for r in batch.records], "method": "llm", "error": ""}
        except Exception as exc:  # network / auth / schema failures
            return {
                "records": _heuristic_extract(text, citation, region),
                "method": "heuristic",
                "error": f"LLM extraction failed ({type(exc).__name__}); showing offline keyword result instead.",
            }
    return {"records": _heuristic_extract(text, citation, region), "method": "heuristic", "error": ""}


@st.cache_data(show_spinner="Resolving taxonomy...")
def resolve_taxon(name: str) -> Dict[str, Any]:
    try:
        return resolve_plant_name(name, timeout=5.0).model_dump(mode="json")
    except Exception:
        return TaxonResolution(
            queried_name=name, accepted_scientific_name=name, taxon_rank="unknown",
            taxonomic_status="UNRESOLVED", gbif_usage_key=None, confidence=0.0,
            match_type="NONE", is_resolved=False,
        ).model_dump(mode="json")


@st.cache_data(show_spinner="Scoring phytochemicals (RDKit)...")
def analyze_records(records_json: str) -> List[Dict[str, Any]]:
    """For each preparation record, resolve the taxon and score LOTUS compounds."""
    out: List[Dict[str, Any]] = []
    for rec in json.loads(records_json):
        taxon = resolve_taxon(rec["verbatim_plant_name"])
        compounds = enrich_taxon_compounds(
            taxon["accepted_scientific_name"], rec["solvent_class"], rec["thermal_regime"],
            gbif_usage_key=taxon.get("gbif_usage_key"),
        )
        if not compounds:  # taxon string may carry authorship the curated set lacks
            compounds = enrich_taxon_compounds(
                rec["verbatim_plant_name"], rec["solvent_class"], rec["thermal_regime"]
            )
        out.append({"record": rec, "taxon": taxon, "compounds": [c.model_dump(mode="json") for c in compounds]})
    return out


def _inline_local_assets(html: str) -> str:
    """Inline pyvis' locally-referenced lib/ assets so the HTML is self-contained in an iframe."""
    def read(rel: str) -> Optional[str]:
        p = ROOT / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def script(m: "re.Match[str]") -> str:
        body = read(m.group(1))
        return f"<script>{body}</script>" if body is not None else m.group(0)

    def css(m: "re.Match[str]") -> str:
        body = read(m.group(1))
        return f"<style>{body}</style>" if body is not None else m.group(0)

    html = re.sub(r'<script[^>]*\ssrc="(lib/[^"]+)"[^>]*></script>', script, html)
    html = re.sub(r'<link[^>]*\shref="(lib/[^"]+\.css)"[^>]*/?>', css, html)
    return html


@st.cache_data(show_spinner="Building knowledge graph...")
def build_graph_html(records_json: str, threshold: float, show_excluded: bool) -> Tuple[str, int, int]:
    """Build the PyVis network and return (html, node_count, edge_count)."""
    analyses = analyze_records(records_json)
    builder = DecoctionGraphBuilder()
    for a in analyses:
        from src.schemas import PhytochemicalRecord  # noqa: F401  (typing aid)
        taxon = TaxonResolution(**a["taxon"])
        builder.add_extraction_record(EthnoPreparationRecord(**a["record"]), taxon_resolution=taxon)
    sub = None if show_excluded else builder.get_filtered_subgraph(min_plausibility=threshold)
    target = sub if sub is not None else builder.graph
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)  # pyvis may drop a lib/ folder in the working dir
        try:
            path = builder.to_pyvis(str(Path(tmp) / "graph.html"), subgraph=sub, min_plausibility=threshold)
            html = Path(path).read_text(encoding="utf-8")
        finally:
            os.chdir(cwd)
    return _inline_local_assets(html), target.number_of_nodes(), target.number_of_edges()


# ==============================================================================
# Data shaping
# ==============================================================================

def status_for(c: Dict[str, Any], threshold: float, flag_toxic: bool) -> str:
    if flag_toxic and c["toxicity_alerts"]:
        return "TOXIC ALERT"
    score, verdict = c["extraction_plausibility_score"], c["extraction_verdict"]
    if score < threshold or verdict == "excluded":
        return "Excluded"
    return {"high": "High", "moderate": "Moderate"}.get(verdict, "Low")


def prep_label(rec: Dict[str, Any]) -> str:
    return (f"{rec['preparation_type'].replace('_', ' ').title()} "
            f"({rec['solvent_class'].replace('_', ' ')}, {rec['thermal_regime'].replace('_', ' ')})")


def build_frame(analysis: Dict[str, Any], threshold: float, flag_toxic: bool, show_excluded: bool) -> pd.DataFrame:
    rows = []
    for c in analysis["compounds"]:
        st_ = status_for(c, threshold, flag_toxic)
        rows.append({
            "Compound Name": c["compound_name"], "SMILES": c["smiles"], "MW": c["mw"],
            "cLogP": c["clogp"], "TPSA": c["tpsa"],
            "Plausibility Score": c["extraction_plausibility_score"], "Status": st_,
            "Toxicity Flags": ", ".join(c["toxicity_alerts"]) if flag_toxic else "",
            "Below Threshold": c["extraction_plausibility_score"] < threshold,
        })
    df = pd.DataFrame(rows, columns=[
        "Compound Name", "SMILES", "MW", "cLogP", "TPSA", "Plausibility Score", "Status", "Toxicity Flags", "Below Threshold"])
    if not show_excluded and not df.empty:
        df = df[~df["Below Threshold"]]
    return df.reset_index(drop=True)


def scatter(df: pd.DataFrame, title: str, ranges: Dict[str, List[float]]) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        sizeref = 2.0 * max(ranges["mw_max"], 1.0) / (44.0 ** 2)
        hover = [
            f"<b>{n}</b><br>Plausibility: {s:.2f}<br>MW: {m:.1f}<br>cLogP: {l:.2f} | TPSA: {t:.1f}<br>"
            f"Toxicity flags: {f if f else 'none flagged'}"
            for n, s, m, l, t, f in zip(df["Compound Name"], df["Plausibility Score"], df["MW"],
                                        df["cLogP"], df["TPSA"], df["Toxicity Flags"])
        ]
        fig.add_trace(go.Scatter(
            x=df["cLogP"], y=df["TPSA"], mode="markers", hovertext=hover, hoverinfo="text",
            marker=dict(
                size=df["MW"], sizemode="area", sizeref=sizeref, sizemin=6,
                color=df["Plausibility Score"], colorscale="Viridis", cmin=0, cmax=1,
                colorbar=dict(title="Plausibility", thickness=12),
                line=dict(width=[3 if s == "TOXIC ALERT" else 0.8 for s in df["Status"]],
                          color=["#C62828" if s == "TOXIC ALERT" else "#FFFFFF" for s in df["Status"]]),
                opacity=0.9,
            ),
        ))
    else:
        fig.add_annotation(text="No compounds at this threshold", showarrow=False, font=dict(size=14))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)), height=380, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="cLogP", range=ranges["x"], zeroline=False),
        yaxis=dict(title="TPSA (Å²)", range=ranges["y"], zeroline=False),
        template="plotly_white",
    )
    return fig


def badge_html(status: str) -> str:
    fg, bg = STATUS_COLORS[status]
    return (f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:999px;"
            f"font-size:0.78rem;font-weight:600;white-space:nowrap'>{status}</span>")


def style_status(val: str) -> str:
    fg, bg = STATUS_COLORS.get(val, ("#000", "#FFF"))
    return f"background-color:{bg};color:{fg};font-weight:600;text-align:center"


# ==============================================================================
# UI
# ==============================================================================

CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 6rem; max-width: 1400px;}
.dg-team {display:inline-block;background:#1B5E20;color:#fff;padding:4px 14px;border-radius:999px;
  font-size:0.85rem;font-weight:600;letter-spacing:.02em;margin-bottom:.4rem}
.dg-banner {background:#FFF4E5;border:1px solid #F5A623;border-left:6px solid #E65100;color:#5D3200;
  padding:12px 16px;border-radius:8px;font-size:0.95rem;line-height:1.4;margin:.6rem 0 1rem 0}
.dg-footer {position:fixed;left:0;right:0;bottom:0;z-index:999;background:#FFF4E5;color:#5D3200;
  border-top:2px solid #E65100;padding:8px 20px;font-size:0.8rem;text-align:center}
.dg-card {border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:12px 14px;height:100%}
.dg-quote {border-left:4px solid #2E7D32;padding:6px 12px;background:rgba(46,125,50,.07);font-style:italic}
h1 {font-size: clamp(1.4rem, 3.2vw, 2.1rem) !important; line-height:1.2 !important}
@media (max-width: 640px){ .dg-footer{font-size:0.7rem;padding:6px 10px} }
</style>
"""


def render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<span class='dg-team'>{TEAM_BADGE}</span>", unsafe_allow_html=True)
    st.title(TITLE)
    st.markdown(f"<div class='dg-banner' role='alert'><b>{DISCLAIMER}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='dg-footer'>{DISCLAIMER}</div>", unsafe_allow_html=True)


def sidebar() -> Dict[str, Any]:
    sb = st.sidebar
    sb.header("Controls")
    mode = sb.radio("Input mode", [MODE_PRESET, MODE_RAW])
    cfg: Dict[str, Any] = {"mode": mode}
    if mode == MODE_PRESET:
        label = sb.selectbox("Case study", list(CASE_STUDIES))
        cfg["case"] = CASE_STUDIES[label]
    sb.divider()
    cfg["threshold"] = sb.slider(
        "Solvent Plausibility Threshold", 0.0, 1.0, 0.40, 0.01,
        help="Compounds scoring below this extraction plausibility are treated as excluded (insoluble / lost).")
    cfg["show_excluded"] = sb.toggle("Show Excluded Insoluble Compounds", value=False)
    cfg["flag_toxic"] = sb.toggle("Flag Toxicity Alerts (SMARTS)", value=True,
                                  help="Flags pyrrolizidine alkaloid, aristolochic acid and illudane scaffolds.")
    sb.caption("Scores are first-order estimates from Wildman-Crippen cLogP, TPSA and thermal liability flags.")
    return cfg


def raw_text_inputs() -> Optional[Dict[str, Any]]:
    """Render the custom-text form and return the mined result stored in session state."""
    has_key = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    st.subheader("Raw Text Mining")
    if not has_key:
        st.info("No LLM API key found (`LLM_API_KEY` or `OPENAI_API_KEY`). Using the low-confidence offline keyword "
                "extractor, which only recognises plants in the curated LOTUS sample: "
                + ", ".join(known_taxa()) + ".")
    with st.form("raw_text_form"):
        text = st.text_area("Passage", height=170, placeholder="Paste a historical or ethnographic passage describing a preparation...")
        c1, c2 = st.columns(2)
        citation = c1.text_input("Publication reference", value="User-supplied text")
        region = c2.text_input("Geographic origin", value="Unspecified")
        submitted = st.form_submit_button("Mine text", type="primary")
    if submitted:
        if not text.strip():
            st.warning("Enter a passage to mine.")
        else:
            st.session_state["raw_result"] = {
                **mine_text(text, citation, region, has_key), "citation": citation, "text": text}
    return st.session_state.get("raw_result")


def render_provenance(analyses: List[Dict[str, Any]], citation: str) -> None:
    st.subheader("Extraction & Provenance")
    for i, a in enumerate(analyses):
        r, tx = a["record"], a["taxon"]
        with st.expander(f"{r['verbatim_plant_name']}: {prep_label(r)}", expanded=(i == 0)):
            st.markdown(f"<div class='dg-quote'>“{r['verbatim_quote']}”</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Publication reference**<br>{citation}", unsafe_allow_html=True)
            c2.markdown(f"**Geographic origin**<br>{r.get('cultural_or_geographic_origin') or 'Unspecified'}", unsafe_allow_html=True)
            c3.metric("Extraction confidence", f"{r['extraction_confidence']:.0%}")
            c4.markdown(
                f"**Accepted taxon**<br>{tx['accepted_scientific_name']}<br>"
                f"<small>{tx['taxonomic_status']}, match {tx['match_type']} ({tx['confidence']:.2f})</small>",
                unsafe_allow_html=True)
            st.caption(f"Part: {r['plant_part']} | Route: {r['administration_route']} | Indication: {r['verbatim_indication']}")
            if r.get("ambiguity_notes"):
                st.caption(f"Notes: {r['ambiguity_notes']}")


def render_comparison(analyses: List[Dict[str, Any]], frames: List[pd.DataFrame], cfg: Dict[str, Any]) -> None:
    st.subheader("Preparation Comparison" if len(analyses) > 1 else "Extracted Bioactives")
    all_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if all_df.empty:
        ranges = {"x": [-4, 8], "y": [0, 300], "mw_max": 500.0}
    else:
        ranges = {
            "x": [float(all_df["cLogP"].min()) - 1, float(all_df["cLogP"].max()) + 1],
            "y": [max(0.0, float(all_df["TPSA"].min()) - 20), float(all_df["TPSA"].max()) + 20],
            "mw_max": float(all_df["MW"].max()),
        }
    cols = st.columns(len(analyses))
    for col, a, df in zip(cols, analyses, frames):
        with col:
            st.markdown(f"##### {prep_label(a['record'])}")
            if df.empty:
                st.warning("No compounds available for this taxon in the curated LOTUS sample, or none pass the threshold.")
            else:
                items = "".join(
                    f"<div style='display:flex;justify-content:space-between;gap:8px;padding:4px 0;"
                    f"border-bottom:1px solid rgba(128,128,128,.2)'><span>{row['Compound Name']}"
                    f" <small style='opacity:.6'>{row['Plausibility Score']:.2f}</small></span>{badge_html(row['Status'])}</div>"
                    for _, row in df.sort_values("Plausibility Score", ascending=False).iterrows())
                st.markdown(f"<div class='dg-card'>{items}</div>", unsafe_allow_html=True)
            st.plotly_chart(scatter(df, "cLogP vs. TPSA (size = MW)", ranges), width="stretch",
                            key=f"scatter_{a['record']['preparation_type']}_{a['record']['solvent_class']}_{a['record']['thermal_regime']}")


def render_table(analyses: List[Dict[str, Any]], frames: List[pd.DataFrame]) -> None:
    st.subheader("Chemical Table")
    parts = []
    for a, df in zip(analyses, frames):
        if not df.empty:
            d = df.copy()
            d.insert(0, "Preparation", prep_label(a["record"]))
            parts.append(d)
    if not parts:
        st.info("No compounds to display.")
        return
    full = pd.concat(parts, ignore_index=True)
    options = sorted(full["Preparation"].unique())
    chosen = st.multiselect("Preparations", options, default=options)
    shown = full[full["Preparation"].isin(chosen)]
    cols = ["Preparation", "Compound Name", "SMILES", "MW", "cLogP", "TPSA", "Plausibility Score", "Status"]
    styled = shown[cols].style.map(style_status, subset=["Status"])
    st.dataframe(
        styled, width="stretch", hide_index=True,
        column_config={
            "Plausibility Score": st.column_config.ProgressColumn("Plausibility Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "MW": st.column_config.NumberColumn(format="%.1f"),
            "cLogP": st.column_config.NumberColumn(format="%.2f"),
            "TPSA": st.column_config.NumberColumn(format="%.1f"),
            "SMILES": st.column_config.TextColumn(width="medium"),
        })
    st.download_button("Download table (CSV)", shown[cols].to_csv(index=False), "decoctiongraph_compounds.csv", "text/csv")


def main() -> None:
    st.set_page_config(page_title="DecoctionGraph | Gentiana Labs", page_icon="🌿", layout="wide")
    render_header()
    cfg = sidebar()

    if cfg["mode"] == MODE_PRESET:
        case = load_case(cfg["case"])
        records, citation = case["records"], case["citation"]
        st.caption("Preloaded case studies use curated demo passages for illustration, not verbatim archival citations.")
    else:
        result = raw_text_inputs()
        if not result:
            st.stop()
        if result.get("error"):
            st.warning(result["error"])
        records, citation = result["records"], result["citation"]
        if not records:
            st.error("No preparation records could be extracted from this text.")
            st.stop()
        st.success(f"Extracted {len(records)} record(s) via {'LLM' if result['method'] == 'llm' else 'offline keyword'} mode.")

    records_json = json.dumps(records, sort_keys=True)
    analyses = analyze_records(records_json)
    frames = [build_frame(a, cfg["threshold"], cfg["flag_toxic"], cfg["show_excluded"]) for a in analyses]

    # Toxicity warnings
    if cfg["flag_toxic"]:
        toxic = {}
        for a in analyses:
            for c in a["compounds"]:
                if c["toxicity_alerts"]:
                    toxic.setdefault(c["compound_name"], set()).update(c["toxicity_alerts"])
        if toxic:
            names = "; ".join(f"{n} ({', '.join(sorted(v)).replace('_', ' ')})" for n, v in toxic.items())
            has_pa = any("pyrrolizidine_alkaloid" in v for v in toxic.values())
            pa_note = (" Pyrrolizidine alkaloids are associated with hepatotoxicity, and internal use of plants "
                       "containing them is discouraged by regulators.") if has_pa else ""
            st.error(f"TOXICITY ALERT: structural alerts detected for {names}.{pa_note} "
                     "Alerts are SMARTS screening flags, not toxicity determinations, and broad scaffold patterns "
                     "(e.g. polycyclic aromatics) can over-trigger.")
            hidden = [c["compound_name"] for a, f in zip(analyses, frames) for c in a["compounds"]
                      if c["toxicity_alerts"] and c["compound_name"] not in set(f["Compound Name"])]
            if hidden:
                st.warning("Toxic-alert compounds hidden by the current threshold: " + ", ".join(sorted(set(hidden)))
                           + ". Enable 'Show Excluded Insoluble Compounds' to see them.")

    render_provenance(analyses, citation)
    st.divider()
    render_comparison(analyses, frames, cfg)
    st.divider()
    render_table(analyses, frames)
    st.divider()
    st.subheader("Knowledge Graph")
    html, n_nodes, n_edges = build_graph_html(records_json, cfg["threshold"], cfg["show_excluded"])
    st.caption(f"{n_nodes} nodes, {n_edges} edges. Green = plant, blue = preparation, orange = indication, "
               "violet = plausible compound, grey = excluded, red = toxic alert.")
    components.html(html, height=700, scrolling=True)


if __name__ == "__main__":
    main()
