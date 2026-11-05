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
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
import streamlit as st
import streamlit.components.v1 as components

from src.cheminformatics import (
    calculate_descriptors,
    calculate_recipe_yield,
    compute_arrhenius_survival,
    compute_logd,
    detect_structural_alerts,
)
from src.extraction import extract_from_text, mock_extract_hypericum_sample
from src.graph_builder import DecoctionGraphBuilder
from src.lotus_retriever import enrich_taxon_compounds, get_compounds_for_taxon
from src.mechanism import get_targets_for_compound
from src.pdf_ingestor import (
    build_atlas_dataset,
    export_atlas_csv,
    export_atlas_jsonld,
    extract_text_chunks_from_pdf,
    extract_text_chunks_from_text,
    get_sample_corpus_text,
    process_document_chunks,
)
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

    if case == "Cinchona pubescens":
        ind = "malarial intermittent fever and ague"
        return [
            _rec(
                verbatim_plant_name="Cinchona pubescens", plant_part="bark", preparation_type="infusion",
                solvent_class="vinegar_acidic", thermal_regime="cold_ambient",
                estimated_ph=3.0, duration_minutes=1440.0, temperature_celsius=20.0,
                verbatim_indication=ind, administration_route="oral",
                verbatim_quote="The powdered Peruvian bark is steeped in sour wine for two days; the acid draws out the bitter salt to quell shaking agues.",
                cultural_or_geographic_origin="Colonial South American / European Jesuit", extraction_confidence=0.96,
                ambiguity_notes="Acidic wine maceration (pH 3.0) protonates quinine alkaloids into water-soluble salts.",
            ),
            _rec(
                verbatim_plant_name="Cinchona pubescens", plant_part="bark", preparation_type="decoction",
                solvent_class="water", thermal_regime="boiling_prolonged",
                estimated_ph=7.0, duration_minutes=45.0, temperature_celsius=100.0,
                verbatim_indication=ind, administration_route="oral",
                verbatim_quote="Boiled in plain well-water without acid, the bark releases meager virtue, its bitter principles remaining bound.",
                cultural_or_geographic_origin="Colonial South American / European Jesuit", extraction_confidence=0.90,
                ambiguity_notes="Neutral water fails to ionize the quinuclidine tertiary amine, restricting extraction.",
            ),
        ], DEMO_CITATION

    if case == "Papaver somniferum":
        return [
            _rec(
                verbatim_plant_name="Papaver somniferum", plant_part="fruit", preparation_type="infusion",
                solvent_class="water", thermal_regime="warm_steep",
                estimated_ph=7.0, duration_minutes=30.0, temperature_celsius=60.0,
                verbatim_indication="intractable pain and severe restlessness", administration_route="oral",
                verbatim_quote="The dried latex harvested from unripe green poppy heads is dissolved in warm water with syrup for agonizing pain and sleeplessness.",
                cultural_or_geographic_origin="Greco-Roman & Islamic Pharmacopeia", extraction_confidence=0.97,
                ambiguity_notes="Latex exudate is rich in morphinan alkaloids (morphine, codeine), yielding potent analgesia.",
            ),
            _rec(
                verbatim_plant_name="Papaver somniferum", plant_part="seed", preparation_type="decoction",
                solvent_class="water", thermal_regime="boiling_prolonged",
                estimated_ph=7.0, duration_minutes=30.0, temperature_celsius=100.0,
                verbatim_indication="mild demulcent nourishment", administration_route="oral",
                verbatim_quote="The ripe seeds, free of any milky juice, are boiled into a demulcent gruel for children, bearing no sleeping virtue.",
                cultural_or_geographic_origin="Greco-Roman & Islamic Pharmacopeia", extraction_confidence=0.93,
                ambiguity_notes="Organ localization: morphinan alkaloids are restricted to capsule laticifers and absent in mature seeds.",
            ),
        ], DEMO_CITATION
    raise KeyError(case)


CASE_STUDIES: Dict[str, str] = {
    "Hypericum perforatum (Oil Maceration vs. Water Tea)": "Hypericum perforatum",
    "Gentiana lutea (Boiling Root Decoction)": "Gentiana lutea",
    "Artemisia annua (Cold Wringing vs. Prolonged Boiling)": "Artemisia annua",
    "Symphytum officinale (Topical Poultice vs. Internal Tea & PA Toxicity Warning)": "Symphytum officinale",
    "Cinchona pubescens (Acidic Wine/Vinegar Maceration vs. Neutral Water)": "Cinchona pubescens",
    "Papaver somniferum (Opium Latex/Capsule vs. Ripe Seed Porridge)": "Papaver somniferum",
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
            taxon["accepted_scientific_name"],
            rec["solvent_class"],
            rec["thermal_regime"],
            gbif_usage_key=taxon.get("gbif_usage_key"),
            estimated_ph=rec.get("estimated_ph"),
            duration_minutes=rec.get("duration_minutes"),
            temperature_celsius=rec.get("temperature_celsius"),
            prepared_part=rec.get("plant_part", "unknown"),
        )
        if not compounds:  # taxon string may carry authorship the curated set lacks
            compounds = enrich_taxon_compounds(
                rec["verbatim_plant_name"],
                rec["solvent_class"],
                rec["thermal_regime"],
                estimated_ph=rec.get("estimated_ph"),
                duration_minutes=rec.get("duration_minutes"),
                temperature_celsius=rec.get("temperature_celsius"),
                prepared_part=rec.get("plant_part", "unknown"),
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
        primary_target = "None"
        if c.get("known_targets"):
            t = c["known_targets"][0]
            gene = t.get("gene_symbol", "")
            action = t.get("action_type", "")
            primary_target = f"{gene} ({action})" if gene else action

        rows.append({
            "Compound Name": c["compound_name"],
            "Class": c.get("acid_base_class", "neutral"),
            "logD (pH)": c.get("logd_at_recipe_ph", c["clogp"]),
            "MW": c["mw"],
            "cLogP": c["clogp"],
            "TPSA": c["tpsa"],
            "Organ Match": c.get("organ_match_factor", 1.0),
            "Thermal Surv": c.get("thermal_survival_fraction", 1.0),
            "Target": primary_target,
            "Plausibility Score": c["extraction_plausibility_score"],
            "Status": st_,
            "Toxicity Flags": ", ".join(c["toxicity_alerts"]) if flag_toxic else "",
            "Below Threshold": c["extraction_plausibility_score"] < threshold,
            "SMILES": c["smiles"],
        })
    df = pd.DataFrame(rows)
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
    st.subheader("Chemical & Pharmacological Table")
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
    cols = [
        "Preparation", "Compound Name", "Class", "logD (pH)", "Organ Match",
        "Thermal Surv", "Target", "Plausibility Score", "Status"
    ]
    styled = shown[cols].style.map(style_status, subset=["Status"])
    st.dataframe(
        styled, width="stretch", hide_index=True,
        column_config={
            "Plausibility Score": st.column_config.ProgressColumn("Plausibility Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Organ Match": st.column_config.ProgressColumn("Organ Match", min_value=0.0, max_value=1.0, format="%.2f"),
            "Thermal Surv": st.column_config.ProgressColumn("Thermal Surv", min_value=0.0, max_value=1.0, format="%.2f"),
            "logD (pH)": st.column_config.NumberColumn(format="%.2f"),
            "Class": st.column_config.TextColumn(width="small"),
            "Target": st.column_config.TextColumn(width="medium"),
        })
    st.download_button("Download table (CSV)", shown.to_csv(index=False), "decoctiongraph_compounds.csv", "text/csv")


@st.cache_data(show_spinner=False)
def compute_optimization_grid(
    target_smiles: str,
    toxin_smiles: Optional[str],
    target_organs_tuple: Tuple[str, ...],
    toxin_organs_tuple: Tuple[str, ...],
    prepared_part: str,
    solvent_mode: str,
    duration_min: float,
    y_axis_mode: str,
    fixed_param: float,
) -> Dict[str, Any]:
    temps = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    if y_axis_mode == "pH":
        y_vals = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
    else:  # Ethanol %
        y_vals = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0]

    yield_matrix: List[List[float]] = []
    selectivity_matrix: List[List[float]] = []

    best_score = -1.0
    best_temp = temps[0]
    best_y = y_vals[0]
    best_target_yield = 0.0
    best_toxin_yield = 0.0

    for y in y_vals:
        row_yield: List[float] = []
        row_sel: List[float] = []
        for t in temps:
            cur_ph = y if y_axis_mode == "pH" else fixed_param
            cur_eth = fixed_param if y_axis_mode == "pH" else y
            mode = "hydroalcoholic" if (y_axis_mode != "pH" or cur_eth > 0) else solvent_mode

            y_target = calculate_recipe_yield(
                target_smiles,
                solvent_mode=mode,
                temp_c=float(t),
                ph=float(cur_ph),
                duration_min=float(duration_min),
                ethanol_pct=float(cur_eth),
                prepared_part=prepared_part,
                documented_organs=list(target_organs_tuple),
            )
            y_tgt_pct = round(y_target * 100.0, 1)
            row_yield.append(y_tgt_pct)

            if toxin_smiles:
                y_toxin = calculate_recipe_yield(
                    toxin_smiles,
                    solvent_mode=mode,
                    temp_c=float(t),
                    ph=float(cur_ph),
                    duration_min=float(duration_min),
                    ethanol_pct=float(cur_eth),
                    prepared_part=prepared_part,
                    documented_organs=list(toxin_organs_tuple),
                )
                sel_pct = round(max(0.0, (y_target - y_toxin) * 100.0), 1)
                toxin_pct = round(y_toxin * 100.0, 1)
            else:
                toxin_pct = 0.0
                sel_pct = y_tgt_pct

            row_sel.append(sel_pct)

            if sel_pct > best_score:
                best_score = sel_pct
                best_temp = t
                best_y = y
                best_target_yield = y_tgt_pct
                best_toxin_yield = toxin_pct

        yield_matrix.append(row_yield)
        selectivity_matrix.append(row_sel)

    return {
        "temps": temps,
        "y_vals": y_vals,
        "yield_matrix": yield_matrix,
        "selectivity_matrix": selectivity_matrix,
        "best_temp": best_temp,
        "best_y": best_y,
        "best_score": best_score,
        "best_target_yield": best_target_yield,
        "best_toxin_yield": best_toxin_yield,
    }


def render_mining_view(cfg: Dict[str, Any]) -> None:
    if cfg["mode"] == MODE_PRESET:
        case = load_case(cfg["case"])
        records, citation = case["records"], case["citation"]
        st.caption("Preloaded case studies use curated demo passages for illustration, not verbatim archival citations.")
    else:
        result = raw_text_inputs()
        if not result:
            st.info("Paste or mine an ethnobotanical passage above to generate the knowledge graph.")
            return
        if result.get("error"):
            st.warning(result["error"])
        records, citation = result["records"], result["citation"]
        if not records:
            st.error("No preparation records could be extracted from this text.")
            return
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
               "violet = plausible compound, grey = excluded, red = toxic alert, gold = macromolecular target. "
               "⭐ Thick emerald edges = Validated Mechanistic Paths (S_ext ≥ 0.50, Alignment ≥ 0.60).")
    components.html(html, height=700, scrolling=True)


def render_recipe_optimizer() -> None:
    st.subheader("🧪 In Silico Recipe Optimizer & What-If Formulator")
    st.markdown(
        "Simulate and optimize continuous physicochemical extraction yields, thermodynamic selectivity "
        "windows, and Arrhenius thermal degradation kinetics across solvent vehicle, pH, temperature, and contact time."
    )

    # 1. Taxon & Compound Controls
    taxa_defaults = [
        "Artemisia annua",
        "Symphytum officinale",
        "Hypericum perforatum",
        "Gentiana lutea",
        "Cinchona pubescens",
        "Papaver somniferum",
    ]
    all_taxa = taxa_defaults + [t for t in known_taxa() if t not in taxa_defaults]

    c_tax, c_org, c_tgt, c_tox = st.columns([1.2, 0.9, 1.2, 1.2])

    with c_tax:
        chosen_taxon = st.selectbox("Plant Taxon", all_taxa, index=0, key="opt_taxon")

    # Smart organ default based on taxon
    organ_options = ["aerial_parts", "leaf", "flower", "root", "bark", "fruit", "seed", "rhizome", "stem", "whole_plant"]
    default_org_idx = 0
    tx_lower = chosen_taxon.lower()
    if "symphytum" in tx_lower:
        default_org_idx = organ_options.index("leaf")
    elif "gentiana" in tx_lower:
        default_org_idx = organ_options.index("root")
    elif "cinchona" in tx_lower:
        default_org_idx = organ_options.index("bark")
    elif "papaver" in tx_lower:
        default_org_idx = organ_options.index("fruit")
    elif "hypericum" in tx_lower:
        default_org_idx = organ_options.index("flower")
    elif "artemisia" in tx_lower:
        default_org_idx = organ_options.index("aerial_parts")

    with c_org:
        chosen_organ = st.selectbox("Plant Organ", organ_options, index=default_org_idx, key="opt_organ")

    # Load compounds for chosen taxon
    raw_compounds = get_compounds_for_taxon(chosen_taxon)
    if not raw_compounds:
        st.warning(f"No compound records found in LOTUS for {chosen_taxon}.")
        return

    comp_names = [c["compound_name"] for c in raw_compounds]
    comp_map = {c["compound_name"]: c for c in raw_compounds}

    # Smart target default
    default_tgt_idx = 0
    target_preferred = {
        "artemisia": "Artemisinin",
        "symphytum": "Allantoin",
        "hypericum": "Hyperforin",
        "gentiana": "Gentiopicroside",
        "cinchona": "Quinine",
        "papaver": "Morphine",
    }
    for k, v in target_preferred.items():
        if k in tx_lower and v in comp_names:
            default_tgt_idx = comp_names.index(v)
            break

    with c_tgt:
        chosen_target_name = st.selectbox("Target Bioactive Compound", comp_names, index=default_tgt_idx, key="opt_target")

    # Detect toxic alerts in compounds
    toxic_names = [c["compound_name"] for c in raw_compounds if detect_structural_alerts(c["smiles"])]
    toxin_choices = ["(None / Maximize Yield Only)"] + comp_names
    default_tox_idx = 0
    if toxic_names:
        default_tox_idx = toxin_choices.index(toxic_names[0])

    with c_tox:
        chosen_toxin_name = st.selectbox("Undesired Toxin to Penalize", toxin_choices, index=default_tox_idx, key="opt_toxin")

    target_comp = comp_map[chosen_target_name]
    target_smiles = target_comp["smiles"]
    target_organs = tuple(target_comp.get("documented_organs") or [])

    has_toxin = chosen_toxin_name != "(None / Maximize Yield Only)"
    if has_toxin:
        toxin_comp = comp_map[chosen_toxin_name]
        toxin_smiles = toxin_comp["smiles"]
        toxin_organs = tuple(toxin_comp.get("documented_organs") or [])
    else:
        toxin_comp = None
        toxin_smiles = None
        toxin_organs = ()

    st.markdown("---")

    # 2. Recipe Parameters Formulator Controls
    st.markdown("##### 🎛️ Extraction Vehicle & Physicochemical Conditions")
    col_veh, col_eth, col_ph, col_temp, col_time = st.columns(5)

    veh_options = [
        "Water (Aqueous)",
        "Hydroalcoholic (Ethanol / Water)",
        "Lipid Oil (Maceration)",
        "Vinegar / Acidic Wine",
    ]
    def_veh_idx = 0
    if "cinchona" in tx_lower:
        def_veh_idx = 3  # Vinegar
    elif "hypericum" in tx_lower:
        def_veh_idx = 2  # Lipid Oil

    with col_veh:
        chosen_veh = st.selectbox("Extraction Vehicle", veh_options, index=def_veh_idx, key="opt_veh")

    veh_mode_map = {
        "Water (Aqueous)": "water",
        "Hydroalcoholic (Ethanol / Water)": "hydroalcoholic",
        "Lipid Oil (Maceration)": "lipid_oil",
        "Vinegar / Acidic Wine": "vinegar_acidic",
    }
    solvent_mode = veh_mode_map[chosen_veh]

    with col_eth:
        if solvent_mode == "hydroalcoholic":
            ethanol_pct = st.slider("Ethanol % (v/v)", 0.0, 95.0, 50.0, 5.0, format="%.0f%%", key="opt_eth")
        else:
            st.slider("Ethanol % (v/v)", 0.0, 95.0, 0.0, disabled=True, format="%.0f%%", key="opt_eth_dis")
            ethanol_pct = 0.0

    with col_ph:
        default_ph = 3.0 if solvent_mode == "vinegar_acidic" else 7.0
        solv_ph = st.slider("Solvent pH", 2.5, 9.5, default_ph, 0.1, format="%.1f", key="opt_ph")

    with col_temp:
        default_t = 25.0 if "annua" in tx_lower else 80.0
        solv_temp = st.slider("Temperature (°C)", 20.0, 100.0, default_t, 5.0, format="%.0f°C", key="opt_temp")

    with col_time:
        solv_time = st.slider("Contact Time (min)", 5.0, 120.0, 30.0, 5.0, format="%.0f min", key="opt_time")

    # 3. Live Metrics Display
    cur_target_yield = calculate_recipe_yield(
        target_smiles,
        solvent_mode=solvent_mode,
        temp_c=solv_temp,
        ph=solv_ph,
        duration_min=solv_time,
        ethanol_pct=ethanol_pct,
        prepared_part=chosen_organ,
        documented_organs=list(target_organs),
    ) * 100.0

    if has_toxin and toxin_smiles:
        cur_toxin_yield = calculate_recipe_yield(
            toxin_smiles,
            solvent_mode=solvent_mode,
            temp_c=solv_temp,
            ph=solv_ph,
            duration_min=solv_time,
            ethanol_pct=ethanol_pct,
            prepared_part=chosen_organ,
            documented_organs=list(toxin_organs),
        ) * 100.0
        cur_selectivity = max(0.0, cur_target_yield - cur_toxin_yield)
    else:
        cur_toxin_yield = 0.0
        cur_selectivity = cur_target_yield

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"🎯 Target Yield: {chosen_target_name}",
        f"{cur_target_yield:.1f}%",
        help="Physical chemistry extraction yield based on logD(pH), Arrhenius kinetics, and organ localization.",
    )
    if has_toxin:
        m2.metric(
            f"⚠️ Toxin Co-Extraction: {chosen_toxin_name}",
            f"{cur_toxin_yield:.1f}%",
            delta=f"-{cur_toxin_yield:.1f}% risk",
            delta_color="inverse",
            help="Co-extracted fraction of undesirable or toxic compound.",
        )
    else:
        m2.metric("⚠️ Toxin Co-Extraction", "None Selected", help="No undesirable compound flagged or selected.")

    m3.metric(
        "✨ Selectivity Window",
        f"{cur_selectivity:.1f}%",
        help="Therapeutic window: Target Yield minus Toxin Co-Extraction.",
    )

    st.markdown("---")

    # 4. 2D Optimization Heatmaps
    st.markdown("##### 🗺️ 2D Extraction Landscape & Sweet Spot Discovery")
    y_axis_choice = st.radio(
        "Heatmap Y-Axis Parameter",
        ["Solvent pH (2.5 - 9.5)", "Ethanol % (0% - 95%)"],
        horizontal=True,
        key="opt_y_choice",
    )
    y_mode = "pH" if "pH" in y_axis_choice else "Ethanol %"
    fixed_p = ethanol_pct if y_mode == "pH" else solv_ph
    y_axis_title = "Solvent pH" if y_mode == "pH" else "Ethanol % (v/v)"

    grid = compute_optimization_grid(
        target_smiles=target_smiles,
        toxin_smiles=toxin_smiles if has_toxin else None,
        target_organs_tuple=target_organs,
        toxin_organs_tuple=toxin_organs,
        prepared_part=chosen_organ,
        solvent_mode=solvent_mode,
        duration_min=solv_time,
        y_axis_mode=y_mode,
        fixed_param=float(fixed_p),
    )

    hm1, hm2 = st.columns(2)

    with hm1:
        fig_yield = go.Figure(data=go.Heatmap(
            z=grid["yield_matrix"],
            x=grid["temps"],
            y=grid["y_vals"],
            colorscale="Viridis",
            colorbar=dict(title="Yield (%)", thickness=12),
            zmin=0, zmax=100,
            hovertemplate="Temp: %{x}°C<br>" + y_axis_title + ": %{y}<br>Yield: %{z:.1f}%<extra></extra>",
        ))
        fig_yield.update_layout(
            title=dict(text=f"Bioactive Yield: {chosen_target_name} (%)", font=dict(size=14)),
            xaxis=dict(title="Extraction Temperature (°C)", dtick=10),
            yaxis=dict(title=y_axis_title),
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
        )
        st.plotly_chart(fig_yield, width="stretch", key="fig_yield_hm")

    with hm2:
        fig_sel = go.Figure(data=go.Heatmap(
            z=grid["selectivity_matrix"],
            x=grid["temps"],
            y=grid["y_vals"],
            colorscale="Plasma",
            colorbar=dict(title="Selectivity (%)", thickness=12),
            zmin=0, zmax=100,
            hovertemplate="Temp: %{x}°C<br>" + y_axis_title + ": %{y}<br>Selectivity: %{z:.1f}%<extra></extra>",
        ))
        # Add Sweet Spot Star marker
        fig_sel.add_trace(go.Scatter(
            x=[grid["best_temp"]],
            y=[grid["best_y"]],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="#00FFCC", line=dict(color="#000000", width=2)),
            text=["★ Sweet Spot"],
            textposition="top center",
            textfont=dict(color="#111827", size=11, family="sans-serif"),
            name="Sweet Spot",
            hoverinfo="text",
            hovertext=(f"Optimal Coordinate: {grid['best_temp']}°C, {y_axis_title}={grid['best_y']}<br>"
                       f"Selectivity: {grid['best_score']:.1f}%"),
        ))
        fig_sel.update_layout(
            title=dict(
                text=f"Selectivity Window ({chosen_target_name}" + (f" vs {chosen_toxin_name}" if has_toxin else "") + ") (%)",
                font=dict(size=14),
            ),
            xaxis=dict(title="Extraction Temperature (°C)", dtick=10),
            yaxis=dict(title=y_axis_title),
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
            showlegend=False,
            template="plotly_white",
        )
        st.plotly_chart(fig_sel, width="stretch", key="fig_sel_hm")

    # 5. Dynamic Arrhenius Thermal Decay Curve
    st.markdown("##### ⏱️ Arrhenius Thermal Degradation Kinetics Profile")
    times = list(range(0, 95, 5))
    mol_target = Chem.MolFromSmiles(target_smiles)
    desc_target = calculate_descriptors(target_smiles, ph=solv_ph) or {"mw": 300.0, "tpsa": 60.0}

    surv_cur = [
        compute_arrhenius_survival(mol_target, temp_c=solv_temp, duration_min=t, mw=desc_target["mw"], tpsa=desc_target["tpsa"])[0] * 100.0
        for t in times
    ]
    surv_25 = [
        compute_arrhenius_survival(mol_target, temp_c=25.0, duration_min=t, mw=desc_target["mw"], tpsa=desc_target["tpsa"])[0] * 100.0
        for t in times
    ]
    surv_100 = [
        compute_arrhenius_survival(mol_target, temp_c=100.0, duration_min=t, mw=desc_target["mw"], tpsa=desc_target["tpsa"])[0] * 100.0
        for t in times
    ]

    fig_kinetics = go.Figure()
    fig_kinetics.add_trace(go.Scatter(
        x=times, y=surv_cur, mode="lines+markers",
        name=f"{chosen_target_name} @ {solv_temp:.0f}°C (Selected)",
        line=dict(color="#2563EB", width=3),
    ))
    fig_kinetics.add_trace(go.Scatter(
        x=times, y=surv_25, mode="lines",
        name=f"{chosen_target_name} @ 25°C (Ambient / Cold)",
        line=dict(color="#059669", width=2, dash="dash"),
    ))
    fig_kinetics.add_trace(go.Scatter(
        x=times, y=surv_100, mode="lines",
        name=f"{chosen_target_name} @ 100°C (Boiling Decoction)",
        line=dict(color="#DC2626", width=2, dash="dot"),
    ))

    if has_toxin and toxin_smiles:
        mol_toxin = Chem.MolFromSmiles(toxin_smiles)
        desc_toxin = calculate_descriptors(toxin_smiles, ph=solv_ph) or {"mw": 300.0, "tpsa": 60.0}
        surv_toxin = [
            compute_arrhenius_survival(mol_toxin, temp_c=solv_temp, duration_min=t, mw=desc_toxin["mw"], tpsa=desc_toxin["tpsa"])[0] * 100.0
            for t in times
        ]
        fig_kinetics.add_trace(go.Scatter(
            x=times, y=surv_toxin, mode="lines",
            name=f"Toxin: {chosen_toxin_name} @ {solv_temp:.0f}°C",
            line=dict(color="#9333EA", width=2, dash="dashdot"),
        ))

    fig_kinetics.update_layout(
        title=dict(text=f"Thermal Stability Profile: {chosen_target_name} Remaining (%) vs Contact Time", font=dict(size=14)),
        xaxis=dict(title="Extraction / Boiling Time (minutes)", dtick=10),
        yaxis=dict(title="Survival Fraction (%)", range=[-2, 105]),
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_kinetics, width="stretch", key="fig_kinetics_curve")

    # 6. Recommendation Callout Banner
    targets = get_targets_for_compound(compound_name=chosen_target_name, inchi_key=target_comp.get("inchikey"))
    primary_tgt = targets[0] if targets else None
    gene_symbol = primary_tgt.gene_symbol if primary_tgt else "Enzyme / Receptor"
    action = primary_tgt.action_type if primary_tgt else "modulator"
    pathway = primary_tgt.therapeutic_pathway if primary_tgt else "Cellular signaling and biological pathway modulation"

    if y_mode == "pH":
        opt_vehicle_desc = f"{chosen_veh} (pH {grid['best_y']:.1f})"
    else:
        opt_vehicle_desc = f"Hydroalcoholic ({grid['best_y']:.0f}% Ethanol)"

    toxin_clause = (
        f"while restricting <b>{chosen_toxin_name}</b> to <b>{grid['best_toxin_yield']:.1f}%</b>"
        if has_toxin else "with unconstrained selectivity"
    )

    rec_html = f"""
    <div style="background: linear-gradient(135deg, #064E3B 0%, #065F46 100%); color: white; padding: 20px 24px; border-radius: 12px; border: 1px solid #10B981; margin: 20px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <h4 style="color: #6EE7B7; margin: 0 0 6px 0; font-size: 1.2rem;">🏆 In Silico Formulation Recommendation</h4>
        <span style="background: #047857; color: #E6FFFA; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">Max Selectivity: {grid['best_score']:.1f}%</span>
      </div>
      <p style="font-size: 1.02rem; line-height: 1.5; margin: 8px 0 14px 0;">
        To maximize <b>{chosen_target_name}</b> (Target: <code>{gene_symbol}</code>, Action: <i>{action}</i>) with predicted yield of <b>{grid['best_target_yield']:.1f}%</b> {toxin_clause}:
      </p>
      <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <div style="background: rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 8px;">
          <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Plant Organ</div>
          <div style="font-size: 1.05rem; font-weight: 700;">{chosen_organ.replace('_', ' ').title()}</div>
        </div>
        <div style="background: rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 8px;">
          <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Optimal Vehicle</div>
          <div style="font-size: 1.05rem; font-weight: 700;">{opt_vehicle_desc}</div>
        </div>
        <div style="background: rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 8px;">
          <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Optimal Temp</div>
          <div style="font-size: 1.05rem; font-weight: 700;">{grid['best_temp']}°C</div>
        </div>
        <div style="background: rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 8px;">
          <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Contact Duration</div>
          <div style="font-size: 1.05rem; font-weight: 700;">{solv_time:.0f} min</div>
        </div>
        <div style="background: rgba(255,255,255,0.12); padding: 8px 14px; border-radius: 8px; max-width: 380px;">
          <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">Target Mechanism</div>
          <div style="font-size: 0.95rem; font-weight: 600; line-height: 1.3;">{pathway[:65]}...</div>
        </div>
      </div>
    </div>
    """
    st.markdown(rec_html, unsafe_allow_html=True)


def render_batch_corpus_miner() -> None:
    st.subheader("📄 Archival PDF & Batch Corpus Miner")
    st.markdown(
        "Ingest multi-page ethnobotanical survey papers, historical pharmacopeias, and archival field notes "
        "(`.pdf`, `.txt`). The engine parses pages, preserves strict page-number provenance (`Page X`), "
        "resolves botanical taxonomy via GBIF, maps chemical constituents via LOTUS, and validates macromolecular "
        "target pathways via ChEMBL to synthesize an interactive **Regional Ethnopharmacology Atlas**."
    )

    # Ingestion Controls
    col_up, col_sample = st.columns([1.6, 1.0])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Ethnobotanical Survey PDF or Herbal (.pdf, .txt)",
            type=["pdf", "txt"],
            key="corpus_file_uploader",
            help="Supports multi-page research surveys and raw archival herbal transcriptions.",
        )

    with col_sample:
        st.markdown("##### ⚡ Quick Demonstration")
        st.caption("No document handy? Test the multi-page chunked pipeline with a curated 3-page historical herbal survey:")
        if st.button("Load Archival Survey Herbal (3 Pages)", type="secondary", key="btn_load_demo_corpus"):
            st.session_state["corpus_file_name"] = "Alpine_and_Maritime_Herbal_Survey_1885.txt"
            st.session_state["corpus_file_bytes"] = get_sample_corpus_text().encode("utf-8")
            st.session_state["corpus_doc_title"] = "Alpine & Maritime Ethnobotanical Survey (1885)"
            st.session_state["corpus_region"] = "Swiss Alps & Mediterranean"
            st.success("Loaded 3-page historical herbal survey into session state!")

    # Check active document
    file_bytes: Optional[bytes] = None
    file_name = "sample_document.txt"

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        default_title = Path(file_name).stem.replace("_", " ").title()
    elif "corpus_file_bytes" in st.session_state:
        file_bytes = st.session_state["corpus_file_bytes"]
        file_name = st.session_state.get("corpus_file_name", "sample_document.txt")
        default_title = st.session_state.get("corpus_doc_title", "Alpine & Maritime Survey (1885)")
    else:
        default_title = "Alpine & Maritime Herbal Survey (1885)"

    st.markdown("---")
    st.markdown("##### 🏷️ Archival Provenance & Ingestion Parameters")
    c1, c2, c3 = st.columns([1.5, 1.2, 1.0])

    with c1:
        doc_title = st.text_input("Document Title / Archival Reference", value=default_title, key="corpus_title")
    with c2:
        doc_region = st.text_input(
            "Geographic or Cultural Region",
            value=st.session_state.get("corpus_region", "European Alps / Mediterranean"),
            key="corpus_region_input",
        )
    with c3:
        has_key = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
        offline_toggle = st.toggle(
            "Fast Offline Demo Mode",
            value=(not has_key),
            help="Enables rapid deterministic rule-based extraction for instant evaluation without LLM API quotas.",
        )

    btn_mine = st.button("🚀 Ingest Document & Synthesize Ethnopharmacology Atlas", type="primary", key="btn_run_mining")

    if btn_mine:
        if file_bytes is None:
            st.warning("Please upload a PDF / TXT document or click 'Load Archival Survey Herbal (3 Pages)' above.")
            return

        pbar = st.progress(0.0)
        status = st.empty()

        # Step 1: Chunk document
        status.info("Parsing document pages and generating chunks with provenance metadata...")
        pbar.progress(0.15)

        is_pdf = file_name.lower().endswith(".pdf")
        if is_pdf:
            chunks = extract_text_chunks_from_pdf(file_bytes)
        else:
            raw_text = file_bytes.decode("utf-8", errors="replace")
            chunks = extract_text_chunks_from_text(raw_text)

        if not chunks:
            st.error("Could not extract any readable text or pages from the uploaded file.")
            return

        pbar.progress(0.35)
        status.info(f"Generated {len(chunks)} text chunks across document pages. Mining ethnobotanical preparation records...")

        # Step 2: Mine chunks
        def progress_tracker(done: int, total: int, msg: str) -> None:
            pct = 0.35 + (float(done) / max(total, 1)) * 0.35
            pbar.progress(min(pct, 0.70))
            status.info(f"Mining chunks: {msg}")

        batch = process_document_chunks(
            chunks=chunks,
            document_title=doc_title,
            region=doc_region,
            progress_callback=progress_tracker,
            force_offline=offline_toggle,
        )

        pbar.progress(0.70)
        status.info(f"Mined {len(batch.records)} preparation records. Resolving taxonomy, LOTUS phytochemistry, and ChEMBL target validation...")

        # Step 3: Synthesize Atlas & Multi-document Knowledge Graph
        def atlas_tracker(pct: float, msg: str) -> None:
            pbar.progress(0.70 + pct * 0.28)
            status.info(f"Synthesizing Atlas: {msg}")

        atlas_result = build_atlas_dataset(batch, progress_callback=atlas_tracker)

        pbar.progress(1.0)
        status.success(f"Atlas synthesized successfully! Mined {len(batch.records)} preparations across {atlas_result['metrics']['total_pages_scanned']} page(s).")

        st.session_state["corpus_atlas_result"] = {
            "atlas": atlas_result,
            "batch": batch,
            "doc_title": doc_title,
        }

    # Render results if available
    result_data = st.session_state.get("corpus_atlas_result")
    if result_data:
        atlas = result_data["atlas"]
        batch = result_data["batch"]
        df: pd.DataFrame = atlas["atlas_df"]
        metrics = atlas["metrics"]

        st.markdown("---")
        st.markdown(f"### 🗺️ Regional Ethnopharmacology Atlas: *{result_data['doc_title']}*")

        # Top Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📄 Pages Scanned", f"{metrics['total_pages_scanned']}")
        m2.metric("🌿 Taxa Resolved", f"{metrics['unique_taxa_count']}")
        m3.metric("🥣 Preparations", f"{metrics['distinct_preparations_count']}")
        m4.metric("⭐ Validated Links", f"{metrics['validated_mechanistic_links_count']}", delta="+mechanistic")
        m5.metric(
            "⚠️ Toxicity Alerts",
            f"{metrics['toxicity_alerts_count']}",
            delta=f"{metrics['toxicity_alerts_count']} flagged" if metrics['toxicity_alerts_count'] > 0 else "None",
            delta_color="inverse" if metrics['toxicity_alerts_count'] > 0 else "off",
        )

        # Export row
        col_dl1, col_dl2 = st.columns(2)
        csv_data = export_atlas_csv(df)
        jsonld_data = export_atlas_jsonld(batch, atlas)

        col_dl1.download_button(
            "📥 Download Atlas Table (CSV)",
            data=csv_data,
            file_name=f"{re.sub(r'[^a-zA-Z0-9]', '_', result_data['doc_title'])}_atlas.csv",
            mime="text/csv",
            key="dl_atlas_csv",
            use_container_width=True,
        )
        col_dl2.download_button(
            "🕸️ Download Knowledge Graph (JSON-LD)",
            data=jsonld_data,
            file_name=f"{re.sub(r'[^a-zA-Z0-9]', '_', result_data['doc_title'])}_kg.jsonld",
            mime="application/ld+json",
            key="dl_atlas_jsonld",
            use_container_width=True,
        )

        # Display views: Table and PyVis Graph
        tab_tbl, tab_kg = st.tabs(["📋 Enriched Ethnopharmacology Atlas Table", "🌐 Multi-Document Knowledge Graph"])

        with tab_tbl:
            # Filter controls
            c_f1, c_f2 = st.columns(2)
            pages = ["All Pages"] + sorted(list(df["Source Page"].unique()))
            chosen_page = c_f1.selectbox("Filter by Source Page", pages, index=0, key="filter_page")

            taxa = ["All Taxa"] + sorted(list(df["Accepted Taxon"].unique()))
            chosen_taxon = c_f2.selectbox("Filter by Plant Taxon", taxa, index=0, key="filter_taxon")

            filtered_df = df.copy()
            if chosen_page != "All Pages":
                filtered_df = filtered_df[filtered_df["Source Page"] == chosen_page]
            if chosen_taxon != "All Taxa":
                filtered_df = filtered_df[filtered_df["Accepted Taxon"] == chosen_taxon]

            show_cols = [
                "Source Page", "Plant Name", "Accepted Taxon", "Plant Part",
                "Preparation", "Solvent", "Thermal Regime", "Historical Indication",
                "Primary Bioactive", "Top Target (Gene)", "Extraction Plausibility",
                "Mechanistic Status", "Toxicity Alerts"
            ]

            def highlight_validated(val: str) -> str:
                if "Validated" in str(val):
                    return "background-color: #D1FAE5; color: #065F46; font-weight: 700; text-align: center;"
                return "text-align: center;"

            def highlight_tox(val: str) -> str:
                if val and val != "None":
                    return "background-color: #FEE2E2; color: #991B1B; font-weight: 600;"
                return ""

            styled_df = (
                filtered_df[show_cols]
                .style
                .map(highlight_validated, subset=["Mechanistic Status"])
                .map(highlight_tox, subset=["Toxicity Alerts"])
            )

            st.dataframe(
                styled_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Extraction Plausibility": st.column_config.ProgressColumn(
                        "Extraction Plausibility", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    "Mechanistic Status": st.column_config.TextColumn(width="medium"),
                    "Toxicity Alerts": st.column_config.TextColumn(width="medium"),
                },
            )

        with tab_kg:
            st.caption(
                "Multi-document knowledge graph integrating all mined plants, preparations, LOTUS phytochemicals, "
                "and ChEMBL macromolecular targets across document pages. "
                "⭐ Thick emerald edges = Validated Mechanistic Paths ($S_{\\text{ext}} \\ge 0.50$, Alignment $\\ge 0.60$)."
            )
            builder: DecoctionGraphBuilder = atlas["graph_builder"]
            with tempfile.TemporaryDirectory() as tmp_dir:
                cwd = os.getcwd()
                os.chdir(tmp_dir)
                try:
                    pyvis_path = builder.to_pyvis(str(Path(tmp_dir) / "corpus_graph.html"), min_plausibility=0.35)
                    kg_html = Path(pyvis_path).read_text(encoding="utf-8")
                finally:
                    os.chdir(cwd)

            inlined_kg_html = _inline_local_assets(kg_html)
            components.html(inlined_kg_html, height=720, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="DecoctionGraph | Gentiana Labs", page_icon="🌿", layout="wide")
    render_header()
    cfg = sidebar()

    tab_mining, tab_optimizer, tab_corpus = st.tabs([
        "🌿 Ethnobotanical Literature Mining & Knowledge Graph",
        "🧪 In Silico Recipe Optimizer (What-If Formulator)",
        "📄 Archival PDF & Batch Corpus Miner",
    ])

    with tab_mining:
        render_mining_view(cfg)

    with tab_optimizer:
        render_recipe_optimizer()

    with tab_corpus:
        render_batch_corpus_miner()


if __name__ == "__main__":
    main()
