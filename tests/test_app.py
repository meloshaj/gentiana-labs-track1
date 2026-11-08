"""Unit tests for the DecoctionGraph Streamlit dashboard and helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.main import (
    CASE_STUDIES,
    _case_records,
    _heuristic_extract,
    _inline_local_assets,
    analyze_records,
    build_frame,
    build_graph_html,
    compute_optimization_grid,
    load_case,
    prep_label,
    rank_candidate_frame,
    scatter,
    selected_target_summary,
    status_for,
)
from src.schemas import EthnoPreparationRecord


def test_load_all_preloaded_cases():
    """Verify that all predefined case studies return valid EthnoPreparationRecords."""
    for label, case_key in CASE_STUDIES.items():
        res = load_case(case_key)
        assert "records" in res
        assert "citation" in res
        assert len(res["records"]) >= 1

        for r_dict in res["records"]:
            # Validate against Pydantic schema
            record = EthnoPreparationRecord(**r_dict)
            assert record.verbatim_plant_name
            assert record.solvent_class in [
                "water", "hydroalcoholic", "lipid_oil", "vinegar_acidic", "honey_syrup", "none", "unknown"
            ]
            assert 0.0 <= record.extraction_confidence <= 1.0


def test_status_for_and_prep_label():
    """Test status badge categorization and preparation label formatting."""
    rec = {
        "preparation_type": "oil_maceration",
        "solvent_class": "lipid_oil",
        "thermal_regime": "warm_steep",
    }
    label = prep_label(rec)
    assert label == "Oil Maceration (lipid oil, warm steep)"

    # Poultice formatting with default None solvent and Ambient thermal regime
    rec_poultice = {
        "preparation_type": "poultice",
        "solvent_class": "none",
        "thermal_regime": "none",
    }
    label_poultice = prep_label(rec_poultice)
    assert label_poultice == "Poultice (None (Direct/Topical), Ambient (25°C))"

    # Plausible high
    comp_high = {
        "compound_name": "Hyperforin",
        "extraction_plausibility_score": 0.90,
        "extraction_verdict": "high",
        "toxicity_alerts": [],
    }
    assert status_for(comp_high, threshold=0.40, flag_toxic=True) == "High"

    # Below threshold -> Excluded
    comp_low = {
        "compound_name": "Rutin",
        "extraction_plausibility_score": 0.10,
        "extraction_verdict": "excluded",
        "toxicity_alerts": [],
    }
    assert status_for(comp_low, threshold=0.40, flag_toxic=True) == "Excluded"

    # Toxic alert overrides score
    comp_toxic = {
        "compound_name": "Symphytine",
        "extraction_plausibility_score": 0.75,
        "extraction_verdict": "high",
        "toxicity_alerts": ["pyrrolizidine_alkaloid"],
    }
    assert status_for(comp_toxic, threshold=0.40, flag_toxic=True) == "TOXIC ALERT"
    # When toxic flagging is disabled, falls back to normal status
    assert status_for(comp_toxic, threshold=0.40, flag_toxic=False) == "High"


def test_offline_heuristic_extractor():
    """Verify keyword fallback extraction when LLM API keys are absent."""
    sample_text = "The fresh Gentiana lutea roots are boiled in spring water for two hours."
    records = _heuristic_extract(sample_text, citation="Test Ref", region="Alps")
    assert len(records) == 1
    rec = records[0]
    assert rec["verbatim_plant_name"] == "Gentiana lutea"
    assert rec["solvent_class"] == "water"
    assert rec["thermal_regime"] == "boiling_prolonged"
    assert rec["plant_part"] == "root"


def test_build_frame_and_scatter():
    """Verify DataFrame generation and Plotly scatter chart construction."""
    case = load_case("Hypericum perforatum")
    records_json = json.dumps(case["records"])
    analyses = analyze_records(records_json)
    assert len(analyses) == 2

    # Frame with show_excluded=True
    df_all = build_frame(analyses[0], threshold=0.40, flag_toxic=True, show_excluded=True)
    assert not df_all.empty
    assert "Compound Name" in df_all.columns
    assert "Plausibility Score" in df_all.columns

    # Frame with show_excluded=False
    df_filtered = build_frame(analyses[0], threshold=0.40, flag_toxic=True, show_excluded=False)
    assert len(df_filtered) <= len(df_all)

    # Plotly scatter figure
    ranges = {"x": [-4, 8], "y": [0, 300], "mw_max": 500.0}
    fig = scatter(df_filtered, "Test Scatter", ranges)
    assert fig is not None
    assert hasattr(fig, "data")


def test_dashboard_target_and_lead_ranking_follow_selected_indication():
    """Keep target cards, comparative leads, and exported rows semantically aligned."""
    case = load_case("Hypericum perforatum")
    analyses = analyze_records(json.dumps(case["records"]))

    oil_frame = build_frame(analyses[0], threshold=0.40, flag_toxic=False, show_excluded=True)
    oil_hyperforin = oil_frame[oil_frame["Compound Name"] == "Hyperforin"].iloc[0]
    assert "TRPC6" in oil_hyperforin["Target"]

    target_name, target_id, potency = selected_target_summary(analyses[0], threshold=0.40)
    assert target_name == "Transient receptor potential cation channel subfamily C member 6"
    assert target_id.startswith("CHEMBL4523378")
    assert potency != "Unavailable"

    ranked = rank_candidate_frame(oil_frame, threshold=0.40)
    assert ranked.iloc[0]["Compound Name"] == "Hyperforin"


def test_build_graph_html_and_inlining():
    """Verify PyVis HTML graph generation and asset inlining."""
    case = load_case("Artemisia annua")
    records_json = json.dumps(case["records"])
    html, n_nodes, n_edges = build_graph_html(records_json, threshold=0.40, show_excluded=False)

    assert n_nodes > 0
    assert n_edges > 0
    assert "<html" in html.lower()
    # Ensure inline local assets function runs without error
    inlined = _inline_local_assets('<script src="lib/bindings/utils.js"></script>')
    assert inlined is not None


def test_all_primary_streamlit_tabs_render_without_exceptions():
    """Exercise all primary tabs under Streamlit's test runtime."""
    app_file = Path(__file__).resolve().parents[1] / "app" / "main.py"
    app = AppTest.from_file(str(app_file))
    app.run(timeout=90)

    assert not app.exception
    labels = {tab.label for tab in app.tabs}
    assert {
        "🌿 Literature & Protocols",
        "🕸️ Knowledge Graph",
        "🧪 In Silico Formulator",
        "📄 Archival Corpus Miner",
    } <= labels


def test_compute_optimization_grid_yield_and_selectivity():
    """Verify in silico recipe grid computation, sweet spot finding, and selectivity windows."""
    from src.lotus_retriever import get_compounds_for_taxon

    # 1. Artemisia annua thermal sensitivity test
    art_comps = get_compounds_for_taxon("Artemisia annua")
    artemisinin = [c for c in art_comps if c["compound_name"] == "Artemisinin"][0]

    grid_art = compute_optimization_grid(
        target_smiles=artemisinin["smiles"],
        toxin_smiles=None,
        target_organs_tuple=tuple(artemisinin.get("documented_organs") or []),
        toxin_organs_tuple=(),
        prepared_part="aerial_parts",
        solvent_mode="hydroalcoholic",
        duration_min=15.0,
        y_axis_mode="Ethanol %",
        fixed_param=7.0,
    )

    assert "temps" in grid_art
    assert "y_vals" in grid_art
    assert len(grid_art["yield_matrix"]) == len(grid_art["y_vals"])
    assert len(grid_art["yield_matrix"][0]) == len(grid_art["temps"])
    # Thermal decay: cold (20°C) must have higher yield than boiling (100°C)
    idx_20 = grid_art["temps"].index(20)
    idx_100 = grid_art["temps"].index(100)
    high_eth_row = -1  # 95% ethanol
    assert grid_art["yield_matrix"][high_eth_row][idx_20] > grid_art["yield_matrix"][high_eth_row][idx_100]
    # Optimal temp should favor cold to moderate extraction due to endoperoxide thermolability
    assert grid_art["best_temp"] <= 40

    # 2. Symphytum officinale organ selectivity test
    sym_comps = get_compounds_for_taxon("Symphytum officinale")
    allantoin = [c for c in sym_comps if c["compound_name"] == "Allantoin"][0]
    symphytine = [c for c in sym_comps if c["compound_name"] == "Symphytine"][0]

    # Leaf extraction: Allantoin present in leaves, Symphytine restricted to root
    grid_leaf = compute_optimization_grid(
        target_smiles=allantoin["smiles"],
        toxin_smiles=symphytine["smiles"],
        target_organs_tuple=tuple(allantoin.get("documented_organs") or []),
        toxin_organs_tuple=tuple(symphytine.get("documented_organs") or []),
        prepared_part="leaf",
        solvent_mode="water",
        duration_min=30.0,
        y_axis_mode="pH",
        fixed_param=0.0,
    )
    # High selectivity window in leaves (Allantoin ~99% - Symphytine < 10%)
    assert grid_leaf["best_score"] >= 80.0
    assert grid_leaf["best_toxin_yield"] < 15.0

    # Root extraction: Symphytine co-extracts significantly
    grid_root = compute_optimization_grid(
        target_smiles=allantoin["smiles"],
        toxin_smiles=symphytine["smiles"],
        target_organs_tuple=tuple(allantoin.get("documented_organs") or []),
        toxin_organs_tuple=tuple(symphytine.get("documented_organs") or []),
        prepared_part="root",
        solvent_mode="water",
        duration_min=30.0,
        y_axis_mode="pH",
        fixed_param=0.0,
    )
    # Selectivity in root is substantially worse than in leaf due to PA co-extraction
    assert grid_root["best_score"] < grid_leaf["best_score"]


def test_batch_corpus_miner_tab_integration():
    """Verify batch corpus miner pipeline components integrated into the Streamlit app."""
    from src.pdf_ingestor import (
        build_atlas_dataset,
        export_atlas_csv,
        export_atlas_jsonld,
        extract_text_chunks_from_text,
        get_sample_corpus_text,
        process_document_chunks,
    )

    sample_text = get_sample_corpus_text()
    chunks = extract_text_chunks_from_text(sample_text)
    assert len(chunks) == 3

    batch = process_document_chunks(
        chunks=chunks,
        document_title="Alpine Herbal (1885)",
        region="Swiss Alps",
        force_offline=True,
    )
    assert len(batch.records) >= 6

    atlas = build_atlas_dataset(batch)
    df = atlas["atlas_df"]
    metrics = atlas["metrics"]

    assert not df.empty
    assert metrics["total_pages_scanned"] == 3
    assert metrics["unique_taxa_count"] >= 5

    csv_data = export_atlas_csv(df)
    assert len(csv_data) > 500

    jsonld_data = export_atlas_jsonld(batch, atlas)
    assert len(jsonld_data) > 1000
    parsed = json.loads(jsonld_data)
    assert "@context" in parsed
    assert "@graph" in parsed


