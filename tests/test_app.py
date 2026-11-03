"""Unit tests for the DecoctionGraph Streamlit dashboard and helper functions."""

from __future__ import annotations

import json
import pandas as pd
import pytest

from app.main import (
    CASE_STUDIES,
    _case_records,
    _heuristic_extract,
    _inline_local_assets,
    analyze_records,
    build_frame,
    build_graph_html,
    load_case,
    prep_label,
    scatter,
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
