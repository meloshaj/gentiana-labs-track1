"""Unit tests for DecoctionGraph LOTUS knowledge retriever."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.schemas import PhytochemicalRecord
from src.lotus_retriever import (
    normalize_botanical_name,
    get_compounds_for_taxon,
    load_lotus_subset,
    enrich_taxon_compounds,
)


def test_normalize_botanical_name():
    """Verify botanical name normalization strips authorities and infraspecific markers."""
    assert normalize_botanical_name("Gentiana lutea L.") == "Gentiana lutea"
    assert normalize_botanical_name("Hypericum perforatum L.") == "Hypericum perforatum"
    assert normalize_botanical_name("Artemisia annua Mill.") == "Artemisia annua"
    assert normalize_botanical_name("Symphytum officinale L.") == "Symphytum officinale"
    assert normalize_botanical_name("gentiana lutea") == "Gentiana lutea"
    assert normalize_botanical_name("Gentiana lutea subsp. lutea") == "Gentiana lutea"


def test_offline_benchmark_species_retrieval():
    """Verify retrieval of secondary metabolites for the 4 core benchmark taxa."""
    # 1. Gentiana lutea
    gentiana_comps = get_compounds_for_taxon("Gentiana lutea L.")
    gentiana_names = {c["compound_name"] for c in gentiana_comps}
    assert len(gentiana_comps) == 5
    assert {"Gentiopicroside", "Amarogentin", "Gentisin", "Swertiamarin", "Loganic acid"}.issubset(gentiana_names)

    # 2. Hypericum perforatum
    hypericum_comps = get_compounds_for_taxon("Hypericum perforatum L.")
    hypericum_names = {c["compound_name"] for c in hypericum_comps}
    assert len(hypericum_comps) == 7
    assert {"Hypericin", "Hyperforin", "Rutin", "Quercetin", "Chlorogenic acid", "Pseudohypericin", "Biapigenin"}.issubset(hypericum_names)

    # 3. Artemisia annua
    artemisia_comps = get_compounds_for_taxon("Artemisia annua L.")
    artemisia_names = {c["compound_name"] for c in artemisia_comps}
    assert len(artemisia_comps) == 4
    assert {"Artemisinin", "Arteannuin B", "Scopoletin", "Chrysosplenol D"}.issubset(artemisia_names)

    # 4. Symphytum officinale
    symphytum_comps = get_compounds_for_taxon("Symphytum officinale L.")
    symphytum_names = {c["compound_name"] for c in symphytum_comps}
    assert len(symphytum_comps) == 4
    assert {"Allantoin", "Rosmarinic acid", "Symphytine", "Echimidine"}.issubset(symphytum_names)


def test_retrieval_by_gbif_key():
    """Verify retrieval using GBIF taxon usage key."""
    # GBIF key for Gentiana lutea
    comps = get_compounds_for_taxon("Some plant", gbif_usage_key=3170040)
    assert len(comps) == 5
    assert any(c["compound_name"] == "Gentiopicroside" for c in comps)

    # GBIF key for Hypericum perforatum
    comps_hyp = get_compounds_for_taxon("Unresolved name", gbif_usage_key=3189486)
    assert len(comps_hyp) == 7
    assert any(c["compound_name"] == "Hyperforin" for c in comps_hyp)


def test_compound_record_structure():
    """Verify presence and validity of required chemical dictionary keys."""
    comps = get_compounds_for_taxon("Gentiana lutea")
    for c in comps:
        assert "compound_name" in c and len(c["compound_name"]) > 0
        assert "smiles" in c and len(c["smiles"]) > 0
        assert "inchi_key" in c and len(c["inchi_key"]) == 27
        assert "doi" in c


def test_parquet_and_csv_loading(tmp_path: Path):
    """Verify loading custom LOTUS subsets from Parquet and CSV files."""
    sample_df = pd.DataFrame([
        {
            "organism_name": "Salix alba L.",
            "organism_gbif_id": 5372420,
            "compound_name": "Salicin",
            "smiles": "OCC1=CC=CC=C1OC2OC(CO)C(O)C(O)C2O",
            "inchi_key": "NGKDIPCHOVECDX-UHFFFAOYSA-N",
            "doi": "10.1016/j.phytochem.2019.01.001",
        },
        {
            "organism_name": "Salix alba L.",
            "organism_gbif_id": 5372420,
            "compound_name": "Saligenin",
            "smiles": "OCC1=CC=CC=C1O",
            "inchi_key": "OZZNZZVHDFHNMS-UHFFFAOYSA-N",
            "doi": "10.1016/j.phytochem.2019.01.002",
        }
    ])

    # 1. Test CSV loading
    csv_file = tmp_path / "lotus_sample.csv"
    sample_df.to_csv(csv_file, index=False)
    load_lotus_subset(csv_file)

    salix_comps = get_compounds_for_taxon("Salix alba L.")
    assert len(salix_comps) == 2
    assert {c["compound_name"] for c in salix_comps} == {"Salicin", "Saligenin"}

    # 2. Test Parquet loading
    pq_file = tmp_path / "lotus_sample.parquet"
    sample_df.to_parquet(pq_file, index=False)
    load_lotus_subset(pq_file)

    salix_by_gbif = get_compounds_for_taxon("Unknown", gbif_usage_key=5372420)
    assert len(salix_by_gbif) == 2


def test_enrich_taxon_compounds_symphytum_pyrrolizidine():
    """Verify chemical enrichment and toxicity alert detection on Symphytum officinale."""
    records = enrich_taxon_compounds(
        accepted_name="Symphytum officinale L.",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        gbif_usage_key=7558421,
    )
    assert len(records) == 4
    for r in records:
        assert isinstance(r, PhytochemicalRecord)
        if r.compound_name in ("Symphytine", "Echimidine"):
            assert "pyrrolizidine_alkaloid" in r.toxicity_alerts


def test_enrich_taxon_compounds_plausibility_filtering():
    """Verify physicochemical plausibility differentiation in Gentiana and Hypericum."""
    # Gentiana lutea in water: glycosides are high plausibility
    gentian_recs = enrich_taxon_compounds(
        accepted_name="Gentiana lutea",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
    )
    gentio_rec = next(r for r in gentian_recs if r.compound_name == "Gentiopicroside")
    assert gentio_rec.extraction_verdict == "high"
    assert gentio_rec.extraction_plausibility_score >= 0.75

    # Hypericum perforatum in lipid_oil: Hyperforin is high, Rutin is excluded
    hyp_recs = enrich_taxon_compounds(
        accepted_name="Hypericum perforatum",
        solvent_class="lipid_oil",
        thermal_regime="cold_ambient",
    )
    hyperforin_rec = next(r for r in hyp_recs if r.compound_name == "Hyperforin")
    rutin_rec = next(r for r in hyp_recs if r.compound_name == "Rutin")

    assert hyperforin_rec.extraction_verdict == "high"
    assert rutin_rec.extraction_verdict == "excluded"

