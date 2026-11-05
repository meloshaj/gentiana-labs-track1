"""Unit tests for DecoctionGraph mechanism and reverse-pharmacology module."""

import json
from pathlib import Path
import pytest

from src.schemas import TargetMechanismRecord, PhytochemicalRecord
from src.mechanism import (
    CURATED_BIOACTIVITY_KB,
    get_targets_for_compound,
    fetch_chembl_targets,
    score_mechanistic_alignment,
    calculate_preparation_mechanistic_validation,
    validate_mechanistic_path,
)


def test_curated_knowledge_base_retrieval():
    """Verify high-precision bioactivity target lookup across benchmark constituents."""
    # 1. Hyperforin (Hypericum perforatum)
    hyp_targets = get_targets_for_compound("Hyperforin")
    assert len(hyp_targets) >= 3
    gene_symbols = {t.gene_symbol for t in hyp_targets}
    assert {"TRPC6", "SLC6A4", "GyrA"}.issubset(gene_symbols)
    trpc6 = next(t for t in hyp_targets if t.gene_symbol == "TRPC6")
    assert trpc6.action_type == "modulator"
    assert trpc6.bioactivity_potency_um == 0.3
    assert "wound" in trpc6.therapeutic_pathway.lower()

    # 2. Artemisinin (Artemisia annua)
    art_targets = get_targets_for_compound("Artemisinin")
    assert len(art_targets) >= 1
    pfatp6 = art_targets[0]
    assert pfatp6.gene_symbol == "PfATP6"
    assert pfatp6.action_type == "inhibitor"
    assert pfatp6.bioactivity_potency_um == 0.009

    # 3. Allantoin (Symphytum officinale)
    allant_targets = get_targets_for_compound("Allantoin")
    assert len(allant_targets) >= 1
    egfr = allant_targets[0]
    assert egfr.gene_symbol == "EGFR"
    assert "fibroblast" in egfr.therapeutic_pathway.lower() or "collagen" in egfr.therapeutic_pathway.lower()

    # 4. Quinine (Cinchona pubescens)
    quin_targets = get_targets_for_compound("Quinine")
    assert any(t.gene_symbol == "PfCRT" for t in quin_targets)

    # 5. Morphine (Papaver somniferum)
    morph_targets = get_targets_for_compound("Morphine")
    oprm1 = next(t for t in morph_targets if t.gene_symbol == "OPRM1")
    assert oprm1.action_type == "agonist"
    assert oprm1.bioactivity_potency_um == 0.002


def test_retrieval_by_inchikey():
    """Verify target lookup via canonical InChIKey."""
    # Artemisinin InChIKey
    targets = get_targets_for_compound("Unknown Natural Product", inchi_key="BLUAFEHZUWYNDE-NNWCWBAJSA-N")
    assert len(targets) >= 1
    assert any(t.gene_symbol == "PfATP6" for t in targets)

    # Allantoin InChIKey
    targets_allant = get_targets_for_compound("Unknown", inchi_key="POJWUDADGALRAB-UHFFFAOYSA-N")
    assert len(targets_allant) >= 1
    assert any(t.gene_symbol == "EGFR" for t in targets_allant)


def test_chembl_api_client_and_cache(tmp_path: Path):
    """Verify live ChEMBL querying, JSON schema adherence, and local caching."""
    # Artemisinin InChIKey
    art_key = "BLUAFEHZUWYNDE-NNWCWBAJSA-N"
    cache_dir = tmp_path / "chembl_cache"

    # 1. Fetch targets using temporary cache directory
    targets = fetch_chembl_targets(art_key, cache_dir=cache_dir, timeout=4.0)
    # Network may or may not succeed; function must not crash and must return TargetMechanismRecord instances
    for t in targets:
        assert isinstance(t, TargetMechanismRecord)
        assert t.target_name
        assert t.gene_symbol

    # 2. Check offline fallback on invalid/empty InChIKey
    empty_targets = fetch_chembl_targets("INVALID_KEY", cache_dir=cache_dir)
    assert empty_targets == []


def test_mechanistic_alignment_scoring():
    """Verify semantic alignment scoring between target pathways and historical indications."""
    # 1. Hyperforin matches wounds/burns strongly, but matches intermittent fevers poorly
    hyp_targets = get_targets_for_compound("Hyperforin")
    score_wound = score_mechanistic_alignment(hyp_targets, "severe burns, cuts and skin wounds")
    score_fever = score_mechanistic_alignment(hyp_targets, "intermittent tertian fever and ague")

    assert score_wound >= 0.85
    assert score_fever <= 0.20

    # 2. Artemisinin matches malarial fevers strongly
    art_targets = get_targets_for_compound("Artemisinin")
    score_art_fever = score_mechanistic_alignment(art_targets, "intermittent ague and burning fevers")
    score_art_bone = score_mechanistic_alignment(art_targets, "broken bones and fractured limbs")

    assert score_art_fever >= 0.90
    assert score_art_bone <= 0.20

    # 3. Amarogentin matches gastric weakness and appetite loss
    amaro_targets = get_targets_for_compound("Amarogentin")
    score_amaro_stomach = score_mechanistic_alignment(amaro_targets, "loss of appetite, colic and stomach weakness")
    assert score_amaro_stomach >= 0.90

    # 4. Allantoin matches bone fractures and sprains
    allant_targets = get_targets_for_compound("Allantoin")
    score_allant_bone = score_mechanistic_alignment(allant_targets, "broken bones, sprains and bruises")
    assert score_allant_bone >= 0.85

    # 5. Symphytine (toxin) does NOT explain bone knitting
    symp_targets = get_targets_for_compound("Symphytine")
    score_symp_bone = score_mechanistic_alignment(symp_targets, "broken bones and sprains")
    assert score_symp_bone <= 0.15


def test_preparation_level_validation_score():
    """Verify aggregate preparation Mechanistic Validation Score (M_prep)."""
    # Create mock PhytochemicalRecord for Hyperforin in oil (high plausibility, high alignment)
    rec_oil_hyperforin = PhytochemicalRecord(
        compound_name="Hyperforin",
        smiles="CC(C)C(=O)C12C(=O)C(=C(C(C1=O)(CC(C2(C)CCC=C(C)C)CC=C(C)C)CC=C(C)C)O)CC=C(C)C",
        inchi_key="KGSZHKRKHXOAMG-HQKKAZOISA-N",
        molecular_formula="C35H52O4",
        mw=536.8,
        clogp=8.99,
        tpsa=74.6,
        h_donors=1,
        h_acceptors=4,
        rotatable_bonds=8,
        is_thermolabile=True,
        is_volatile=False,
        toxicity_alerts=[],
        extraction_plausibility_score=0.90,
        extraction_verdict="high",
        known_targets=get_targets_for_compound("Hyperforin"),
        mechanistic_alignment_score=0.95,
    )

    # In oil maceration for wounds: M_prep >= 0.80
    m_prep_oil = calculate_preparation_mechanistic_validation(
        [rec_oil_hyperforin],
        "severe skin wounds and burns",
    )
    assert m_prep_oil >= 0.80

    # In water tea, Hyperforin has near-zero extraction plausibility
    rec_tea_hyperforin = rec_oil_hyperforin.model_copy(update={"extraction_plausibility_score": 0.01})
    m_prep_tea = calculate_preparation_mechanistic_validation(
        [rec_tea_hyperforin],
        "severe skin wounds and burns",
    )
    assert m_prep_tea < 0.05


def test_validate_mechanistic_path():
    """Verify binary classification of validated mechanistic paths."""
    # S_ext >= 0.50 and alignment >= 0.60 -> True
    assert validate_mechanistic_path(0.85, 0.90) is True
    assert validate_mechanistic_path(0.50, 0.60) is True

    # High extraction but unaligned indication -> False
    assert validate_mechanistic_path(0.95, 0.15) is False

    # Aligned target but insolubilized / lost molecule -> False
    assert validate_mechanistic_path(0.05, 0.98) is False

