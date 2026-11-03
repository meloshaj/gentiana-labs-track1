"""Unit tests for DecoctionGraph cheminformatics scoring engine."""

import pytest
import numpy as np

from src.schemas import PhytochemicalRecord
from src.cheminformatics import (
    calculate_descriptors,
    detect_structural_alerts,
    calculate_extraction_plausibility,
    map_plausibility_to_verdict,
    create_phytochemical_record,
)

RUTIN_SMILES = "CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C=C5)O)O)O)O)O)O)O)O"
HYPERFORIN_SMILES = "CC(C)C(=O)C12C(=O)C(=C(C(C1=O)(CC(C2(C)CCC=C(C)C)CC=C(C)C)CC=C(C)C)O)CC=C(C)C"
ARTEMISININ_SMILES = "CC1CCC2C(C(=O)OC3C24C1CCC(O3)(OO4)C)C"
MENTHOL_SMILES = "CC1CCC(C(C1)O)C(C)C"  # Volatile monoterpenoid
PTAQUILOSIDE_SMILES = "CC1CC2(C=C(C3(CC3)C(C2C1=O)(C)O)C)OC4C(C(C(C(O4)CO)O)O)O"
SENECIONINE_SMILES = "CC=C1CC(C)C(O)(C(=O)OCC2=CCN3CCC(C23)OC1=O)C"


# ==============================================================================
# Requirement 2: Structural Alert Detection Tests
# ==============================================================================

def test_structural_alerts():
    """Verify detection of toxic structural scaffolds."""
    desc_pta = calculate_descriptors(PTAQUILOSIDE_SMILES)
    assert desc_pta is not None
    assert "ptaquiloside_illudane_skeleton" in desc_pta["toxicity_alerts"]

    desc_sen = calculate_descriptors(SENECIONINE_SMILES)
    assert desc_sen is not None
    assert "pyrrolizidine_alkaloid" in desc_sen["toxicity_alerts"]

    # Benign molecule has empty alerts
    desc_rutin = calculate_descriptors(RUTIN_SMILES)
    assert desc_rutin is not None
    assert desc_rutin["toxicity_alerts"] == []


# ==============================================================================
# Requirement 3: Molecular Property Calculation Tests
# ==============================================================================

def test_calculate_descriptors_valid():
    """Verify accurate calculation of 2D descriptors on Rutin."""
    desc = calculate_descriptors(RUTIN_SMILES)
    assert desc is not None
    assert desc["mw"] == pytest.approx(610.52, abs=0.5)
    assert desc["clogp"] == pytest.approx(-1.69, abs=0.5)
    assert desc["tpsa"] == pytest.approx(269.43, abs=1.0)
    assert desc["h_donors"] == 10
    assert desc["h_acceptors"] == 16
    assert desc["rotatable_bonds"] == 6
    assert desc["is_volatile"] is False
    assert desc["is_thermolabile"] is False
    assert "C27H30O16" in desc["molecular_formula"]
    assert len(desc["inchi_key"]) == 27


def test_calculate_descriptors_invalid_smiles():
    """Verify invalid SMILES returns None."""
    assert calculate_descriptors("INVALID_SMILES_STRING") is None
    assert calculate_descriptors("") is None
    assert calculate_descriptors(None) is None


def test_volatility_and_thermolability_flags():
    """Verify volatility and thermolability detection."""
    # Menthol: volatile monoterpene (MW < 220, cLogP < 3.5, TPSA < 40)
    menthol_desc = calculate_descriptors(MENTHOL_SMILES)
    assert menthol_desc is not None
    assert menthol_desc["is_volatile"] is True
    assert menthol_desc["is_thermolabile"] is False

    # Artemisinin: thermolabile endoperoxide
    art_desc = calculate_descriptors(ARTEMISININ_SMILES)
    assert art_desc is not None
    assert art_desc["is_thermolabile"] is True
    assert art_desc["is_volatile"] is False


# ==============================================================================
# Requirement 4 & 6: Continuous Plausibility Scoring Tests
# ==============================================================================

def test_water_vs_lipid_scoring():
    """Verify polarity contrast between water and lipid_oil for Rutin and Hyperforin."""
    rutin_desc = calculate_descriptors(RUTIN_SMILES)
    hyp_desc = calculate_descriptors(HYPERFORIN_SMILES)

    # Rutin: polar -> favors water, excluded from oil
    r_water = calculate_extraction_plausibility(rutin_desc, "water", "cold_ambient")
    r_oil = calculate_extraction_plausibility(rutin_desc, "lipid_oil", "cold_ambient")
    assert r_water > r_oil
    assert r_water >= 0.75
    assert r_oil < 0.15

    # Hyperforin: lipophilic -> favors oil, excluded from water
    h_water = calculate_extraction_plausibility(hyp_desc, "water", "cold_ambient")
    h_oil = calculate_extraction_plausibility(hyp_desc, "lipid_oil", "cold_ambient")
    assert h_oil > h_water
    assert h_oil >= 0.75
    assert h_water < 0.15


def test_thermal_regime_penalties():
    """Verify thermal penalties for boiling prolonged on thermolabile and volatile molecules."""
    art_desc = calculate_descriptors(ARTEMISININ_SMILES)
    menthol_desc = calculate_descriptors(MENTHOL_SMILES)

    # Artemisinin: 0.70x penalty under boiling_prolonged
    art_cold = calculate_extraction_plausibility(art_desc, "water", "cold_ambient")
    art_boil = calculate_extraction_plausibility(art_desc, "water", "boiling_prolonged")
    assert art_cold > art_boil
    assert np.isclose(art_boil, art_cold * 0.70, atol=1e-3)

    # Menthol: 0.50x penalty for volatility under boiling_prolonged
    m_cold = calculate_extraction_plausibility(menthol_desc, "water", "cold_ambient")
    m_boil = calculate_extraction_plausibility(menthol_desc, "water", "boiling_prolonged")
    assert m_cold > m_boil
    assert np.isclose(m_boil, m_cold * 0.50, atol=1e-3)

    # Poultice/none: always returns 1.0
    assert calculate_extraction_plausibility(art_desc, "none", "cold_ambient") == 1.0


# ==============================================================================
# Requirement 5: Qualitative Binning Tests
# ==============================================================================

def test_verdict_binning():
    """Verify score mapping into categorical verdict bins."""
    assert map_plausibility_to_verdict(0.95) == "high"
    assert map_plausibility_to_verdict(0.75) == "high"
    assert map_plausibility_to_verdict(0.74) == "moderate"
    assert map_plausibility_to_verdict(0.40) == "moderate"
    assert map_plausibility_to_verdict(0.39) == "negligible"
    assert map_plausibility_to_verdict(0.15) == "negligible"
    assert map_plausibility_to_verdict(0.14) == "excluded"
    assert map_plausibility_to_verdict(0.00) == "excluded"


# ==============================================================================
# Model Integration Test: create_phytochemical_record
# ==============================================================================

def test_create_phytochemical_record():
    """Verify creation of a validated PhytochemicalRecord model."""
    record = create_phytochemical_record(
        compound_name="Rutin",
        smiles=RUTIN_SMILES,
        solvent_class="water",
        thermal_regime="warm_steep",
        provenance_doi="10.1016/j.phytochem.2021.1001",
    )
    assert isinstance(record, PhytochemicalRecord)
    assert record.compound_name == "Rutin"
    assert record.extraction_verdict == "high"
    assert record.extraction_plausibility_score >= 0.75
    assert record.provenance_doi == "10.1016/j.phytochem.2021.1001"
    assert record.is_thermolabile is False
    assert record.is_volatile is False

