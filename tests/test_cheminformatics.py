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
    compute_logd,
    compute_arrhenius_survival,
    compute_organ_factor,
    simulate_extraction_grid,
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


def test_aristolochic_acid_structural_alert_specificity():
    """Verify that bare phenanthrene does not trigger aristolochic acid derivative alert."""
    phenanthrene_smiles = "c1ccc2c(c1)ccc1ccccc21"
    desc_phen = calculate_descriptors(phenanthrene_smiles)
    assert desc_phen is not None
    assert "aristolochic_acid_derivative" not in desc_phen["toxicity_alerts"]

    # InChIKey dynamic calculation check
    rec = create_phytochemical_record(
        compound_name="Phenanthrene",
        smiles=phenanthrene_smiles,
    )
    assert rec is not None
    assert len(rec.inchi_key) == 27
    assert rec.inchi_key == desc_phen["inchi_key"]


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
    """Verify thermal penalties for boiling prolonged on thermolabile and volatile molecules via Arrhenius kinetics."""
    art_desc = calculate_descriptors(ARTEMISININ_SMILES)
    menthol_desc = calculate_descriptors(MENTHOL_SMILES)

    # Artemisinin: severe degradation under boiling_prolonged (100 C, 30 min) due to endoperoxide
    art_cold = calculate_extraction_plausibility(art_desc, "water", "cold_ambient")
    art_boil = calculate_extraction_plausibility(art_desc, "water", "boiling_prolonged")
    assert art_cold > art_boil
    assert art_boil < art_cold * 0.30  # >70% destroyed by 30 min boiling

    # Menthol: evaporative loss under open boiling_prolonged
    m_cold = calculate_extraction_plausibility(menthol_desc, "water", "cold_ambient")
    m_boil = calculate_extraction_plausibility(menthol_desc, "water", "boiling_prolonged")
    assert m_cold > m_boil
    assert m_boil < m_cold * 0.25  # Evaporative loss

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


# ==============================================================================
# Requirement: Physical Chemistry & Organ Localization Tests
# ==============================================================================

def test_compute_logd_alkaloid_vinegar_protonation():
    """Verify that basic alkaloids protonate at acidic pH (vinegar/wine), dropping logD into water-soluble range."""
    from rdkit import Chem
    # Quinine SMILES
    quinine_smi = "C=CC1CN2CCC1CC2C(O)c1ccnc2ccc(OC)cc12"
    mol = Chem.MolFromSmiles(quinine_smi)
    assert mol is not None

    clogp = 3.17
    # At neutral water pH 7.0
    logd_neutral, acid_base, pka = compute_logd(mol, clogp, ph=7.0)
    assert acid_base == "basic_alkaloid"
    assert pka == 8.8
    assert logd_neutral < clogp

    # In vinegar / sour wine pH 3.0: protonation drops logD by > 5 units!
    logd_vinegar, _, _ = compute_logd(mol, clogp, ph=3.0)
    assert logd_vinegar < -1.0  # Highly hydrophilic salt species
    assert logd_vinegar < logd_neutral - 3.0

    # In wood ash / alkaline pH 10.0: uncharged freebase, logD approaches neutral cLogP
    logd_alkaline, _, _ = compute_logd(mol, clogp, ph=10.0)
    assert np.isclose(logd_alkaline, clogp, atol=0.15)


def test_compute_logd_weak_acid_and_phenol():
    """Verify that carboxylic acids and phenols deprotonate at basic pH, dropping logD."""
    from rdkit import Chem
    # Salicylic acid: carboxylic acid + phenol
    acid_smi = "O=C(O)c1ccccc1O"
    mol = Chem.MolFromSmiles(acid_smi)
    assert mol is not None

    logd_ph3, acid_base, pka = compute_logd(mol, clogp=2.26, ph=3.0)
    assert acid_base == "weak_acid"
    assert pka == 4.2
    # At pH 3.0 (below pKa 4.2), mostly uncharged acid
    assert logd_ph3 > 1.8

    # At pH 8.0 (above pKa 4.2), deprotonated carboxylate anion drops logD
    logd_ph8, _, _ = compute_logd(mol, clogp=2.26, ph=8.0)
    assert logd_ph8 < -1.0


def test_compute_arrhenius_survival_artemisinin_cold_vs_boil():
    """Verify Arrhenius kinetic survival for Artemisinin: >99% at 25 C vs <20% at 100 C for 45 min."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(ARTEMISININ_SMILES)
    assert mol is not None

    # Cold wringing: 25 C for 60 min
    surv_cold, t_half, is_thermo = compute_arrhenius_survival(mol, temp_c=25.0, duration_min=60.0)
    assert is_thermo is True
    assert np.isclose(t_half, 15.07, atol=0.1)
    assert surv_cold > 0.99  # Preserved in cold preparation!

    # Traditional Chinese medicine prolonged decoction: 100 C for 45 min
    surv_boil, _, _ = compute_arrhenius_survival(mol, temp_c=100.0, duration_min=45.0)
    assert surv_boil < 0.15  # >85% thermally degraded, matching historical reports!


def test_compute_organ_factor():
    """Verify organ-level localization multipliers."""
    # Match: 1.0
    assert compute_organ_factor("root", ["root", "rhizome"]) == 1.0
    assert compute_organ_factor("whole_plant", ["leaf"]) == 1.0
    assert compute_organ_factor("leaf", []) == 1.0

    # Aerial parts relationship: 0.85
    assert compute_organ_factor("aerial_parts", ["flower"]) == 0.85
    assert compute_organ_factor("flower", ["aerial_parts"]) == 0.85

    # Strictly disjoint organs: 0.08
    assert compute_organ_factor("root", ["flower", "aerial_parts"]) == 0.08
    assert compute_organ_factor("bark", ["seed"]) == 0.08


def test_simulate_extraction_grid():
    """Verify generation of 2D in silico simulation grid for recipe optimization."""
    grid = simulate_extraction_grid(
        compound_smiles=ARTEMISININ_SMILES,
        tpsa=54.2,
        clogp=2.71,
        documented_organs=["leaf", "aerial_parts"],
        prepared_part="aerial_parts",
        temp_range=[25.0, 50.0, 80.0, 100.0],
        param_range=[2.0, 4.0, 7.0, 9.0],
        param_type="ph",
        solvent_class="water",
        fixed_duration_min=30.0,
    )
    assert "temp_axis" in grid
    assert "param_axis" in grid
    assert "plausibility_grid" in grid
    assert len(grid["plausibility_grid"]) == 4  # 4 temp points
    assert len(grid["plausibility_grid"][0]) == 4  # 4 pH points
    assert grid["organ_factor"] == 1.0
    # Higher temperature should have lower survival/plausibility due to thermal breakdown
    assert grid["plausibility_grid"][0][2] > grid["plausibility_grid"][3][2]

