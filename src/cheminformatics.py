"""DecoctionGraph Cheminformatics Scoring Engine.

Models physicochemical extraction plausibility of secondary metabolites under
traditional preparation regimes (solvent class, thermal conditions) using RDKit
and NumPy. Includes structural toxicity alert detection (SMARTS) and qualitative
verdict binning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.schemas import (
    ExtractionVerdict,
    PhytochemicalRecord,
    SolventClass,
    ThermalRegime,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Structural Toxicity Alert SMARTS Definitions
# ==============================================================================

# Compiled SMARTS patterns for toxic phytochemical scaffolds
STRUCTURAL_ALERT_DEFINITIONS: Dict[str, List[str]] = {
    "pyrrolizidine_alkaloid": [
        # Explicit spec pattern
        "[#6]1-[#6]-[#6]2-[#6](-[#6]-1)-[#7]-[#6]-[#6]=2",
        # Canonical 1,2-unsaturated necine base core (senecionine, retronecine, monocrotaline)
        "[#6]1[#6][#6][#7]2[#6][#6]=[#6][#6]12",
    ],
    "aristolochic_acid_derivative": [
        # Explicit spec pattern
        "c1cc2c(cc1)c1c(c(c2)O)c(=O)oc1",
        # Phenanthrene lactone / aristolic acid / aristolactam cores
        "c1ccc2c(c1)ccc1ccccc21",
        "c1cc2c(cc1)c1c(c2)c(=O)oc1",
    ],
    "ptaquiloside_illudane_skeleton": [
        # Spiro-cyclopropyl indanone illudane sesquiterpene core (ptaquiloside, pterosins)
        "[#6;r3]1-[#6;r3]-[#6;r3;!R1]1",
        # General cyclopropyl norsesquiterpene core
        "[#6]1-[#6]-[#6]1([#6])[#6]",
    ],
}

# Precompile SMARTS patterns for high performance
_COMPILED_ALERTS: Dict[str, List[Chem.Mol]] = {
    alert_name: [Chem.MolFromSmarts(pat) for pat in patterns if Chem.MolFromSmarts(pat) is not None]
    for alert_name, patterns in STRUCTURAL_ALERT_DEFINITIONS.items()
}

# Thermolabile / heat-sensitive structural motifs (endoperoxides, peroxides, strained lactones)
THERMOLABILE_SMARTS = [
    Chem.MolFromSmarts("[#8]-[#8]"),                  # Endoperoxide bridge / dialkyl peroxide
    Chem.MolFromSmarts("[#8]-[#8;H1]"),               # Hydroperoxide
    Chem.MolFromSmarts("[#6](=O)[#8]-[#8]"),          # Acyl peroxide
    Chem.MolFromSmarts("[#6;r4](=O)[#8;r4]"),         # Strained beta-lactone
    Chem.MolFromSmarts("[#6](=O)[#8][#6](=O)"),       # Anhydride
]
_COMPILED_THERMOLABILE = [p for p in THERMOLABILE_SMARTS if p is not None]


# ==============================================================================
# Structural Alert Detection
# ==============================================================================

def detect_structural_alerts(mol: Chem.Mol) -> List[str]:
    """Detect presence of toxic chemical scaffolds via SMARTS patterns.

    Scans for pyrrolizidine alkaloids, aristolochic acid derivatives, and
    ptaquiloside/illudane norsesquiterpene skeletons.

    Args:
        mol: Valid RDKit Mol object.

    Returns:
        List of detected alert names, or empty list if none detected.
    """
    detected: List[str] = []
    for alert_name, compiled_patterns in _COMPILED_ALERTS.items():
        if any(mol.HasSubstructMatch(pattern) for pattern in compiled_patterns):
            detected.append(alert_name)
    return detected


# ==============================================================================
# Molecular Property Calculation
# ==============================================================================

def calculate_descriptors(smiles: str) -> Optional[Dict[str, Any]]:
    """Compute 2D molecular physicochemical descriptors and flags using RDKit.

    Calculates:
      * Wildman-Crippen cLogP (Descriptors.MolLogP)
      * TPSA (rdMolDescriptors.CalcTPSA)
      * Molecular Weight (Descriptors.MolWt)
      * Hydrogen Bond Donors (rdMolDescriptors.CalcNumHBD)
      * Hydrogen Bond Acceptors (rdMolDescriptors.CalcNumHBA)
      * Rotatable Bonds (rdMolDescriptors.CalcNumRotatableBonds)
      * Volatility flag (MW < 220 and cLogP < 3.5 and TPSA < 40)
      * Thermolabile flag (endoperoxide bridges or labile linkages)
      * Structural toxicity alerts

    Args:
        smiles: Chemical structure string in SMILES notation.

    Returns:
        Dictionary of descriptor values and flags, or None if SMILES is invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    clean_smiles = smiles.strip()
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES string provided: '{clean_smiles}'")
        return None

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi_key = Chem.MolToInchiKey(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    mw = float(Descriptors.MolWt(mol))
    clogp = float(Descriptors.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

    # Flag is_volatile: MW < 220 and cLogP < 3.5 and TPSA < 40
    is_volatile = (mw < 220.0) and (clogp < 3.5) and (tpsa < 40.0)

    # Flag is_thermolabile: presence of endoperoxide bridges or thermolabile linkages
    is_thermolabile = any(mol.HasSubstructMatch(p) for p in _COMPILED_THERMOLABILE)

    # Toxicity alert detection
    alerts = detect_structural_alerts(mol)

    return {
        "smiles": canonical_smiles,
        "inchi_key": inchi_key,
        "molecular_formula": formula,
        "mw": round(mw, 2),
        "clogp": round(clogp, 2),
        "tpsa": round(tpsa, 2),
        "h_donors": hbd,
        "h_acceptors": hba,
        "rotatable_bonds": rotb,
        "is_volatile": is_volatile,
        "is_thermolabile": is_thermolabile,
        "toxicity_alerts": alerts,
    }


# ==============================================================================
# Sigmoidal Continuous Plausibility Scoring Functions
# ==============================================================================

def _sigmoid_range_desirability(
    val: float,
    low: float,
    high: float,
    k_low: float = 1.0,
    k_high: float = 2.0,
) -> float:
    """Evaluate desirability within an optimal range [low, high] with sigmoid roll-off."""
    if low <= val <= high:
        return 1.0
    elif val < low:
        diff = low - val
        return float(2.0 / (1.0 + np.exp(k_low * diff)))
    else:
        diff = val - high
        return float(2.0 / (1.0 + np.exp(k_high * diff)))


def _sigmoid_min_desirability(val: float, min_val: float, k: float = 1.5) -> float:
    """Evaluate one-sided desirability favoring val >= min_val with sigmoid penalty."""
    if val >= min_val:
        return 1.0
    diff = min_val - val
    return float(2.0 / (1.0 + np.exp(k * diff)))


def _sigmoid_max_desirability(val: float, max_val: float, k: float = 0.03) -> float:
    """Evaluate one-sided desirability favoring val <= max_val with sigmoid penalty."""
    if val <= max_val:
        return 1.0
    diff = val - max_val
    return float(2.0 / (1.0 + np.exp(k * diff)))


def calculate_extraction_plausibility(
    descriptors: Dict[str, Any],
    solvent_class: str,
    thermal_regime: str,
) -> float:
    """Calculate continuous extraction plausibility score [0.0, 1.0].

    Models solvent polarity partitioning and thermal degradation:
      * "water": Favors -2.5 <= cLogP <= 2.0 and TPSA >= 60.0. Penalizes cLogP > 3.0 heavily.
      * "lipid_oil": Favors cLogP >= 2.5 and TPSA <= 65.0. Penalizes hydrophilic (cLogP < 0.5, TPSA > 100).
      * "hydroalcoholic": Broad spectrum compatibility (-1.0 <= cLogP <= 4.5).
      * "none" (poultice/raw): Returns 1.0 (no solvent extraction filter applied).
      * "vinegar_acidic": Favors polar/basic species (-2.5 <= cLogP <= 2.5, TPSA >= 40.0).
      * "honey_syrup": Favors water-soluble polar glycosides (-2.5 <= cLogP <= 2.0, TPSA >= 50.0).
      * Thermal penalty: boiling_prolonged applies 0.70x if is_thermolabile and 0.50x if is_volatile.

    Args:
        descriptors: Dictionary returned by calculate_descriptors.
        solvent_class: Solvent class string (e.g., "water", "lipid_oil", "hydroalcoholic").
        thermal_regime: Thermal regime string (e.g., "boiling_prolonged", "cold_ambient").

    Returns:
        Continuous plausibility score between 0.0 and 1.0.
    """
    clogp = float(descriptors.get("clogp", 0.0))
    tpsa = float(descriptors.get("tpsa", 0.0))
    is_thermolabile = bool(descriptors.get("is_thermolabile", False))
    is_volatile = bool(descriptors.get("is_volatile", False))

    solvent = solvent_class.lower().strip()
    thermal = thermal_regime.lower().strip()

    # 1. Solvent polarity partitioning score
    if solvent == "water":
        # Favors -2.5 <= cLogP <= 2.0 and TPSA >= 60.0. Heavily penalize cLogP > 3.0.
        s_logp = _sigmoid_range_desirability(clogp, low=-2.5, high=2.0, k_low=0.8, k_high=3.0)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=60.0, k=0.03)
        base_score = s_logp * s_tpsa

    elif solvent == "lipid_oil":
        # Favors cLogP >= 2.5 and TPSA <= 65.0. Heavily penalize hydrophilic (cLogP < 0.5, TPSA > 100).
        s_logp = _sigmoid_min_desirability(clogp, min_val=2.5, k=1.5)
        s_tpsa = _sigmoid_max_desirability(tpsa, max_val=65.0, k=0.03)
        base_score = s_logp * s_tpsa

    elif solvent == "hydroalcoholic":
        # Broad spectrum. Compatible with -1.0 <= cLogP <= 4.5.
        s_logp = _sigmoid_range_desirability(clogp, low=-1.0, high=4.5, k_low=0.8, k_high=1.2)
        base_score = s_logp

    elif solvent == "none":
        # Poultice, mastication, or raw administration: no solvent filter applied
        base_score = 1.0

    elif solvent == "vinegar_acidic":
        # Aqueous acetic acid: favors polar and moderately polar molecules
        s_logp = _sigmoid_range_desirability(clogp, low=-2.5, high=2.5, k_low=0.8, k_high=2.0)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=40.0, k=0.03)
        base_score = s_logp * s_tpsa

    elif solvent == "honey_syrup":
        # Viscous aqueous sucrose/fructose matrix: favors polar glycosides
        s_logp = _sigmoid_range_desirability(clogp, low=-2.5, high=2.0, k_low=0.8, k_high=2.5)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=50.0, k=0.03)
        base_score = s_logp * s_tpsa

    else:
        # Unknown solvent class: baseline neutral plausibility
        base_score = 0.50

    # 2. Thermal energy adjustments
    final_score = base_score
    if thermal == "boiling_prolonged":
        if is_thermolabile:
            final_score *= 0.70
        if is_volatile:
            final_score *= 0.50
    elif thermal == "charred_ash":
        # Thermal combustion / destructive ashing destroys organic compounds
        final_score *= 0.05
    elif thermal == "warm_steep":
        # Mild heat: minor evaporation for highly volatiles
        if is_volatile:
            final_score *= 0.85
    elif thermal in ("cold_ambient", "none", "unknown"):
        pass

    # Bounded in [0.0, 1.0]
    return round(float(np.clip(final_score, 0.0, 1.0)), 4)


# ==============================================================================
# Categorical Verdict Binning
# ==============================================================================

def map_plausibility_to_verdict(score: float) -> ExtractionVerdict:
    """Map continuous plausibility score [0.0, 1.0] into qualitative verdict bins.

    Bins:
      * >= 0.75: "high"
      * 0.40 - 0.74: "moderate"
      * 0.15 - 0.39: "negligible"
      * < 0.15: "excluded"

    Args:
        score: Continuous plausibility score.

    Returns:
        Categorical ExtractionVerdict literal.
    """
    if score >= 0.75:
        return "high"
    elif score >= 0.40:
        return "moderate"
    elif score >= 0.15:
        return "negligible"
    else:
        return "excluded"


# ==============================================================================
# High-Level Factory Utility
# ==============================================================================

def create_phytochemical_record(
    compound_name: str,
    smiles: str,
    solvent_class: str = "water",
    thermal_regime: str = "boiling_prolonged",
    provenance_doi: Optional[str] = None,
) -> Optional[PhytochemicalRecord]:
    """Calculate descriptors, score extraction plausibility, and assemble PhytochemicalRecord.

    Args:
        compound_name: IUPAC or common name of compound.
        smiles: Chemical structure string in SMILES notation.
        solvent_class: Solvent class to evaluate.
        thermal_regime: Thermal extraction conditions.
        provenance_doi: Optional literature citation or DOI.

    Returns:
        Fully populated and validated PhytochemicalRecord, or None if SMILES is invalid.
    """
    desc = calculate_descriptors(smiles)
    if desc is None:
        return None

    score = calculate_extraction_plausibility(
        descriptors=desc,
        solvent_class=solvent_class,
        thermal_regime=thermal_regime,
    )
    verdict = map_plausibility_to_verdict(score)

    return PhytochemicalRecord(
        compound_name=compound_name,
        smiles=desc["smiles"],
        inchi_key=desc["inchi_key"],
        molecular_formula=desc["molecular_formula"],
        mw=desc["mw"],
        clogp=desc["clogp"],
        tpsa=desc["tpsa"],
        h_donors=desc["h_donors"],
        h_acceptors=desc["h_acceptors"],
        rotatable_bonds=desc["rotatable_bonds"],
        is_thermolabile=desc["is_thermolabile"],
        is_volatile=desc["is_volatile"],
        toxicity_alerts=desc["toxicity_alerts"],
        extraction_plausibility_score=score,
        extraction_verdict=verdict,
        provenance_doi=provenance_doi,
    )


# ==============================================================================
# CLI Entry Point & Unit Test Assertions
# ==============================================================================

if __name__ == "__main__":
    print("=" * 90)
    print("DecoctionGraph: Cheminformatics Scoring Engine Verification")
    print("=" * 90)

    # Test Structures
    RUTIN_SMILES = "CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C=C5)O)O)O)O)O)O)O)O"
    HYPERFORIN_SMILES = "CC(C)C(=O)C12C(=O)C(=C(C(C1=O)(CC(C2(C)CCC=C(C)C)CC=C(C)C)CC=C(C)C)O)CC=C(C)C"
    ARTEMISININ_SMILES = "CC1CCC2C(C(=O)OC3C24C1CCC(O3)(OO4)C)C"

    # 1. Calculate descriptors
    rutin_desc = calculate_descriptors(RUTIN_SMILES)
    hyperforin_desc = calculate_descriptors(HYPERFORIN_SMILES)
    artemisinin_desc = calculate_descriptors(ARTEMISININ_SMILES)

    assert rutin_desc is not None, "Rutin descriptors calculation failed"
    assert hyperforin_desc is not None, "Hyperforin descriptors calculation failed"
    assert artemisinin_desc is not None, "Artemisinin descriptors calculation failed"

    print("\nMolecular Descriptors:")
    print(f"  * Rutin:       MW={rutin_desc['mw']}, cLogP={rutin_desc['clogp']}, TPSA={rutin_desc['tpsa']}, Thermolabile={rutin_desc['is_thermolabile']}")
    print(f"  * Hyperforin:  MW={hyperforin_desc['mw']}, cLogP={hyperforin_desc['clogp']}, TPSA={hyperforin_desc['tpsa']}, Volatile={hyperforin_desc['is_volatile']}")
    print(f"  * Artemisinin: MW={artemisinin_desc['mw']}, cLogP={artemisinin_desc['clogp']}, TPSA={artemisinin_desc['tpsa']}, Thermolabile={artemisinin_desc['is_thermolabile']}")

    # 2. Rutin (polar flavonoid) in water vs. lipid_oil
    rutin_water_score = calculate_extraction_plausibility(rutin_desc, "water", "cold_ambient")
    rutin_water_verdict = map_plausibility_to_verdict(rutin_water_score)
    rutin_oil_score = calculate_extraction_plausibility(rutin_desc, "lipid_oil", "cold_ambient")
    rutin_oil_verdict = map_plausibility_to_verdict(rutin_oil_score)

    print(f"\n1. Rutin (Polar Flavonoid):")
    print(f"   - Water:     score={rutin_water_score:.4f} -> verdict={rutin_water_verdict}")
    print(f"   - Lipid Oil: score={rutin_oil_score:.4f} -> verdict={rutin_oil_verdict}")

    assert rutin_water_score > rutin_oil_score, "Rutin should have higher plausibility in water than in lipid_oil"
    assert rutin_water_verdict == "high", f"Rutin in water should be 'high', got {rutin_water_verdict}"
    assert rutin_oil_verdict == "excluded", f"Rutin in lipid_oil should be 'excluded', got {rutin_oil_verdict}"

    # 3. Hyperforin (lipophilic phloroglucinol) in water vs. lipid_oil
    hyp_water_score = calculate_extraction_plausibility(hyperforin_desc, "water", "cold_ambient")
    hyp_water_verdict = map_plausibility_to_verdict(hyp_water_score)
    hyp_oil_score = calculate_extraction_plausibility(hyperforin_desc, "lipid_oil", "cold_ambient")
    hyp_oil_verdict = map_plausibility_to_verdict(hyp_oil_score)

    print(f"\n2. Hyperforin (Lipophilic Phloroglucinol):")
    print(f"   - Water:     score={hyp_water_score:.4f} -> verdict={hyp_water_verdict}")
    print(f"   - Lipid Oil: score={hyp_oil_score:.4f} -> verdict={hyp_oil_verdict}")

    assert hyp_oil_score > hyp_water_score, "Hyperforin should have higher plausibility in lipid_oil than in water"
    assert hyp_oil_verdict == "high", f"Hyperforin in lipid_oil should be 'high', got {hyp_oil_verdict}"
    assert hyp_water_verdict == "excluded", f"Hyperforin in water should be 'excluded', got {hyp_water_verdict}"

    # 4. Artemisinin (endoperoxide) in cold water vs. prolonged boiling
    art_cold_score = calculate_extraction_plausibility(artemisinin_desc, "water", "cold_ambient")
    art_cold_verdict = map_plausibility_to_verdict(art_cold_score)
    art_boil_score = calculate_extraction_plausibility(artemisinin_desc, "water", "boiling_prolonged")
    art_boil_verdict = map_plausibility_to_verdict(art_boil_score)

    print(f"\n3. Artemisinin (Thermolabile Endoperoxide):")
    print(f"   - Cold Water:         score={art_cold_score:.4f} -> verdict={art_cold_verdict}")
    print(f"   - Prolonged Boiling:  score={art_boil_score:.4f} -> verdict={art_boil_verdict}")

    assert artemisinin_desc["is_thermolabile"] is True, "Artemisinin must be detected as thermolabile"
    assert art_cold_score > art_boil_score, "Artemisinin should have higher plausibility in cold water than prolonged boiling"
    assert np.isclose(art_boil_score, art_cold_score * 0.70, atol=1e-3), "Prolonged boiling must apply 0.70x thermal penalty"

    print("\n" + "=" * 90)
    print("SUCCESS: All unit test assertions passed successfully!")
    print("=" * 90)

