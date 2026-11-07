"""DecoctionGraph Cheminformatics Scoring Engine.

Models physical chemistry extraction plausibility of secondary metabolites under
traditional preparation regimes using RDKit and NumPy.
Includes:
  * SMARTS-based ionization detection and Henderson-Hasselbalch logD(pH) calculation
  * Arrhenius first-order thermal degradation kinetics
  * Organ-specific localization multipliers
  * Structural toxicity alert detection (SMARTS)
  * Vectorized 2D simulation grid generation for in silico recipe optimization
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.schemas import (
    AcidBaseClass,
    ExtractionVerdict,
    PhytochemicalRecord,
    SolventClass,
    TargetMechanismRecord,
    ThermalRegime,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Structural Toxicity Alert SMARTS Definitions
# ==============================================================================

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
        # Phenanthrene lactone / aristolic acid / aristolactam core
        "c1cc2c(cc1)c1c(c2)c(=O)oc1",
    ],
    "ptaquiloside_illudane_skeleton": [
        # Spiro-cyclopropyl indanone illudane sesquiterpene core (ptaquiloside, pterosins)
        "[#6;r3]1-[#6;r3]-[#6;r3;!R1]1",
        # General cyclopropyl norsesquiterpene core
        "[#6]1-[#6]-[#6]1([#6])[#6]",
    ],
}

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

# Strained polyprenylated acylphloroglucinols (e.g. hyperforin) / labile beta-dicarbonyls
_SMARTS_PHLOROGLUCINOL = Chem.MolFromSmarts("[#6]1(=O)[#6][#6](=O)[#6][#6](=O)[#6]1")

# Ionization functional groups
_SMARTS_BASIC_AMINE = Chem.MolFromSmarts("[$([NX3;H2,H1,H0;!$(NC=O);!$(Nc)])]")
_SMARTS_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
_SMARTS_PHENOL = Chem.MolFromSmarts("[OX2H][cX3]:[c]")


# ==============================================================================
# Structural Alert Detection
# ==============================================================================

def detect_structural_alerts(mol: Union[Chem.Mol, str]) -> List[str]:
    """Detect presence of toxic chemical scaffolds via SMARTS patterns.

    Scans for pyrrolizidine alkaloids, aristolochic acid derivatives, and
    ptaquiloside/illudane norsesquiterpene skeletons.

    Args:
        mol: Valid RDKit Mol object or SMILES string.

    Returns:
        List of detected alert names, or empty list if none detected.
    """
    if isinstance(mol, str):
        rd_mol = Chem.MolFromSmiles(mol)
        if rd_mol is None:
            return []
        mol = rd_mol

    detected: List[str] = []
    for alert_name, compiled_patterns in _COMPILED_ALERTS.items():
        if any(mol.HasSubstructMatch(pattern) for pattern in compiled_patterns):
            detected.append(alert_name)
    return detected


# ==============================================================================
# SMARTS-Based Ionization & logD(pH) Calculator
# ==============================================================================

def compute_logd(
    mol: Chem.Mol,
    clogp: float,
    ph: float,
) -> Tuple[float, str, Optional[float]]:
    """Calculate pH-dependent distribution coefficient logD via Henderson-Hasselbalch equation.

    Detects:
      * Basic amine / alkaloid centers: pKa = 8.8
      * Carboxylic acids: pKa = 4.2
      * Phenolic hydroxyls: pKa = 9.6
      * Zwitterionic centers (both basic amine and carboxylic acid)
      * Neutral molecules

    Henderson-Hasselbalch Equations:
      * Base: logD = clogp - log10(1 + 10^(pKa - pH))
      * Acid/Phenol: logD = clogp - log10(1 + 10^(pH - pKa))
      * Neutral: logD = clogp

    Args:
        mol: Valid RDKit Mol instance.
        clogp: Wildman-Crippen neutral logP.
        ph: Solvent pH (e.g. 3.0 for vinegar, 7.0 for neutral water, 9.5 for wood ash).

    Returns:
        Tuple of (logd, acid_base_class, estimated_pka).
    """
    has_amine = mol.HasSubstructMatch(_SMARTS_BASIC_AMINE) if _SMARTS_BASIC_AMINE else False
    has_acid = mol.HasSubstructMatch(_SMARTS_CARBOXYL) if _SMARTS_CARBOXYL else False
    has_phenol = mol.HasSubstructMatch(_SMARTS_PHENOL) if _SMARTS_PHENOL else False

    if has_amine and has_acid:
        acid_base_class = "zwitterion"
        pka: Optional[float] = 8.8
        # Dually ionized partition penalty
        penalty_base = math.log10(1.0 + 10.0 ** max(-10.0, min(10.0, 8.8 - ph)))
        penalty_acid = math.log10(1.0 + 10.0 ** max(-10.0, min(10.0, ph - 4.2)))
        logd = clogp - penalty_base - penalty_acid
    elif has_amine:
        acid_base_class = "basic_alkaloid"
        pka = 8.8
        delta = max(-10.0, min(10.0, pka - ph))
        logd = clogp - math.log10(1.0 + 10.0 ** delta)
    elif has_acid:
        acid_base_class = "weak_acid"
        pka = 4.2
        delta = max(-10.0, min(10.0, ph - pka))
        logd = clogp - math.log10(1.0 + 10.0 ** delta)
    elif has_phenol:
        acid_base_class = "phenolic"
        pka = 9.6
        delta = max(-10.0, min(10.0, ph - pka))
        logd = clogp - math.log10(1.0 + 10.0 ** delta)
    else:
        acid_base_class = "neutral"
        pka = None
        logd = clogp

    return round(float(logd), 2), acid_base_class, pka


# ==============================================================================
# Arrhenius Thermal Degradation Kinetics
# ==============================================================================

def compute_arrhenius_survival(
    mol: Chem.Mol,
    temp_c: float,
    duration_min: float,
    preparation_type: str = "decoction",
    mw: float = 300.0,
    tpsa: float = 60.0,
) -> Tuple[float, Optional[float], bool]:
    """Model first-order thermal degradation and volatile evaporative loss via Arrhenius kinetics.

    Equation:
      Remaining(t, T) = exp(-k(T) * duration_min)
      k(T) = k_ref * exp((-Ea / R) * (1/T_kelvin - 1/373.15))
      where T_ref = 373.15 K (100 C) and R = 8.314e-3 kJ/(mol*K).

    Thermal Scaffold Regimes:
      * Endoperoxides (Artemisinin): k_ref = 0.046 min^-1, Ea = 85.0 kJ/mol (half-life ~15 min at 100 C).
      * Labile acylphloroglucinols / secoiridoids: k_ref = 0.012 min^-1, Ea = 75.0 kJ/mol (half-life ~58 min).
      * Volatile monoterpenes: MW < 210, TPSA < 45 at temp >= 90 C: k_evap = 0.06 min^-1, Ea = 40.0 kJ/mol.
      * Stable scaffolds (flavonoids, alkaloids, phenolics): k_ref = 0.0005 min^-1, Ea = 60.0 kJ/mol.

    Args:
        mol: Valid RDKit Mol instance.
        temp_c: Temperature in Celsius (e.g. 25 for cold wringing, 100 for boiling).
        duration_min: Contact or boiling time in minutes.
        preparation_type: Preparation type string (e.g. "decoction", "steam_inhalation").
        mw: Molecular weight in Daltons.
        tpsa: Topological polar surface area.

    Returns:
        Tuple of (survival_fraction, half_life_min_at_100c, is_thermolabile).
    """
    temp_safe = max(0.0, float(temp_c))
    t_kelvin = temp_safe + 273.15
    t_ref = 373.15  # 100 C in Kelvin
    r_const = 8.314e-3  # kJ / (mol * K)

    has_peroxide = any(mol.HasSubstructMatch(p) for p in _COMPILED_THERMOLABILE)
    has_phloroglucinol = mol.HasSubstructMatch(_SMARTS_PHLOROGLUCINOL) if _SMARTS_PHLOROGLUCINOL else False
    is_volatile = (mw < 210.0) and (tpsa < 45.0)

    if has_peroxide:
        k_ref = 0.046
        ea = 85.0
        is_thermolabile = True
    elif has_phloroglucinol:
        k_ref = 0.012
        ea = 75.0
        is_thermolabile = True
    elif is_volatile and temp_safe >= 85.0 and preparation_type.lower() != "steam_inhalation":
        # Evaporative loss in open boiling
        k_ref = 0.060
        ea = 40.0
        is_thermolabile = False
    else:
        k_ref = 0.0005
        ea = 60.0
        is_thermolabile = False

    # Arrhenius rate constant calculation
    exponent = (-ea / r_const) * (1.0 / t_kelvin - 1.0 / t_ref)
    exponent_clamped = max(-25.0, min(10.0, exponent))
    k_t = k_ref * math.exp(exponent_clamped)

    half_life_100c = round(math.log(2.0) / k_ref, 2)
    duration_safe = max(0.0, float(duration_min))

    survival = math.exp(-k_t * duration_safe)
    survival_clamped = float(np.clip(survival, 0.0, 1.0))

    return round(survival_clamped, 4), half_life_100c, is_thermolabile


# ==============================================================================
# Organ-Specific Localization Multiplier
# ==============================================================================

def compute_organ_factor(
    prepared_part: str,
    documented_organs: Sequence[str],
) -> float:
    """Evaluate anatomical plant organ alignment between recipe and metabolite localization.

    Rules:
      * 1.00: prepared_part matches documented_organs, or whole_plant is used, or documented_organs is empty/unknown.
      * 0.85: prepared_part == "aerial_parts" and compound is documented in "leaf" or "flower" (or vice versa).
      * 0.08: prepared_part is strictly disjoint from documented_organs (e.g. recipe boils root, while compound
              is exclusively biosynthesized in flower).

    Args:
        prepared_part: Anatomical part extracted in recipe (e.g. "root", "leaf", "flower", "whole_plant").
        documented_organs: List of organs where metabolite is documented (e.g. ["flower", "aerial_parts"]).

    Returns:
        Organ multiplier factor between 0.0 and 1.0.
    """
    part = str(prepared_part).lower().strip()
    organs = [str(o).lower().strip() for o in documented_organs if o and str(o).strip()]

    # If no documented organs recorded or recipe utilizes whole plant
    if not organs or part in ("whole_plant", "unknown", "") or any(o in ("whole_plant", "unknown") for o in organs):
        return 1.0

    # Direct match
    if part in organs:
        return 1.0

    # Aerial parts relationship
    aerial_components = {"leaf", "flower", "seed", "fruit", "aerial_parts"}
    if part == "aerial_parts" and any(o in aerial_components for o in organs):
        return 0.85
    if part in ("leaf", "flower", "seed", "fruit") and "aerial_parts" in organs:
        return 0.85

    # Root and rhizome relationship
    if part in ("root", "rhizome") and any(o in ("root", "rhizome") for o in organs):
        return 1.0

    # Disjoint organ restriction
    return 0.08


# ==============================================================================
# Molecular Property Calculation
# ==============================================================================

def calculate_descriptors(smiles: str, ph: float = 7.0) -> Optional[Dict[str, Any]]:
    """Compute 2D molecular descriptors, logD(pH), and toxicity alerts using RDKit.

    Calculates:
      * Wildman-Crippen cLogP (Descriptors.MolLogP)
      * TPSA (rdMolDescriptors.CalcTPSA)
      * Molecular Weight (Descriptors.MolWt)
      * Hydrogen Bond Donors & Acceptors
      * Rotatable Bonds
      * Acid-base class and logD at specified pH
      * Volatility and thermolability flags
      * Structural toxicity alerts

    Args:
        smiles: Chemical structure string in SMILES notation.
        ph: Solvent pH for logD estimation (default: 7.0).

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

    # Flag is_thermolabile: presence of endoperoxide bridges or labile motifs
    is_thermolabile = any(mol.HasSubstructMatch(p) for p in _COMPILED_THERMOLABILE)
    if not is_thermolabile and _SMARTS_PHLOROGLUCINOL:
        is_thermolabile = mol.HasSubstructMatch(_SMARTS_PHLOROGLUCINOL)

    # SMARTS-based ionization & logD
    logd, acid_base_class, pka = compute_logd(mol, clogp, ph)

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
        "acid_base_class": acid_base_class,
        "estimated_pka": pka,
        "logd_at_recipe_ph": logd,
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
    estimated_ph: Optional[float] = None,
    duration_minutes: Optional[float] = None,
    temperature_celsius: Optional[float] = None,
    prepared_part: str = "unknown",
    documented_organs: Optional[Sequence[str]] = None,
    preparation_type: str = "decoction",
) -> float:
    """Calculate continuous extraction plausibility score [0.0, 1.0] using unified physical chemistry.

    Formula:
      S_ext = S_solvent(logD, TPSA, solvent_class) * thermal_survival_fraction * organ_match_factor

    Args:
        descriptors: Dictionary returned by calculate_descriptors.
        solvent_class: Solvent class string (e.g. "water", "lipid_oil", "vinegar_acidic").
        thermal_regime: Thermal regime string (e.g. "boiling_prolonged", "warm_steep").
        estimated_ph: Optional solvent pH (inferred from solvent if None).
        duration_minutes: Optional contact time in minutes (inferred from thermal regime if None).
        temperature_celsius: Optional temperature in Celsius (inferred from thermal regime if None).
        prepared_part: Plant organ extracted (e.g. "root", "leaf", "flower").
        documented_organs: Plant organs where compound accumulates.
        preparation_type: Preparation modality (e.g. "decoction", "infusion").

    Returns:
        Continuous plausibility score between 0.0 and 1.0.
    """
    clogp = float(descriptors.get("clogp", 0.0))
    tpsa = float(descriptors.get("tpsa", 0.0))
    mw = float(descriptors.get("mw", 300.0))
    smiles = descriptors.get("smiles", "")
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    solvent = solvent_class.lower().strip()
    thermal = thermal_regime.lower().strip()

    # 1. Infer physicochemical operating conditions if not provided
    if estimated_ph is not None:
        ph = float(estimated_ph)
    elif solvent == "vinegar_acidic":
        ph = 3.0
    elif thermal == "charred_ash":
        ph = 9.5
    else:
        ph = 7.0

    if temperature_celsius is not None:
        temp_c = float(temperature_celsius)
    elif thermal == "boiling_prolonged":
        temp_c = 100.0
    elif thermal == "warm_steep":
        temp_c = 80.0
    elif thermal == "charred_ash":
        temp_c = 250.0
    else:
        temp_c = 25.0

    if duration_minutes is not None:
        duration_min = float(duration_minutes)
    elif thermal == "boiling_prolonged":
        duration_min = 30.0
    elif thermal == "warm_steep":
        duration_min = 60.0
    elif solvent in ("lipid_oil", "hydroalcoholic"):
        duration_min = 1440.0
    else:
        duration_min = 15.0

    # 2. Determine logD at recipe pH
    if mol is not None:
        logd, _, _ = compute_logd(mol, clogp, ph)
    else:
        logd = float(descriptors.get("logd_at_recipe_ph", clogp))

    # 3. Solvent partitioning desirability S_solvent(logD, TPSA, solvent)
    if solvent == "water":
        # Favors -2.5 <= logD <= 2.0 and TPSA >= 60.0. Penalizes logD > 3.0 heavily.
        s_logd = _sigmoid_range_desirability(logd, low=-2.5, high=2.0, k_low=0.8, k_high=3.0)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=60.0, k=0.03)
        s_solvent = s_logd * s_tpsa

    elif solvent == "lipid_oil":
        # Favors logD >= 2.5 and TPSA <= 65.0. Heavily penalizes hydrophilic (logD < 0.5, TPSA > 100).
        s_logd = _sigmoid_min_desirability(logd, min_val=2.5, k=1.5)
        s_tpsa = _sigmoid_max_desirability(tpsa, max_val=65.0, k=0.03)
        s_solvent = s_logd * s_tpsa

    elif solvent == "hydroalcoholic":
        # Broad spectrum compatibility with -1.0 <= logD <= 4.5.
        s_logd = _sigmoid_range_desirability(logd, low=-1.0, high=4.5, k_low=0.8, k_high=1.2)
        s_solvent = s_logd

    elif solvent == "none":
        # Poultice or raw: no vehicle filter applied
        s_solvent = 1.0

    elif solvent == "vinegar_acidic":
        # Aqueous acetic acid: favors polar species (-2.5 <= logD <= 2.5, TPSA >= 40.0)
        s_logd = _sigmoid_range_desirability(logd, low=-2.5, high=2.5, k_low=0.8, k_high=2.0)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=40.0, k=0.03)
        s_solvent = s_logd * s_tpsa

    elif solvent == "honey_syrup":
        # Viscous aqueous sucrose/fructose matrix: favors polar glycosides
        s_logd = _sigmoid_range_desirability(logd, low=-2.5, high=2.0, k_low=0.8, k_high=2.5)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=50.0, k=0.03)
        s_solvent = s_logd * s_tpsa

    else:
        # Unknown solvent class: baseline neutral plausibility
        s_solvent = 0.50

    # 4. Arrhenius thermal survival fraction
    if solvent == "none" or thermal in ("none", "cold_ambient"):
        survival_fraction = 1.0
    elif mol is not None:
        survival_fraction, _, _ = compute_arrhenius_survival(
            mol=mol,
            temp_c=temp_c,
            duration_min=duration_min,
            preparation_type=preparation_type,
            mw=mw,
            tpsa=tpsa,
        )
    else:
        # Fallback to legacy thermal multipliers if mol object unavailable
        survival_fraction = 1.0
        if thermal == "boiling_prolonged":
            if bool(descriptors.get("is_thermolabile", False)):
                survival_fraction *= 0.70
            if bool(descriptors.get("is_volatile", False)):
                survival_fraction *= 0.50
        elif thermal == "warm_steep" and bool(descriptors.get("is_volatile", False)):
            survival_fraction *= 0.85

    # 5. Organ-specific localization multiplier
    organs = documented_organs if documented_organs is not None else descriptors.get("documented_organs", [])
    organ_factor = compute_organ_factor(prepared_part, organs)

    # 6. Unified Plausibility Score
    final_score = s_solvent * survival_fraction * organ_factor
    if thermal == "charred_ash":
        final_score *= 0.05

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
# Vectorized Simulation Grid for In Silico Recipe Optimization
# ==============================================================================

def simulate_extraction_grid(
    compound_smiles: str,
    tpsa: float,
    clogp: float,
    documented_organs: Sequence[str],
    prepared_part: str = "unknown",
    temp_range: Optional[Sequence[float]] = None,
    param_range: Optional[Sequence[float]] = None,
    param_type: str = "ph",  # "ph" or "duration"
    solvent_class: str = "water",
    fixed_duration_min: float = 15.0,
    fixed_ph: float = 7.0,
) -> Dict[str, Any]:
    """Generate a 2D physical chemistry simulation grid for interactive recipe optimization.

    Simulates extraction yield over a temperature axis and a secondary parameter axis (pH or duration).

    Args:
        compound_smiles: SMILES string of the compound.
        tpsa: Topological polar surface area.
        clogp: Neutral wildman-crippen cLogP.
        documented_organs: Organs where compound accumulates.
        prepared_part: Anatomical part extracted.
        temp_range: Sequence of temperatures in Celsius (default: 20 to 100 in 9 steps).
        param_range: Sequence of pH (2 to 10) or durations (5 to 120 min).
        param_type: Secondary parameter: "ph" or "duration".
        solvent_class: Extraction vehicle (default: "water").
        fixed_duration_min: Duration used when param_type == "ph".
        fixed_ph: Solvent pH used when param_type == "duration".

    Returns:
        Dictionary containing grid matrixes (plausibility, logD, survival) and axis metadata.
    """
    mol = Chem.MolFromSmiles(compound_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string for simulation: {compound_smiles}")

    desc = calculate_descriptors(compound_smiles) or {}
    mw = float(desc.get("mw", Descriptors.MolWt(mol)))

    temps = list(temp_range) if temp_range is not None else [float(t) for t in np.linspace(20, 100, 9)]
    if param_type == "ph":
        params = list(param_range) if param_range is not None else [float(p) for p in np.linspace(2.0, 10.0, 9)]
    else:
        params = list(param_range) if param_range is not None else [5.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0]

    organ_factor = compute_organ_factor(prepared_part, documented_organs)

    plausibility_matrix: List[List[float]] = []
    logd_matrix: List[List[float]] = []
    survival_matrix: List[List[float]] = []

    for t in temps:
        p_row: List[float] = []
        l_row: List[float] = []
        s_row: List[float] = []
        for p in params:
            cur_ph = p if param_type == "ph" else fixed_ph
            cur_dur = p if param_type == "duration" else fixed_duration_min

            # Calculate logD
            logd, _, _ = compute_logd(mol, clogp, cur_ph)
            l_row.append(logd)

            # Solvent partitioning S_solvent
            if solvent_class == "water":
                s_logd = _sigmoid_range_desirability(logd, low=-2.5, high=2.0, k_low=0.8, k_high=3.0)
                s_tpsa = _sigmoid_min_desirability(tpsa, min_val=60.0, k=0.03)
                s_solvent = s_logd * s_tpsa
            elif solvent_class == "lipid_oil":
                s_logd = _sigmoid_min_desirability(logd, min_val=2.5, k=1.5)
                s_tpsa = _sigmoid_max_desirability(tpsa, max_val=65.0, k=0.03)
                s_solvent = s_logd * s_tpsa
            else:
                s_solvent = _sigmoid_range_desirability(logd, low=-1.0, high=4.5, k_low=0.8, k_high=1.2)

            # Arrhenius survival
            surv, _, _ = compute_arrhenius_survival(
                mol=mol, temp_c=t, duration_min=cur_dur, mw=mw, tpsa=tpsa
            )
            s_row.append(surv)

            total = float(np.clip(s_solvent * surv * organ_factor, 0.0, 1.0))
            p_row.append(round(total, 4))

        plausibility_matrix.append(p_row)
        logd_matrix.append(l_row)
        survival_matrix.append(s_row)

    return {
        "temp_axis": temps,
        "param_axis": params,
        "param_type": param_type,
        "solvent_class": solvent_class,
        "organ_factor": organ_factor,
        "plausibility_grid": plausibility_matrix,
        "logd_grid": logd_matrix,
        "survival_grid": survival_matrix,
    }


def calculate_recipe_yield(
    compound_smiles: str,
    solvent_mode: str = "water",
    temp_c: float = 25.0,
    ph: float = 7.0,
    duration_min: float = 30.0,
    ethanol_pct: float = 0.0,
    prepared_part: str = "unknown",
    documented_organs: Optional[Sequence[str]] = None,
) -> float:
    """Calculate the comprehensive physical chemistry extraction yield [0.0, 1.0].

    Integrates:
      1. SMARTS-based ionization & Henderson-Hasselbalch logD(pH).
      2. Solvent partitioning S_solvent (supporting water, hydroalcoholic mixtures with ethanol %, lipid oil, vinegar).
      3. First-order Arrhenius thermal degradation kinetics exp(-k(T)*t).
      4. Organ-level anatomical localization factor.

    Args:
        compound_smiles: Chemical SMILES string.
        solvent_mode: Extraction vehicle: "water", "hydroalcoholic", "lipid_oil", "vinegar_acidic".
        temp_c: Extraction temperature in Celsius.
        ph: Solvent pH.
        duration_min: Contact/boiling time in minutes.
        ethanol_pct: Ethanol volume percentage [0.0, 95.0] for hydroalcoholic mixtures.
        prepared_part: Anatomical plant organ extracted (e.g. "leaf", "root").
        documented_organs: Documented organs where compound accumulates.

    Returns:
        Continuous yield between 0.0 and 1.0.
    """
    mol = Chem.MolFromSmiles(compound_smiles)
    if mol is None:
        return 0.0

    desc = calculate_descriptors(compound_smiles, ph=ph)
    if desc is None:
        return 0.0

    clogp = float(desc["clogp"])
    tpsa = float(desc["tpsa"])
    mw = float(desc["mw"])

    # 1. logD at pH
    logd, _, _ = compute_logd(mol, clogp, ph)

    # 2. Vehicle solubility S_solvent
    s_mode = solvent_mode.lower().strip()
    if s_mode == "lipid_oil":
        s_logd = _sigmoid_min_desirability(logd, min_val=2.5, k=1.5)
        s_tpsa = _sigmoid_max_desirability(tpsa, max_val=65.0, k=0.03)
        s_solvent = s_logd * s_tpsa
    elif s_mode == "vinegar_acidic" or ph <= 3.8:
        s_logd = _sigmoid_range_desirability(logd, low=-3.0, high=1.8, k_low=0.7, k_high=3.0)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=40.0, k=0.03)
        s_solvent = s_logd * s_tpsa
    elif s_mode == "hydroalcoholic":
        e = float(np.clip(ethanol_pct, 0.0, 95.0))
        low_bound = -2.5 + (e / 95.0) * 2.0
        high_bound = 2.0 + (e / 95.0) * 4.0
        min_tpsa = 60.0 - (e / 95.0) * 45.0
        s_logd = _sigmoid_range_desirability(logd, low=low_bound, high=high_bound, k_low=0.8, k_high=1.5)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=min_tpsa, k=0.03)
        s_solvent = s_logd * s_tpsa
    else:  # water
        s_logd = _sigmoid_range_desirability(logd, low=-2.5, high=2.0, k_low=0.8, k_high=3.0)
        s_tpsa = _sigmoid_min_desirability(tpsa, min_val=60.0, k=0.03)
        s_solvent = s_logd * s_tpsa

    # 3. Arrhenius thermal survival
    survival_fraction, _, _ = compute_arrhenius_survival(
        mol=mol, temp_c=temp_c, duration_min=duration_min, mw=mw, tpsa=tpsa
    )

    # 4. Organ factor
    organs = documented_organs if documented_organs is not None else []
    organ_factor = compute_organ_factor(prepared_part, organs)

    # 5. Final yield
    yield_val = float(np.clip(s_solvent * survival_fraction * organ_factor, 0.0, 1.0))
    return round(yield_val, 4)


# ==============================================================================
# High-Level Factory Utility
# ==============================================================================

def create_phytochemical_record(
    compound_name: str,
    smiles: str,
    solvent_class: str = "water",
    thermal_regime: str = "boiling_prolonged",
    provenance_doi: Optional[str] = None,
    estimated_ph: Optional[float] = None,
    duration_minutes: Optional[float] = None,
    temperature_celsius: Optional[float] = None,
    prepared_part: str = "unknown",
    documented_organs: Optional[List[str]] = None,
    known_targets: Optional[List[TargetMechanismRecord]] = None,
    mechanistic_alignment_score: float = 0.0,
) -> Optional[PhytochemicalRecord]:
    """Calculate descriptors, score physical chemistry extraction plausibility, and assemble PhytochemicalRecord.

    Args:
        compound_name: IUPAC or common name of compound.
        smiles: Chemical structure string in SMILES notation.
        solvent_class: Solvent class to evaluate.
        thermal_regime: Thermal extraction conditions.
        provenance_doi: Optional literature citation or DOI.
        estimated_ph: Optional solvent pH.
        duration_minutes: Optional extraction duration.
        temperature_celsius: Optional extraction temperature.
        prepared_part: Anatomical plant organ extracted.
        documented_organs: Plant organs where compound is documented.
        known_targets: List of protein targets modulated by this compound.
        mechanistic_alignment_score: Relevance score linking targets to historical indication.

    Returns:
        Fully populated and validated PhytochemicalRecord, or None if SMILES is invalid.
    """
    # 1. Infer operating pH
    if estimated_ph is not None:
        ph = float(estimated_ph)
    elif solvent_class.lower().strip() == "vinegar_acidic":
        ph = 3.0
    elif thermal_regime.lower().strip() == "charred_ash":
        ph = 9.5
    else:
        ph = 7.0

    desc = calculate_descriptors(smiles, ph=ph)
    if desc is None:
        return None

    # 2. Infer operating temperature and duration
    if temperature_celsius is not None:
        temp_c = float(temperature_celsius)
    elif thermal_regime.lower().strip() == "boiling_prolonged":
        temp_c = 100.0
    elif thermal_regime.lower().strip() == "warm_steep":
        temp_c = 80.0
    elif thermal_regime.lower().strip() == "charred_ash":
        temp_c = 250.0
    else:
        temp_c = 25.0

    if duration_minutes is not None:
        dur_min = float(duration_minutes)
    elif thermal_regime.lower().strip() == "boiling_prolonged":
        dur_min = 30.0
    elif thermal_regime.lower().strip() == "warm_steep":
        dur_min = 60.0
    elif solvent_class.lower().strip() in ("lipid_oil", "hydroalcoholic"):
        dur_min = 1440.0
    else:
        dur_min = 15.0

    organs = list(documented_organs) if documented_organs is not None else []
    organ_factor = compute_organ_factor(prepared_part, organs)

    mol = Chem.MolFromSmiles(desc["smiles"])
    calc_ikey = None
    if mol is not None:
        try:
            calc_ikey = Chem.MolToInchiKey(mol)
        except Exception:
            pass
        surv_fraction, half_life_100c, is_thermolabile = compute_arrhenius_survival(
            mol=mol, temp_c=temp_c, duration_min=dur_min, mw=desc["mw"], tpsa=desc["tpsa"]
        )
    else:
        surv_fraction, half_life_100c, is_thermolabile = 1.0, None, desc["is_thermolabile"]

    final_inchi_key = calc_ikey or desc.get("inchi_key") or ""

    score = calculate_extraction_plausibility(
        descriptors=desc,
        solvent_class=solvent_class,
        thermal_regime=thermal_regime,
        estimated_ph=ph,
        duration_minutes=dur_min,
        temperature_celsius=temp_c,
        prepared_part=prepared_part,
        documented_organs=organs,
    )
    verdict = map_plausibility_to_verdict(score)

    targets = list(known_targets) if known_targets is not None else []

    return PhytochemicalRecord(
        compound_name=compound_name,
        smiles=desc["smiles"],
        inchi_key=final_inchi_key,
        molecular_formula=desc["molecular_formula"],
        mw=desc["mw"],
        clogp=desc["clogp"],
        tpsa=desc["tpsa"],
        h_donors=desc["h_donors"],
        h_acceptors=desc["h_acceptors"],
        rotatable_bonds=desc["rotatable_bonds"],
        is_thermolabile=is_thermolabile,
        is_volatile=desc["is_volatile"],
        toxicity_alerts=desc["toxicity_alerts"],
        extraction_plausibility_score=score,
        extraction_verdict=verdict,
        provenance_doi=provenance_doi,
        documented_organs=organs,
        organ_match_factor=organ_factor,
        acid_base_class=desc.get("acid_base_class", "neutral"),
        estimated_pka=desc.get("estimated_pka"),
        logd_at_recipe_ph=desc.get("logd_at_recipe_ph", desc["clogp"]),
        thermal_half_life_min_at_100c=half_life_100c,
        thermal_survival_fraction=surv_fraction,
        known_targets=targets,
        mechanistic_alignment_score=mechanistic_alignment_score,
    )
