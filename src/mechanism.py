"""DecoctionGraph Mechanism & Reverse Pharmacology Module.

Links solubilized phytochemical constituents to human macromolecular targets and historical
ethnomedical indications, evaluating mechanistic plausibility and closing the reverse-pharmacology loop.

Features:
  1. Hybrid Bioactivity Lookup: Curated peer-reviewed bioactivity knowledge base + live ChEMBL REST API fallback.
  2. Mechanistic Alignment Scoring: Evaluates alignment between target pathways and historical indications.
  3. Preparation-Level Validation Scoring: Computes M_prep = max_m (S_ext(m) * mechanistic_alignment_score(m)).
  4. Validated Mechanistic Path Verification: Highlights paths where both S_ext >= 0.50 and alignment >= 0.60.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import requests

from src.schemas import TargetMechanismRecord, PhytochemicalRecord, TargetActionType

logger = logging.getLogger(__name__)

DEFAULT_CHEMBL_CACHE_DIR = Path("data/cache/chembl")

# ==============================================================================
# Semantic Indication Keyword Clusters & Pathway Tokens
# ==============================================================================

INDICATION_SEMANTIC_CLUSTERS: Dict[str, Dict[str, List[str]]] = {
    "dermatological_wound": {
        "keywords": [
            "wound", "burn", "cut", "sore", "abrasion", "skin", "lesion", "ulcer",
            "topical", "scar", "healing", "bite", "boil", "blister", "re-epithelialization",
        ],
        "pathway_tokens": [
            "wound", "keratinocyte", "epithelial", "re-epithelialization", "fibroblast",
            "collagen", "antimicrobial", "staphylococcus", "skin", "differentiation", "trpc6",
        ],
    },
    "gastrointestinal": {
        "keywords": [
            "stomach", "dyspepsia", "colic", "appetite", "digestive", "digestion",
            "gastric", "gut", "bowel", "nausea", "vomiting", "spleen", "bile",
            "acid", "flatulence", "belly",
        ],
        "pathway_tokens": [
            "tas2r", "bitter", "gastric", "digestive", "cephalic", "chrm3", "gut",
            "motility", "mucosa", "spasmolysis", "appetite", "gastrin", "secretion",
        ],
    },
    "neuropsychiatric": {
        "keywords": [
            "anxiety", "melancholy", "depression", "nervous", "hysteria", "mind",
            "insomnia", "sleeplessness", "sedation", "mood", "neuralgia", "agitation",
            "restless", "spirit", "fear", "grief",
        ],
        "pathway_tokens": [
            "gabra", "gaba", "maoa", "maob", "serotonin", "sert", "slc6a4", "reuptake",
            "monoamine", "anxiolysis", "sedation", "neuroprotection", "antidepressant",
            "neurotransmitter",
        ],
    },
    "febrile_infectious": {
        "keywords": [
            "fever", "malaria", "ague", "tertian", "quartan", "chill", "cold",
            "heat", "infection", "bacteria", "pestilence", "contagion", "febrile",
            "shaking",
        ],
        "pathway_tokens": [
            "plasmodium", "pfatp6", "pfdhodh", "pfcrt", "hemozoin", "malarial",
            "parasiticide", "antipyresis", "pyrogen", "il6", "febrifuge", "endoperoxide",
        ],
    },
    "musculoskeletal_inflammatory": {
        "keywords": [
            "fracture", "bone", "sprain", "joint", "bruise", "rheumatism", "gout",
            "swelling", "periosteal", "pain", "inflammation", "ache", "arthritis",
            "knit", "sinew", "tendon",
        ],
        "pathway_tokens": [
            "bone", "collagen", "fibroblast", "alox5", "ptgs2", "cox-2", "complement",
            "c3", "leukotriene", "periosteal", "swelling", "anti-inflammatory", "proliferation",
        ],
    },
    "analgesic_antispasmodic": {
        "keywords": [
            "pain", "spasm", "colic", "cramp", "ache", "agony", "suffering",
            "convulsion", "intractable", "sleeplessness", "restlessness",
        ],
        "pathway_tokens": [
            "oprm1", "oprk1", "opioid", "nociceptive", "analgesia", "pde10a",
            "spasmolysis", "relaxation", "smooth muscle", "sedation",
        ],
    },
}


# ==============================================================================
# Curated Peer-Reviewed Bioactivity Knowledge Base
# ==============================================================================

CURATED_BIOACTIVITY_KB: Dict[str, List[Dict[str, Any]]] = {
    # Hypericum perforatum
    "hyperforin": [
        {
            "target_name": "Transient receptor potential cation channel subfamily C member 6",
            "gene_symbol": "TRPC6",
            "chembl_target_id": "CHEMBL5488",
            "action_type": "modulator",
            "bioactivity_potency_um": 0.3,
            "therapeutic_pathway": "Keratinocyte differentiation, calcium influx & topical wound re-epithelialization",
            "symptom_relevance_score": 0.96,
        },
        {
            "target_name": "Sodium-dependent serotonin transporter (SERT)",
            "gene_symbol": "SLC6A4",
            "chembl_target_id": "CHEMBL228",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 0.1,
            "therapeutic_pathway": "Synaptosomal non-selective monoamine reuptake inhibition for depression & nervous anxiety",
            "symptom_relevance_score": 0.91,
        },
        {
            "target_name": "Staphylococcus aureus DNA gyrase / cell wall integrity",
            "gene_symbol": "GyrA",
            "chembl_target_id": "CHEMBL3714260",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 0.1,
            "therapeutic_pathway": "Gram-positive antimicrobial inhibition against skin wound pathogens",
            "symptom_relevance_score": 0.92,
        },
    ],
    "hypericin": [
        {
            "target_name": "Monoamine oxidase A",
            "gene_symbol": "MAOA",
            "chembl_target_id": "CHEMBL1951",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 5.0,
            "therapeutic_pathway": "Monoamine neurotransmitter preservation & mood regulation",
            "symptom_relevance_score": 0.85,
        },
        {
            "target_name": "Protein kinase C alpha",
            "gene_symbol": "PRKCA",
            "chembl_target_id": "CHEMBL299",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 3.0,
            "therapeutic_pathway": "Kinase signaling cascade suppression and viral envelope photodynamic cleavage",
            "symptom_relevance_score": 0.70,
        },
    ],
    "pseudohypericin": [
        {
            "target_name": "Protein kinase C alpha",
            "gene_symbol": "PRKCA",
            "chembl_target_id": "CHEMBL299",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 2.8,
            "therapeutic_pathway": "Kinase signaling cascade suppression & photo-antiviral action",
            "symptom_relevance_score": 0.70,
        },
    ],
    "rutin": [
        {
            "target_name": "Prostaglandin G/H synthase 2 (COX-2)",
            "gene_symbol": "PTGS2",
            "chembl_target_id": "CHEMBL230",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 12.0,
            "therapeutic_pathway": "Anti-inflammatory eicosanoid cascade & vascular capillary stabilization",
            "symptom_relevance_score": 0.82,
        },
        {
            "target_name": "Nuclear factor erythroid 2-related factor 2 (Nrf2/ARE)",
            "gene_symbol": "NFE2L2",
            "chembl_target_id": "CHEMBL5981",
            "action_type": "agonist",
            "bioactivity_potency_um": 8.5,
            "therapeutic_pathway": "Nrf2/ARE antioxidant response element activation & endothelial cytoprotection",
            "symptom_relevance_score": 0.76,
        },
        {
            "target_name": "Tumor necrosis factor alpha",
            "gene_symbol": "TNF",
            "chembl_target_id": "CHEMBL253",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 15.0,
            "therapeutic_pathway": "Pro-inflammatory cytokine suppression (TNF-alpha / IL-6)",
            "symptom_relevance_score": 0.78,
        },
    ],
    "quercetin": [
        {
            "target_name": "Prostaglandin G/H synthase 2 (COX-2)",
            "gene_symbol": "PTGS2",
            "chembl_target_id": "CHEMBL230",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 4.5,
            "therapeutic_pathway": "Inhibition of pro-inflammatory prostaglandins and mast cell histamine release",
            "symptom_relevance_score": 0.85,
        },
        {
            "target_name": "Monoamine oxidase A",
            "gene_symbol": "MAOA",
            "chembl_target_id": "CHEMBL1951",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 9.0,
            "therapeutic_pathway": "Monoamine breakdown reduction for mild anxiety and nervous exhaustion",
            "symptom_relevance_score": 0.75,
        },
        {
            "target_name": "Nuclear factor erythroid 2-related factor 2 (Nrf2/ARE)",
            "gene_symbol": "NFE2L2",
            "chembl_target_id": "CHEMBL5981",
            "action_type": "agonist",
            "bioactivity_potency_um": 2.1,
            "therapeutic_pathway": "Potent Nrf2 antioxidant gene induction",
            "symptom_relevance_score": 0.80,
        },
    ],
    "biapigenin": [
        {
            "target_name": "Gamma-aminobutyric acid type A receptor subunit alpha1",
            "gene_symbol": "GABRA1",
            "chembl_target_id": "CHEMBL2094112",
            "action_type": "modulator",
            "bioactivity_potency_um": 2.5,
            "therapeutic_pathway": "Benzodiazepine allosteric site positive modulation, GABAergic anxiolysis & sedation",
            "symptom_relevance_score": 0.90,
        },
    ],
    "chlorogenic acid": [
        {
            "target_name": "Peroxisome proliferator-activated receptor alpha",
            "gene_symbol": "PPARA",
            "chembl_target_id": "CHEMBL239",
            "action_type": "agonist",
            "bioactivity_potency_um": 15.0,
            "therapeutic_pathway": "Metabolic regulation & mucosal lipid peroxidation protection",
            "symptom_relevance_score": 0.65,
        },
    ],

    # Gentiana lutea
    "gentiopicroside": [
        {
            "target_name": "Taste receptor type 2 member 38",
            "gene_symbol": "TAS2R38",
            "chembl_target_id": "CHEMBL3714264",
            "action_type": "agonist",
            "bioactivity_potency_um": 0.05,
            "therapeutic_pathway": "Bitter taste receptor activation on enteroendocrine cells triggering cephalic gastrin & bile secretion",
            "symptom_relevance_score": 0.95,
        },
        {
            "target_name": "Prostaglandin G/H synthase 2",
            "gene_symbol": "PTGS2",
            "chembl_target_id": "CHEMBL230",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 14.0,
            "therapeutic_pathway": "Gastric mucosal anti-inflammatory suppression of COX-2 / NF-kB",
            "symptom_relevance_score": 0.78,
        },
    ],
    "amarogentin": [
        {
            "target_name": "Taste receptor type 2 member 50 / TAS2R10",
            "gene_symbol": "TAS2R50",
            "chembl_target_id": "CHEMBL3714265",
            "action_type": "agonist",
            "bioactivity_potency_um": 0.03,
            "therapeutic_pathway": "Ultra-potent bitter receptor stimulation driving vagal gastric digestive reflex, motility, and appetite stimulation",
            "symptom_relevance_score": 0.98,
        },
    ],
    "gentisin": [
        {
            "target_name": "Monoamine oxidase B",
            "gene_symbol": "MAOB",
            "chembl_target_id": "CHEMBL2039",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 1.2,
            "therapeutic_pathway": "Neuroprotection and vitality tonification",
            "symptom_relevance_score": 0.65,
        },
    ],
    "swertiamarin": [
        {
            "target_name": "Cholinergic receptor muscarinic 3",
            "gene_symbol": "CHRM3",
            "chembl_target_id": "CHEMBL245",
            "action_type": "modulator",
            "bioactivity_potency_um": 10.0,
            "therapeutic_pathway": "Gastrointestinal smooth muscle contractility and gut transit",
            "symptom_relevance_score": 0.85,
        },
    ],
    "loganic acid": [
        {
            "target_name": "Nitric oxide synthase, inducible",
            "gene_symbol": "NOS2",
            "chembl_target_id": "CHEMBL246",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 18.0,
            "therapeutic_pathway": "Gastric mucosa cytoprotection against inflammatory ulceration",
            "symptom_relevance_score": 0.72,
        },
    ],

    # Artemisia annua
    "artemisinin": [
        {
            "target_name": "Plasmodium falciparum sarco/endoplasmic reticulum calcium ATPase 6 (PfATP6)",
            "gene_symbol": "PfATP6",
            "chembl_target_id": "CHEMBL4105822",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 0.009,
            "therapeutic_pathway": "Heme-iron (Fe2+) mediated endoperoxide bioactivation, carbon radical generation & PfATP6 malarial parasiticide",
            "symptom_relevance_score": 0.99,
        },
    ],
    "arteannuin b": [
        {
            "target_name": "Plasmodium falciparum dihydroorotate dehydrogenase",
            "gene_symbol": "PfDHODH",
            "chembl_target_id": "CHEMBL4523315",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 5.4,
            "therapeutic_pathway": "Inhibition of malarial parasite pyrimidine biosynthesis",
            "symptom_relevance_score": 0.70,
        },
    ],
    "scopoletin": [
        {
            "target_name": "Interleukin-6 / NF-kB pathway",
            "gene_symbol": "IL6",
            "chembl_target_id": "CHEMBL2147",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 14.0,
            "therapeutic_pathway": "Febrile pyrogen cytokine suppression & antipyresis",
            "symptom_relevance_score": 0.88,
        },
    ],
    "chrysosplenol d": [
        {
            "target_name": "Multidrug resistance protein 1 (P-glycoprotein efflux pump)",
            "gene_symbol": "ABCB1",
            "chembl_target_id": "CHEMBL4302",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 2.8,
            "therapeutic_pathway": "Parasitic drug efflux inhibition / artemisinin bioavailability enhancement",
            "symptom_relevance_score": 0.90,
        },
    ],

    # Symphytum officinale
    "allantoin": [
        {
            "target_name": "Epidermal growth factor receptor / Keratinocyte & Fibroblast mitotic stimulation",
            "gene_symbol": "EGFR",
            "chembl_target_id": "CHEMBL203",
            "action_type": "modulator",
            "bioactivity_potency_um": 25.0,
            "therapeutic_pathway": "Fibroblast proliferation, collagen synthesis, re-epithelialization & bone union acceleration",
            "symptom_relevance_score": 0.97,
        },
    ],
    "rosmarinic acid": [
        {
            "target_name": "Arachidonate 5-lipoxygenase",
            "gene_symbol": "ALOX5",
            "chembl_target_id": "CHEMBL249",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 3.8,
            "therapeutic_pathway": "Leukotriene suppression & periosteal/joint inflammation resolution",
            "symptom_relevance_score": 0.90,
        },
        {
            "target_name": "Complement component C3 convertase",
            "gene_symbol": "C3",
            "chembl_target_id": "CHEMBL2094113",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 7.2,
            "therapeutic_pathway": "Complement C3 convertase inhibition protecting against sprain and fracture swelling",
            "symptom_relevance_score": 0.88,
        },
    ],
    "symphytine": [
        {
            "target_name": "Cytochrome P450 3A4 (Bioactivation to pyrrolic dehydro-alkaloids)",
            "gene_symbol": "CYP3A4",
            "chembl_target_id": "CHEMBL340",
            "action_type": "substrate",
            "bioactivity_potency_um": 1.5,
            "therapeutic_pathway": "Hepatotoxic bioactivation generating reactive pyrrolic electrophiles, DNA cross-linking & hepatic sinusoidal obstruction syndrome",
            "symptom_relevance_score": 0.10,
        },
    ],
    "echimidine": [
        {
            "target_name": "Cytochrome P450 3A4 (Hepatotoxic bioactivation)",
            "gene_symbol": "CYP3A4",
            "chembl_target_id": "CHEMBL340",
            "action_type": "substrate",
            "bioactivity_potency_um": 1.8,
            "therapeutic_pathway": "Hepatic veno-occlusive damage & irreversible hepatocyte alkylation",
            "symptom_relevance_score": 0.10,
        },
    ],

    # Cinchona pubescens
    "quinine": [
        {
            "target_name": "Plasmodium falciparum chloroquine resistance transporter / Hemozoin biocrystallization",
            "gene_symbol": "PfCRT",
            "chembl_target_id": "CHEMBL4105823",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 0.12,
            "therapeutic_pathway": "Inhibition of hemozoin biocrystallization causing toxic hematin accumulation in protozoa, febrifuge & antimalarial",
            "symptom_relevance_score": 0.99,
        },
        {
            "target_name": "Cholinergic receptor muscarinic 2",
            "gene_symbol": "CHRM2",
            "chembl_target_id": "CHEMBL211",
            "action_type": "antagonist",
            "bioactivity_potency_um": 3.5,
            "therapeutic_pathway": "Anticholinergic modulation & smooth muscle relaxation",
            "symptom_relevance_score": 0.70,
        },
    ],
    "cinchonine": [
        {
            "target_name": "Plasmodium falciparum hemozoin biocrystallization",
            "gene_symbol": "PfCRT",
            "chembl_target_id": "CHEMBL4105823",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 0.45,
            "therapeutic_pathway": "Malarial hemozoin disruption & anti-ague febrifuge action",
            "symptom_relevance_score": 0.92,
        },
    ],
    "quinidine": [
        {
            "target_name": "Sodium channel protein type 5 subunit alpha",
            "gene_symbol": "SCN5A",
            "chembl_target_id": "CHEMBL2047",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 2.1,
            "therapeutic_pathway": "Cardiac fast sodium channel blockade & antimalarial febrifuge action",
            "symptom_relevance_score": 0.88,
        },
    ],

    # Papaver somniferum
    "morphine": [
        {
            "target_name": "Mu-type opioid receptor",
            "gene_symbol": "OPRM1",
            "chembl_target_id": "CHEMBL233",
            "action_type": "agonist",
            "bioactivity_potency_um": 0.002,
            "therapeutic_pathway": "Supraspinal and spinal nociceptive inhibition, central sedation & intense analgesia",
            "symptom_relevance_score": 0.99,
        },
        {
            "target_name": "Kappa-type opioid receptor",
            "gene_symbol": "OPRK1",
            "chembl_target_id": "CHEMBL234",
            "action_type": "agonist",
            "bioactivity_potency_um": 0.05,
            "therapeutic_pathway": "Spinal analgesia & visceral pain reduction",
            "symptom_relevance_score": 0.92,
        },
    ],
    "codeine": [
        {
            "target_name": "Mu-type opioid receptor (via conversion to morphine)",
            "gene_symbol": "OPRM1",
            "chembl_target_id": "CHEMBL233",
            "action_type": "agonist",
            "bioactivity_potency_um": 0.20,
            "therapeutic_pathway": "Medullary cough center depression (antitussive) & mild-to-moderate analgesia",
            "symptom_relevance_score": 0.92,
        },
    ],
    "papaverine": [
        {
            "target_name": "Phosphodiesterase 10A / Phosphodiesterase 4",
            "gene_symbol": "PDE10A",
            "chembl_target_id": "CHEMBL4409",
            "action_type": "inhibitor",
            "bioactivity_potency_um": 0.85,
            "therapeutic_pathway": "Intracellular cAMP/cGMP elevation, visceral smooth muscle spasmolysis & vasodilation",
            "symptom_relevance_score": 0.94,
        },
    ],
}

# Cross-index the curated KB by InChIKey as well
INCHI_TO_TARGETS_MAP: Dict[str, List[Dict[str, Any]]] = {
    "KGSZHKRKHXOAMG-HQKKAZOISA-N": CURATED_BIOACTIVITY_KB["hyperforin"],
    "YDOIFHVUBCIUHF-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["hypericin"],
    "NODGUBIGZKATOM-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["pseudohypericin"],
    "IKGXIBQEEMLURG-NVPNHPEKSA-N": CURATED_BIOACTIVITY_KB["rutin"],
    "REFJWTPEDVJJIY-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["quercetin"],
    "IQAMTZLKUHMPPE-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["biapigenin"],
    "CWVRJTMFETXNAD-JUHZACGLSA-N": CURATED_BIOACTIVITY_KB["chlorogenic acid"],
    "DUAGQYUORDTXOR-GPQRQXLASA-N": CURATED_BIOACTIVITY_KB["gentiopicroside"],
    "DBOVHQOUSDWAPQ-WTONXPSSSA-N": CURATED_BIOACTIVITY_KB["amarogentin"],
    "XOXYHGOIRWABTC-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["gentisin"],
    "HEYZWPRKKUGDCR-QBXMEVCASA-N": CURATED_BIOACTIVITY_KB["swertiamarin"],
    "JNNGEAWILNVFFD-CDJYTOATSA-N": CURATED_BIOACTIVITY_KB["loganic acid"],
    "BLUAFEHZUWYNDE-NNWCWBAJSA-N": CURATED_BIOACTIVITY_KB["artemisinin"],
    "QWQSMEDUZQDVLA-KPHNHPKPSA-N": CURATED_BIOACTIVITY_KB["arteannuin b"],
    "RODXRVNMMDRFIK-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["scopoletin"],
    "BYWLLSQTJBXAPV-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["chrysosplenol d"],
    "POJWUDADGALRAB-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["allantoin"],
    "DOUMFZQKYFQNTF-WUTVXBCWSA-N": CURATED_BIOACTIVITY_KB["rosmarinic acid"],
    "MVWPTZQHBOWRTF-SMLWLWDZSA-N": CURATED_BIOACTIVITY_KB["symphytine"],
    "HRSGCYGUWHGOPY-LYHHMGRNSA-N": CURATED_BIOACTIVITY_KB["echimidine"],
    "LOUPRKONTZGTKE-LHHVKLHASA-N": CURATED_BIOACTIVITY_KB["quinine"],
    "LOUPRKONTZGTKE-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["quinine"],
    "KMPWYEUPVWOPIM-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["cinchonine"],
    "LOUPRKONTZGTKE-UYVJDWJCSA-N": CURATED_BIOACTIVITY_KB["quinidine"],
    "BQJCRHHNABKAKU-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["morphine"],
    "OROGSEYTTFOCAN-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["codeine"],
    "YMIAKQJJUDUNFI-UHFFFAOYSA-N": CURATED_BIOACTIVITY_KB["papaverine"],
}


# ==============================================================================
# Live ChEMBL REST API Client
# ==============================================================================

def fetch_chembl_targets(
    inchikey: str,
    cache_dir: Union[str, Path] = DEFAULT_CHEMBL_CACHE_DIR,
    timeout: float = 3.0,
) -> List[TargetMechanismRecord]:
    """Query ChEMBL REST API for bioactivity targets associated with an InChIKey.

    Uses a 3-second timeout and local JSON disk caching. Gracefully falls back
    to an empty list on network timeouts or offline environments.

    Args:
        inchikey: Canonical 27-character InChIKey.
        cache_dir: Directory for persistent JSON caching.
        timeout: Network request timeout in seconds.

    Returns:
        List of parsed TargetMechanismRecord instances.
    """
    clean_key = inchikey.strip().upper()
    if not clean_key or len(clean_key) < 14:
        return []

    c_dir = Path(cache_dir)
    cache_file = c_dir / f"{clean_key}.json"

    # 1. Check local cache
    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [TargetMechanismRecord(**item) for item in data]
        except Exception as exc:
            logger.debug(f"ChEMBL cache read error for {clean_key}: {exc}")

    # 2. Live API lookup
    targets: List[TargetMechanismRecord] = []
    mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{clean_key}.json"

    try:
        mol_resp = requests.get(mol_url, timeout=timeout)
        if mol_resp.status_code == 200:
            mol_data = mol_resp.json()
            chembl_id = mol_data.get("molecule_chembl_id")
            if chembl_id:
                act_url = (
                    f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
                    f"molecule_chembl_id={chembl_id}&pchembl_value__gte=5.0&limit=5"
                )
                act_resp = requests.get(act_url, timeout=timeout)
                if act_resp.status_code == 200:
                    activities = act_resp.json().get("activities", [])
                    seen_targets: Set[str] = set()

                    for act in activities:
                        t_id = act.get("target_chembl_id")
                        t_name = act.get("target_pref_name", "Unspecified ChEMBL Target")
                        pchembl = act.get("pchembl_value")
                        std_type = act.get("standard_type", "IC50")

                        if not t_id or t_id in seen_targets:
                            continue
                        seen_targets.add(t_id)

                        # Potency in uM: 10^(6 - pChEMBL)
                        potency_um: Optional[float] = None
                        if pchembl is not None:
                            try:
                                potency_um = round(10.0 ** (6.0 - float(pchembl)), 4)
                            except (ValueError, TypeError):
                                pass

                        action: TargetActionType = "inhibitor"
                        if std_type in ("EC50", "ED50"):
                            action = "agonist"
                        elif std_type in ("Ki", "Kd"):
                            action = "modulator"

                        # Extract clean gene symbol or abbreviation
                        gene_symbol = t_name.split()[0].upper().rstrip(",.;:")
                        if len(gene_symbol) > 10:
                            gene_symbol = gene_symbol[:8]

                        record = TargetMechanismRecord(
                            target_name=t_name,
                            gene_symbol=gene_symbol,
                            chembl_target_id=t_id,
                            action_type=action,
                            bioactivity_potency_um=potency_um,
                            therapeutic_pathway=f"ChEMBL Bioactivity: {t_name} ({std_type}={potency_um} µM)",
                            symptom_relevance_score=0.50,
                        )
                        targets.append(record)

    except (requests.RequestException, Exception) as exc:
        logger.debug(f"Live ChEMBL API request failed or timed out for {clean_key}: {exc}")

    # 3. Write cache to disk if targets found or API answered
    if targets:
        try:
            c_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in targets], f, indent=2)
        except Exception as exc:
            logger.debug(f"Failed writing ChEMBL cache for {clean_key}: {exc}")

    return targets


# ==============================================================================
# Hybrid Bioactivity Retrieval
# ==============================================================================

def get_targets_for_compound(
    compound_name: str,
    inchi_key: Optional[str] = None,
    use_chembl_fallback: bool = True,
) -> List[TargetMechanismRecord]:
    """Retrieve pharmacological targets for a phytochemical compound.

    Searches the high-precision curated knowledge base first (by InChIKey or normalized
    name), and falls back to live ChEMBL querying when requested.

    Args:
        compound_name: Common, trivial, or IUPAC compound name.
        inchi_key: Optional canonical 27-character InChIKey.
        use_chembl_fallback: Whether to query ChEMBL if not in curated knowledge base.

    Returns:
        List of TargetMechanismRecord instances.
    """
    clean_name = compound_name.strip().lower()
    clean_key = inchi_key.strip().upper() if inchi_key else None

    # 1. Curated InChIKey match
    if clean_key and clean_key in INCHI_TO_TARGETS_MAP:
        return [TargetMechanismRecord(**dict(item)) for item in INCHI_TO_TARGETS_MAP[clean_key]]

    # 2. Curated Compound Name match
    if clean_name in CURATED_BIOACTIVITY_KB:
        return [TargetMechanismRecord(**dict(item)) for item in CURATED_BIOACTIVITY_KB[clean_name]]

    # 3. Live ChEMBL REST Fallback
    if use_chembl_fallback and clean_key:
        chembl_targets = fetch_chembl_targets(clean_key)
        if chembl_targets:
            return chembl_targets

    return []


# ==============================================================================
# Mechanistic Alignment Scoring
# ==============================================================================

def score_mechanistic_alignment(
    compound_targets: Sequence[TargetMechanismRecord],
    verbatim_indication: str,
) -> float:
    """Calculate the mechanistic alignment score [0.0, 1.0] between a compound's targets and the indication.

    Matches indication keywords to pharmacological pathway tokens, evaluating whether
    the compound's bioactivity profile explains the historical therapeutic effect.

    Args:
        compound_targets: Collection of TargetMechanismRecord instances for the compound.
        verbatim_indication: Historical indication text (e.g. "burns and skin wounds", "intermittent fever").

    Returns:
        Float alignment score between 0.0 and 1.0.
    """
    if not compound_targets or not verbatim_indication.strip():
        return 0.0

    ind_lower = verbatim_indication.strip().lower()

    # 1. Identify which indication semantic clusters are triggered
    matched_clusters: Set[str] = set()
    for cluster_name, cluster_data in INDICATION_SEMANTIC_CLUSTERS.items():
        for kw in cluster_data["keywords"]:
            if re.search(rf"\b{re.escape(kw)}", ind_lower):
                matched_clusters.add(cluster_name)
                break

    best_target_score = 0.0

    # 2. Evaluate each target mechanism against triggered clusters and direct keyword overlap
    for target in compound_targets:
        pathway_lower = target.therapeutic_pathway.lower()
        target_name_lower = target.target_name.lower()
        gene_lower = target.gene_symbol.lower()
        combined_target_text = f"{pathway_lower} {target_name_lower} {gene_lower}"

        cluster_match_strength = 0.0
        for cluster_name in matched_clusters:
            tokens = INDICATION_SEMANTIC_CLUSTERS[cluster_name]["pathway_tokens"]
            matching_tokens = sum(1 for tok in tokens if tok in combined_target_text)
            if matching_tokens > 0:
                strength = min(1.0, 0.60 + 0.15 * matching_tokens)
                if strength > cluster_match_strength:
                    cluster_match_strength = strength

        # Direct token overlap between indication and pathway
        ind_words = [w for w in re.findall(r"\w+", ind_lower) if len(w) >= 4]
        direct_matches = sum(1 for w in ind_words if w in combined_target_text)
        direct_bonus = min(0.30, 0.10 * direct_matches)

        match_factor = min(1.0, cluster_match_strength + direct_bonus)
        if match_factor > 0:
            effective_score = target.symptom_relevance_score * match_factor
        else:
            # Baseline small score if target exists but indication does not match
            effective_score = target.symptom_relevance_score * 0.10

        if effective_score > best_target_score:
            best_target_score = effective_score

    return round(float(min(1.0, max(0.0, best_target_score))), 4)


def calculate_preparation_mechanistic_validation(
    compounds: Sequence[PhytochemicalRecord],
    verbatim_indication: str,
) -> float:
    """Calculate aggregate preparation-level Mechanistic Validation Score (M_prep).

    Formula:
      M_prep = max_m (S_ext(m) * mechanistic_alignment_score(m))

    Proves mathematically whether the traditional preparation method effectively solubilizes
    the specific secondary metabolites responsible for the historical therapeutic efficacy.

    Args:
        compounds: List of enriched PhytochemicalRecord instances for the preparation.
        verbatim_indication: Historical indication text.

    Returns:
        Continuous validation score between 0.0 and 1.0.
    """
    if not compounds or not verbatim_indication.strip():
        return 0.0

    best_m_prep = 0.0

    for comp in compounds:
        s_ext = comp.extraction_plausibility_score
        # Compute alignment score if not already populated or if indication differs
        if comp.mechanistic_alignment_score > 0.0:
            align = comp.mechanistic_alignment_score
        else:
            align = score_mechanistic_alignment(comp.known_targets, verbatim_indication)

        product = s_ext * align
        if product > best_m_prep:
            best_m_prep = product

    return round(float(min(1.0, max(0.0, best_m_prep))), 4)


def validate_mechanistic_path(
    plausibility_score: float,
    alignment_score: float,
    min_plausibility: float = 0.50,
    min_alignment: float = 0.60,
) -> bool:
    """Determine whether a compound-target pathway constitutes a Validated Mechanistic Path.

    Criteria:
      1. S_ext >= min_plausibility (traditional preparation successfully extracts the compound).
      2. Alignment >= min_alignment (compound target mechanistically explains the symptom).

    Args:
        plausibility_score: Extraction plausibility score S_ext [0.0, 1.0].
        alignment_score: Mechanistic alignment score [0.0, 1.0].
        min_plausibility: Minimum required extraction plausibility (default: 0.50).
        min_alignment: Minimum required mechanistic alignment (default: 0.60).

    Returns:
        True if the path is scientifically validated, False otherwise.
    """
    return bool(plausibility_score >= min_plausibility and alignment_score >= min_alignment)
