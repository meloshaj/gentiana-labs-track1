"""DecoctionGraph LOTUS Knowledge Retriever Module.

Maps standardized botanical taxa to known phytochemical constituents using the LOTUS
(Natural Products Online) dataset. Supports dual operating modes:
  1. Offline Curated Benchmark Mode: Fast offline lookup of verified secondary metabolites
     for key benchmark taxa (Hypericum perforatum, Gentiana lutea, Artemisia annua,
     Symphytum officinale, Cinchona pubescens, Papaver somniferum).
  2. Parquet/CSV Loading Mode: Streaming/batch querying of local LOTUS extracts (Zenodo dump).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from rdkit import Chem

from src.schemas import (
    PhytochemicalRecord,
    SolventClass,
    ThermalRegime,
    TargetMechanismRecord,
)
from src.cheminformatics import create_phytochemical_record

logger = logging.getLogger(__name__)

DEFAULT_CURATED_JSON_PATH = Path("data/reference/lotus_curated_sample.json")
LOTUS_BASELINE_DOI = "10.7554/eLife.70780 (Rutz et al., 2022, eLife - LOTUS Initiative)"

# ==============================================================================
# Built-In Curated Offline Benchmark Dataset
# ==============================================================================

# Embedded authoritative benchmark dataset for offline operation and deterministic tests
EMBEDDED_LOTUS_BENCHMARK: Dict[str, List[Dict[str, Any]]] = {
    "Hypericum perforatum": [
        {
            "compound_name": "Hypericin",
            "smiles": "CC1=CC(=O)C2=C(C3=C(C=C(C4=C3C5=C2C1=C6C(=CC(=O)C7=C(C8=C(C=C(C4=C8C5=C67)O)O)O)C)O)O)O",
            "inchi_key": "YDOIFHVUBCIUHF-UHFFFAOYSA-N",
            "doi": "EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Monoamine oxidase A",
                    "gene_symbol": "MAOA",
                    "chembl_target_id": "CHEMBL1951",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 5.0,
                    "therapeutic_pathway": "Monoamine neurotransmitter homeostasis & mood regulation [EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.85,
                }
            ],
            "mechanistic_alignment_score": 0.82,
        },
        {
            "compound_name": "Hyperforin",
            "smiles": "CC(C)C(=O)C12C(=O)C(=C(C(C1=O)(CC(C2(C)CCC=C(C)C)CC=C(C)C)CC=C(C)C)O)CC=C(C)C",
            "inchi_key": "KGSZHKRKHXOAMG-HQKKAZOISA-N",
            "doi": "10.1073/pnas.0706216104 (Leuner et al., 2007, PNAS) & EMA/HMPC/101304/2008",
            "documented_organs": ["flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Transient receptor potential cation channel subfamily C member 6",
                    "gene_symbol": "TRPC6",
                    "chembl_target_id": "CHEMBL4523378 (UniProt:Q9Y210)",
                    "action_type": "modulator",
                    "bioactivity_potency_um": 0.3,
                    "therapeutic_pathway": "Keratinocyte differentiation, wound healing & non-selective monoamine reuptake inhibition [10.1073/pnas.0706216104 (Leuner et al., 2007, PNAS)]",
                    "symptom_relevance_score": 0.95,
                }
            ],
            "mechanistic_alignment_score": 0.92,
        },
        {
            "compound_name": "Rutin",
            "smiles": "CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C=C5)O)O)O)O)O)O)O)O",
            "inchi_key": "IKGXIBQEEMLURG-NVPNHPEKSA-N",
            "doi": "EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["leaf", "flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Prostaglandin G/H synthase 2",
                    "gene_symbol": "PTGS2",
                    "chembl_target_id": "CHEMBL230",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 12.0,
                    "therapeutic_pathway": "Anti-inflammatory eicosanoid cascade modulation [EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.75,
                }
            ],
            "mechanistic_alignment_score": 0.70,
        },
        {
            "compound_name": "Quercetin",
            "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
            "inchi_key": "REFJWTPEDVJJIY-UHFFFAOYSA-N",
            "doi": "EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["leaf", "flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Prostaglandin G/H synthase 2",
                    "gene_symbol": "PTGS2",
                    "chembl_target_id": "CHEMBL230",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 4.5,
                    "therapeutic_pathway": "Inhibition of pro-inflammatory prostaglandins [EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.80,
                }
            ],
            "mechanistic_alignment_score": 0.75,
        },
        {
            "compound_name": "Chlorogenic acid",
            "smiles": "C1C(C(C(CC1(C(=O)O)O)OC(=O)C=CC2=CC(=C(C=C2)O)O)O)O",
            "inchi_key": "CWVRJTMFETXNAD-JUHZACGLSA-N",
            "doi": "EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["leaf", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Peroxisome proliferator-activated receptor alpha",
                    "gene_symbol": "PPARA",
                    "chembl_target_id": "CHEMBL239",
                    "action_type": "agonist",
                    "bioactivity_potency_um": 15.0,
                    "therapeutic_pathway": "Antioxidant and metabolic regulation [EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.60,
                }
            ],
            "mechanistic_alignment_score": 0.60,
        },
        {
            "compound_name": "Pseudohypericin",
            "smiles": "CC1=CC(=O)C2=C(C3=C(C=C(C4=C3C5=C2C1=C6C(=CC(=O)C7=C(C8=C(C=C(C4=C8C5=C67)O)O)O)CO)O)O)O",
            "inchi_key": "NODGUBIGZKATOM-UHFFFAOYSA-N",
            "doi": "EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Protein kinase C",
                    "gene_symbol": "PRKCA",
                    "chembl_target_id": "CHEMBL299",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 3.0,
                    "therapeutic_pathway": "Signaling cascade modulation and antiviral activity [EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.70,
                }
            ],
            "mechanistic_alignment_score": 0.68,
        },
        {
            "compound_name": "Biapigenin",
            "smiles": "C1=CC(=CC=C1C2=CC(=O)C3=C(O2)C(=C(C=C3O)O)C4=C(OC5=CC(=CC(=C5C4=O)O)O)C6=CC=C(C=C6)O)O",
            "inchi_key": "IQAMTZLKUHMPPE-UHFFFAOYSA-N",
            "doi": "EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Benzodiazepine receptor (GABA-A)",
                    "gene_symbol": "GABRA1",
                    "chembl_target_id": "CHEMBL2094112",
                    "action_type": "modulator",
                    "bioactivity_potency_um": 2.5,
                    "therapeutic_pathway": "GABAergic anxiolysis & sedation [EMA/HMPC/101304/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.85,
                }
            ],
            "mechanistic_alignment_score": 0.80,
        },
    ],
    "Gentiana lutea": [
        {
            "compound_name": "Gentiopicroside",
            "smiles": "C=CC1C(OC=C2C1=CCOC2=O)OC3C(C(C(C(O3)CO)O)O)O",
            "inchi_key": "DUAGQYUORDTXOR-GPQRQXLASA-N",
            "doi": "10.1021/jf072611k (Brockhoff et al., 2007) & EMA/HMPC/578322/2008",
            "documented_organs": ["root", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Taste receptor type 2 member 38",
                    "gene_symbol": "TAS2R38",
                    "chembl_target_id": "CHEMBL2176774 (UniProt:P59533)",
                    "action_type": "agonist",
                    "bioactivity_potency_um": 0.05,
                    "therapeutic_pathway": "Cephalic phase gastrin secretion & digestive stimulation [10.1021/jf072611k (Brockhoff et al., 2007) & EMA/HMPC/578322/2008]",
                    "symptom_relevance_score": 0.95,
                }
            ],
            "mechanistic_alignment_score": 0.90,
        },
        {
            "compound_name": "Amarogentin",
            "smiles": "C=CC1C2CCOC(=O)C2=COC1OC3C(C(C(C(O3)CO)O)O)OC(=O)C4=C(C=C(C=C4O)O)C5=CC(=CC=C5)O",
            "inchi_key": "DBOVHQOUSDWAPQ-WTONXPSSSA-N",
            "doi": "10.1021/jf9014334 (Behrens et al., 2009) & EMA/HMPC/578322/2008",
            "documented_organs": ["root", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Taste receptor type 2 member 10",
                    "gene_symbol": "TAS2R10",
                    "chembl_target_id": "UniProt:Q9NYW0 (TAS2R10)",
                    "action_type": "agonist",
                    "bioactivity_potency_um": 0.03,
                    "therapeutic_pathway": "Intense bitter reflex stimulation & gastric motility [10.1021/jf9014334 (Behrens et al., 2009) & EMA/HMPC/578322/2008]",
                    "symptom_relevance_score": 0.98,
                }
            ],
            "mechanistic_alignment_score": 0.94,
        },
        {
            "compound_name": "Gentisin",
            "smiles": "COC1=CC(=C2C(=C1)OC3=C(C2=O)C=C(C=C3)O)O",
            "inchi_key": "XOXYHGOIRWABTC-UHFFFAOYSA-N",
            "doi": "EMA/HMPC/578322/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["root", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Monoamine oxidase B",
                    "gene_symbol": "MAOB",
                    "chembl_target_id": "CHEMBL2039",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 1.2,
                    "therapeutic_pathway": "Neuroprotection & tonifying vitality [EMA/HMPC/578322/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.65,
                }
            ],
            "mechanistic_alignment_score": 0.60,
        },
        {
            "compound_name": "Swertiamarin",
            "smiles": "C=CC1C(OC=C2C1(CCOC2=O)O)OC3C(C(C(C(O3)CO)O)O)O",
            "inchi_key": "HEYZWPRKKUGDCR-QBXMEVCASA-N",
            "doi": "EMA/HMPC/578322/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["root", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Cholinergic receptor muscarinic 3",
                    "gene_symbol": "CHRM3",
                    "chembl_target_id": "CHEMBL245",
                    "action_type": "modulator",
                    "bioactivity_potency_um": 10.0,
                    "therapeutic_pathway": "Gastrointestinal smooth muscle contractility [EMA/HMPC/578322/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.85,
                }
            ],
            "mechanistic_alignment_score": 0.80,
        },
        {
            "compound_name": "Loganic acid",
            "smiles": "CC1C(CC2C1C(OC=C2C(=O)O)OC3C(C(C(C(O3)CO)O)O)O)O",
            "inchi_key": "JNNGEAWILNVFFD-CDJYTOATSA-N",
            "doi": "EMA/HMPC/578322/2008 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["root", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Nitric oxide synthase, inducible",
                    "gene_symbol": "NOS2",
                    "chembl_target_id": "CHEMBL246",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 18.0,
                    "therapeutic_pathway": "Gastric mucosa cytoprotection [EMA/HMPC/578322/2008 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.72,
                }
            ],
            "mechanistic_alignment_score": 0.68,
        },
    ],
    "Artemisia annua": [
        {
            "compound_name": "Artemisinin",
            "smiles": "CC1CCC2C(C(=O)OC3C24C1CCC(O3)(OO4)C)C",
            "inchi_key": "BLUAFEHZUWYNDE-NNWCWBAJSA-N",
            "doi": "10.1038/nature01813 (Eckstein-Ludwig et al., 2003, Nature) & 10.1038/nm.2471 (Tu, 2011, Nature Med)",
            "documented_organs": ["leaf", "flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Plasmodium falciparum sarco/endoplasmic reticulum calcium ATPase (PfATP6)",
                    "gene_symbol": "PfATP6",
                    "chembl_target_id": "CHEMBL2366517 (UniProt:Q08853)",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 0.009,
                    "therapeutic_pathway": "Heme-activated radical cleavage & antimalarial parasiticide [10.1038/nature01813 & 10.1038/nm.2471]",
                    "symptom_relevance_score": 0.99,
                }
            ],
            "mechanistic_alignment_score": 0.96,
        },
        {
            "compound_name": "Arteannuin B",
            "smiles": "CC1CCC2C(=C)C(=O)OC23C1CCC4(C3O4)C",
            "inchi_key": "QWQSMEDUZQDVLA-KPHNHPKPSA-N",
            "doi": "10.1038/nm.2471 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["leaf", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Plasmodium falciparum dihydroorotate dehydrogenase",
                    "gene_symbol": "PfDHODH",
                    "chembl_target_id": "CHEMBL4523315",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 5.4,
                    "therapeutic_pathway": "Pyrimidine biosynthesis inhibition in protozoans [10.1038/nm.2471 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.70,
                }
            ],
            "mechanistic_alignment_score": 0.65,
        },
        {
            "compound_name": "Scopoletin",
            "smiles": "COC1=C(C=C2C(=C1)C=CC(=O)O2)O",
            "inchi_key": "RODXRVNMMDRFIK-UHFFFAOYSA-N",
            "doi": "10.1038/nm.2471 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["leaf", "stem", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Interleukin 6 / NF-kB pathway",
                    "gene_symbol": "IL6",
                    "chembl_target_id": "CHEMBL2147",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 14.0,
                    "therapeutic_pathway": "Febrile cytokine downregulation & antipyresis [10.1038/nm.2471 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.88,
                }
            ],
            "mechanistic_alignment_score": 0.80,
        },
        {
            "compound_name": "Chrysosplenol D",
            "smiles": "COC1=C(C(=C2C(=C1)OC(=C(C2=O)OC)C3=CC(=C(C=C3)O)O)O)OC",
            "inchi_key": "BYWLLSQTJBXAPV-UHFFFAOYSA-N",
            "doi": "10.1038/nm.2471 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["leaf", "flower", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Multidrug resistance protein 1 (P-glycoprotein efflux pump)",
                    "gene_symbol": "ABCB1",
                    "chembl_target_id": "CHEMBL4302",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 2.8,
                    "therapeutic_pathway": "Bioavailability enhancement / synergy with artemisinin [10.1038/nm.2471 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.90,
                }
            ],
            "mechanistic_alignment_score": 0.84,
        },
    ],
    "Symphytum officinale": [
        {
            "compound_name": "Allantoin",
            "smiles": "C1(C(=O)NC(=O)N1)NC(=O)N",
            "inchi_key": "POJWUDADGALRAB-UHFFFAOYSA-N",
            "doi": "10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009",
            "documented_organs": ["root", "leaf", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Epithelial growth factor receptor pathway / Keratinocyte stimulation",
                    "gene_symbol": "EGFR",
                    "chembl_target_id": "CHEMBL203",
                    "action_type": "modulator",
                    "bioactivity_potency_um": 25.0,
                    "therapeutic_pathway": "Mitotic tissue regeneration, cell proliferation & wound repair [10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009]",
                    "symptom_relevance_score": 0.97,
                }
            ],
            "mechanistic_alignment_score": 0.92,
        },
        {
            "compound_name": "Rosmarinic acid",
            "smiles": "C1=CC(=C(C=C1CC(C(=O)O)OC(=O)C=CC2=CC(=C(C=C2)O)O)O)O",
            "inchi_key": "DOUMFZQKYFQNTF-WUTVXBCWSA-N",
            "doi": "10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009",
            "documented_organs": ["leaf", "aerial_parts"],
            "known_targets": [
                {
                    "target_name": "Lipoxygenase-5",
                    "gene_symbol": "ALOX5",
                    "chembl_target_id": "CHEMBL249",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 3.8,
                    "therapeutic_pathway": "Leukotriene suppression & periosteal inflammation resolution [10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009]",
                    "symptom_relevance_score": 0.90,
                }
            ],
            "mechanistic_alignment_score": 0.85,
        },
        {
            "compound_name": "Symphytine",
            "smiles": "CC=C(C)C(=O)OC1CCN2C1C(=CC2)COC(=O)C(C(C)C)(C(C)O)O",
            "inchi_key": "MVWPTZQHBOWRTF-SMLWLWDZSA-N",
            "doi": "10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009",
            "documented_organs": ["root", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Cytochrome P450 3A4 (Bioactivation to reactive pyrrolic esters)",
                    "gene_symbol": "CYP3A4",
                    "chembl_target_id": "CHEMBL340",
                    "action_type": "substrate",
                    "bioactivity_potency_um": 1.5,
                    "therapeutic_pathway": "Hepatotoxic sinusoidal obstruction syndrome & DNA alkylation [10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009]",
                    "symptom_relevance_score": 0.10,
                }
            ],
            "mechanistic_alignment_score": 0.10,
        },
        {
            "compound_name": "Echimidine",
            "smiles": "CC=C(C)C(=O)OC1CCN2C1C(=CC2)COC(=O)C(C(C)O)(C(C)(C)O)O",
            "inchi_key": "HRSGCYGUWHGOPY-LYHHMGRNSA-N",
            "doi": "10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009",
            "documented_organs": ["root", "rhizome"],
            "known_targets": [
                {
                    "target_name": "Cytochrome P450 3A4 (Bioactivation)",
                    "gene_symbol": "CYP3A4",
                    "chembl_target_id": "CHEMBL340",
                    "action_type": "substrate",
                    "bioactivity_potency_um": 1.8,
                    "therapeutic_pathway": "Hepatotoxic veno-occlusive disease [10.1002/ptr.4612 (Staiger, 2012) & EMA/HMPC/572844/2009]",
                    "symptom_relevance_score": 0.10,
                }
            ],
            "mechanistic_alignment_score": 0.10,
        },
    ],
    "Cinchona pubescens": [
        {
            "compound_name": "Quinine",
            "smiles": "C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@@H](O)c1ccnc2ccc(OC)cc12",
            "inchi_key": "LOUPRKONTZGTKE-LHHVKLHASA-N",
            "doi": "10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["bark", "trunk_bark", "root_bark"],
            "known_targets": [
                {
                    "target_name": "Plasmodium falciparum chloroquine resistance transporter (PfCRT) / Hemozoin crystallization",
                    "gene_symbol": "PfCRT",
                    "chembl_target_id": "UniProt:Q9N623 (PfCRT)",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 0.12,
                    "therapeutic_pathway": "Toxic ferriprotoporphyrin IX biocrystallization inhibition & malarial parasiticide [10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.99,
                }
            ],
            "mechanistic_alignment_score": 0.96,
        },
        {
            "compound_name": "Cinchonine",
            "smiles": "C=CC1CN2CCC1CC2C(O)c1ccnc2ccccc12",
            "inchi_key": "KMPWYEUPVWOPIM-UHFFFAOYSA-N",
            "doi": "10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["bark", "trunk_bark", "root_bark"],
            "known_targets": [
                {
                    "target_name": "Plasmodium falciparum hemozoin biocrystallization",
                    "gene_symbol": "PfCRT",
                    "chembl_target_id": "UniProt:Q9N623 (PfCRT)",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 0.45,
                    "therapeutic_pathway": "Malarial hemozoin disruption & febrifuge action [10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.92,
                }
            ],
            "mechanistic_alignment_score": 0.88,
        },
        {
            "compound_name": "Quinidine",
            "smiles": "C=C[C@@H]1CN2CC[C@@H]1C[C@@H]2[C@H](O)c1ccnc2ccc(OC)cc12",
            "inchi_key": "LOUPRKONTZGTKE-UYVJDWJCSA-N",
            "doi": "10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["bark", "trunk_bark", "root_bark"],
            "known_targets": [
                {
                    "target_name": "Sodium channel protein type 5 subunit alpha",
                    "gene_symbol": "SCN5A",
                    "chembl_target_id": "CHEMBL2047",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 2.1,
                    "therapeutic_pathway": "Cardiac class Ia antiarrhythmic action & antimalarial activity [10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.88,
                }
            ],
            "mechanistic_alignment_score": 0.84,
        },
    ],
    "Papaver somniferum": [
        {
            "compound_name": "Morphine",
            "smiles": "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
            "inchi_key": "BQJCRHHNABKAKU-UHFFFAOYSA-N",
            "doi": "10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["fruit", "seed_capsule", "exudate_latex"],
            "known_targets": [
                {
                    "target_name": "Mu-type opioid receptor",
                    "gene_symbol": "OPRM1",
                    "chembl_target_id": "CHEMBL233",
                    "action_type": "agonist",
                    "bioactivity_potency_um": 0.002,
                    "therapeutic_pathway": "Supraspinal and spinal analgesia & nociceptive inhibition [10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.99,
                }
            ],
            "mechanistic_alignment_score": 0.98,
        },
        {
            "compound_name": "Codeine",
            "smiles": "COc1ccc2c3c1OC1C(O)C=CC4C(C2)N(C)CCC341",
            "inchi_key": "OROGSEYTTFOCAN-UHFFFAOYSA-N",
            "doi": "10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["fruit", "seed_capsule", "exudate_latex"],
            "known_targets": [
                {
                    "target_name": "Mu-type opioid receptor (via metabolic conversion to morphine)",
                    "gene_symbol": "OPRM1",
                    "chembl_target_id": "CHEMBL233",
                    "action_type": "agonist",
                    "bioactivity_potency_um": 0.20,
                    "therapeutic_pathway": "Central antitussive reflex depression & moderate analgesia [10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.92,
                }
            ],
            "mechanistic_alignment_score": 0.90,
        },
        {
            "compound_name": "Papaverine",
            "smiles": "COc1ccc(Cc2nc3cc(OC)c(OC)cc3cc2OC)cc1OC",
            "inchi_key": "YMIAKQJJUDUNFI-UHFFFAOYSA-N",
            "doi": "10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)",
            "documented_organs": ["fruit", "seed_capsule", "exudate_latex"],
            "known_targets": [
                {
                    "target_name": "Phosphodiesterase 10A / Phosphodiesterase 4",
                    "gene_symbol": "PDE10A",
                    "chembl_target_id": "CHEMBL4409",
                    "action_type": "inhibitor",
                    "bioactivity_potency_um": 0.85,
                    "therapeutic_pathway": "Visceral smooth muscle spasmolysis & peripheral vasodilation [10.1093/nar/gky1075 & 10.7554/eLife.70780 (LOTUS)]",
                    "symptom_relevance_score": 0.94,
                }
            ],
            "mechanistic_alignment_score": 0.88,
        },
    ],
}

# Known GBIF keys mapped to canonical benchmark species
GBIF_BENCHMARK_TAXON_MAP: Dict[int, str] = {
    3170040: "Gentiana lutea",
    3170154: "Gentiana lutea",
    3189486: "Hypericum perforatum",
    8373355: "Artemisia annua",
    3120644: "Artemisia annua",
    7558421: "Symphytum officinale",
    2926101: "Symphytum officinale",
    3019808: "Cinchona pubescens",
    3019809: "Cinchona pubescens",
    3019803: "Cinchona pubescens",
    2888448: "Papaver somniferum",
    8101486: "Papaver somniferum",
}

# In-memory registry for dynamically loaded Parquet/CSV dataset subsets
_LOADED_LOTUS_TAXON_MAP: Dict[str, List[Dict[str, Any]]] = {}
_LOADED_LOTUS_GBIF_MAP: Dict[int, List[Dict[str, Any]]] = {}


# ==============================================================================
# Normalization Utilities
# ==============================================================================

def normalize_botanical_name(scientific_name: str) -> str:
    """Normalize accepted botanical scientific name by removing author strings and infraspecific markers.

    Extracts the canonical binomial (Genus species).

    Examples:
        'Gentiana lutea L.' -> 'Gentiana lutea'
        'Hypericum perforatum L.' -> 'Hypericum perforatum'
        'Artemisia annua Mill.' -> 'Artemisia annua'
        'Gentiana lutea subsp. lutea' -> 'Gentiana lutea'
        'Cinchona pubescens Vahl' -> 'Cinchona pubescens'

    Args:
        scientific_name: Raw accepted scientific name.

    Returns:
        Canonical binomial string 'Genus species'.
    """
    clean = scientific_name.strip()
    if not clean:
        return ""

    tokens = clean.split()
    if len(tokens) >= 2:
        genus = tokens[0].capitalize()
        # Remove any punctuation from species epithet
        species = tokens[1].lower().rstrip(",.;:")
        return f"{genus} {species}"
    return clean.capitalize()


# ==============================================================================
# Parquet and CSV Dataset Loading
# ==============================================================================

def load_lotus_subset(
    filepath: Union[str, Path],
    clear_existing: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load a subset of the LOTUS Zenodo dataset from Parquet or CSV into the in-memory registry.

    Capable of indexing by normalized species name and by GBIF taxon key.

    Supported file formats:
      * `.parquet` / `.pq`
      * `.csv` / `.tsv` / `.txt`

    Args:
        filepath: Path to local Parquet or CSV extract.
        clear_existing: If True, clears previously loaded external records.

    Returns:
        Dictionary mapping normalized scientific names to lists of chemical records.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"LOTUS subset file not found at: {path}")

    if clear_existing:
        _LOADED_LOTUS_TAXON_MAP.clear()
        _LOADED_LOTUS_GBIF_MAP.clear()

    # 1. Read file into DataFrame
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Must be .parquet or .csv")

    # 2. Identify and standardize column mappings (case-insensitive)
    col_lookup = {col.lower().strip(): col for col in df.columns}

    taxon_col = (
        col_lookup.get("organism_name")
        or col_lookup.get("taxon_name")
        or col_lookup.get("species")
        or col_lookup.get("scientific_name")
        or col_lookup.get("accepted_scientific_name")
    )
    gbif_col = (
        col_lookup.get("organism_gbif_id")
        or col_lookup.get("gbif_usage_key")
        or col_lookup.get("gbif_id")
        or col_lookup.get("gbif_taxon_key")
    )
    compound_col = (
        col_lookup.get("compound_name")
        or col_lookup.get("traditional_name")
        or col_lookup.get("name")
        or col_lookup.get("chemical_name")
    )
    smiles_col = (
        col_lookup.get("smiles")
        or col_lookup.get("canonical_smiles")
        or col_lookup.get("connectivity_smiles")
    )
    inchi_col = (
        col_lookup.get("inchi_key")
        or col_lookup.get("inchikey")
    )
    doi_col = (
        col_lookup.get("doi")
        or col_lookup.get("reference_doi")
        or col_lookup.get("lotus_doi")
    )
    organs_col = (
        col_lookup.get("documented_organs")
        or col_lookup.get("plant_part")
        or col_lookup.get("organ")
    )

    if not smiles_col or not inchi_col or not (taxon_col or gbif_col):
        raise ValueError(
            f"Dataset at {path} lacks required columns (needs SMILES, InChIKey, and Taxon/GBIF identifier). "
            f"Found: {list(df.columns)}"
        )

    loaded_count = 0
    for _, row in df.iterrows():
        smi = str(row[smiles_col]).strip() if pd.notna(row[smiles_col]) else None
        ikey = str(row[inchi_col]).strip() if pd.notna(row[inchi_col]) else None
        if not smi or not ikey:
            continue

        c_name = str(row[compound_col]).strip() if compound_col and pd.notna(row[compound_col]) else "Unnamed natural product"
        doi = str(row[doi_col]).strip() if doi_col and pd.notna(row[doi_col]) else None

        raw_organs = row[organs_col] if organs_col and pd.notna(row[organs_col]) else []
        if isinstance(raw_organs, str):
            documented_organs = [o.strip().lower() for o in raw_organs.split(",") if o.strip()]
        elif isinstance(raw_organs, (list, tuple)):
            documented_organs = [str(o).strip().lower() for o in raw_organs]
        else:
            documented_organs = []

        calc_ikey = None
        if smi:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    calc_ikey = Chem.MolToInchiKey(mol)
            except Exception:
                pass
        final_ikey = calc_ikey or ikey.upper()

        record = {
            "compound_name": c_name,
            "smiles": smi,
            "inchi_key": final_ikey,
            "doi": doi or LOTUS_BASELINE_DOI,
            "documented_organs": documented_organs,
        }

        # Index by species name
        if taxon_col and pd.notna(row[taxon_col]):
            norm_name = normalize_botanical_name(str(row[taxon_col]))
            if norm_name:
                _LOADED_LOTUS_TAXON_MAP.setdefault(norm_name, []).append(record)

        # Index by GBIF key
        if gbif_col and pd.notna(row[gbif_col]):
            try:
                g_key = int(row[gbif_col])
                _LOADED_LOTUS_GBIF_MAP.setdefault(g_key, []).append(record)
            except (ValueError, TypeError):
                pass

        loaded_count += 1

    logger.info(f"Loaded {loaded_count} chemical-taxon occurrence records from {path}")
    return _LOADED_LOTUS_TAXON_MAP


# ==============================================================================
# Compound Retrieval
# ==============================================================================

def get_compounds_for_taxon(
    accepted_scientific_name: str,
    gbif_usage_key: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve known secondary metabolites for a botanical taxon.

    Matches by normalized binomial or GBIF usageKey across both the offline curated
    benchmark set and any dynamically loaded LOTUS Parquet/CSV subsets.

    Args:
        accepted_scientific_name: Botanical accepted binomial with or without authorities.
        gbif_usage_key: Optional GBIF usageKey/taxonKey.

    Returns:
        List of chemical dictionaries with keys: compound_name, smiles, inchi_key, doi,
        documented_organs, known_targets, mechanistic_alignment_score.
    """
    norm_name = normalize_botanical_name(accepted_scientific_name)
    candidates: List[Dict[str, Any]] = []

    # 1. Check dynamically loaded Parquet/CSV data
    if gbif_usage_key and gbif_usage_key in _LOADED_LOTUS_GBIF_MAP:
        candidates.extend(_LOADED_LOTUS_GBIF_MAP[gbif_usage_key])
    if norm_name and norm_name in _LOADED_LOTUS_TAXON_MAP:
        candidates.extend(_LOADED_LOTUS_TAXON_MAP[norm_name])

    # 2. Check offline curated benchmark dataset
    # Direct name match
    if norm_name in EMBEDDED_LOTUS_BENCHMARK:
        candidates.extend(EMBEDDED_LOTUS_BENCHMARK[norm_name])
    # GBIF key match
    elif gbif_usage_key and gbif_usage_key in GBIF_BENCHMARK_TAXON_MAP:
        mapped_name = GBIF_BENCHMARK_TAXON_MAP[gbif_usage_key]
        if mapped_name in EMBEDDED_LOTUS_BENCHMARK:
            candidates.extend(EMBEDDED_LOTUS_BENCHMARK[mapped_name])

    # 3. Fallback: try loading from reference JSON on disk if empty
    if not candidates and DEFAULT_CURATED_JSON_PATH.is_file():
        try:
            with open(DEFAULT_CURATED_JSON_PATH, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
            if norm_name in disk_data:
                candidates.extend(disk_data[norm_name])
            elif gbif_usage_key and gbif_usage_key in GBIF_BENCHMARK_TAXON_MAP:
                mapped_name = GBIF_BENCHMARK_TAXON_MAP[gbif_usage_key]
                if mapped_name in disk_data:
                    candidates.extend(disk_data[mapped_name])
        except Exception as exc:
            logger.warning(f"Failed reading disk JSON cache {DEFAULT_CURATED_JSON_PATH}: {exc}")

    # 4. De-duplicate compounds by canonical InChIKey
    seen_inchis: Set[str] = set()
    unique_compounds: List[Dict[str, Any]] = []

    for item in candidates:
        smi = item.get("smiles")
        calc_ikey = None
        if smi:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    calc_ikey = Chem.MolToInchiKey(mol)
            except Exception:
                pass
        raw_ikey = str(item.get("inchi_key", "")).strip().upper()
        ikey = calc_ikey or raw_ikey
        if ikey and ikey not in seen_inchis:
            seen_inchis.add(ikey)
            updated_item = dict(item)
            updated_item["inchi_key"] = ikey
            if not updated_item.get("doi"):
                updated_item["doi"] = LOTUS_BASELINE_DOI
            unique_compounds.append(updated_item)

    return unique_compounds


# ==============================================================================
# Chemical Enrichment & Plausibility Evaluation
# ==============================================================================

def enrich_taxon_compounds(
    accepted_name: str,
    solvent_class: str,
    thermal_regime: str,
    gbif_usage_key: Optional[int] = None,
    estimated_ph: Optional[float] = None,
    duration_minutes: Optional[float] = None,
    temperature_celsius: Optional[float] = None,
    prepared_part: str = "unknown",
) -> List[PhytochemicalRecord]:
    """Retrieve secondary metabolites for a taxon and compute physicochemical extraction plausibility.

    Interfaces with `src.cheminformatics` to calculate 2D descriptors, detect toxicity alerts,
    evaluate SMARTS ionization (logD at recipe pH), calculate Arrhenius thermal survival,
    apply organ localization factors, and score continuous extraction plausibility.

    Args:
        accepted_name: Botanical accepted scientific name (e.g., 'Gentiana lutea L.').
        solvent_class: Extraction solvent class (e.g., 'water', 'lipid_oil', 'vinegar_acidic').
        thermal_regime: Thermal conditions (e.g., 'boiling_prolonged', 'cold_ambient').
        gbif_usage_key: Optional GBIF backbone usage key.
        estimated_ph: Optional operating pH of solvent.
        duration_minutes: Optional contact time in minutes.
        temperature_celsius: Optional extraction temperature in Celsius.
        prepared_part: Plant anatomical organ extracted (e.g., 'root', 'leaf', 'flower').

    Returns:
        List of enriched, validated PhytochemicalRecord instances.
    """
    raw_compounds = get_compounds_for_taxon(accepted_name, gbif_usage_key=gbif_usage_key)
    enriched_records: List[PhytochemicalRecord] = []

    for comp in raw_compounds:
        # Parse targets if present
        raw_targets = comp.get("known_targets", [])
        targets: List[TargetMechanismRecord] = []
        for t in raw_targets:
            if isinstance(t, TargetMechanismRecord):
                targets.append(t)
            elif isinstance(t, dict):
                try:
                    targets.append(TargetMechanismRecord(**t))
                except Exception as e:
                    logger.warning(f"Error parsing TargetMechanismRecord for {comp.get('compound_name')}: {e}")

        documented_organs = comp.get("documented_organs", [])
        align_score = float(comp.get("mechanistic_alignment_score", 0.0))

        record = create_phytochemical_record(
            compound_name=comp["compound_name"],
            smiles=comp["smiles"],
            solvent_class=solvent_class,
            thermal_regime=thermal_regime,
            provenance_doi=comp.get("doi") or LOTUS_BASELINE_DOI,
            estimated_ph=estimated_ph,
            duration_minutes=duration_minutes,
            temperature_celsius=temperature_celsius,
            prepared_part=prepared_part,
            documented_organs=documented_organs,
            known_targets=targets,
            mechanistic_alignment_score=align_score,
        )
        if record is not None:
            enriched_records.append(record)

    return enriched_records


# ==============================================================================
# Cache Persistence Utility
# ==============================================================================

def export_curated_benchmark_to_json(filepath: Union[str, Path] = DEFAULT_CURATED_JSON_PATH) -> Path:
    """Save the embedded curated benchmark dataset to disk in JSON format.

    Args:
        filepath: Destination file path.

    Returns:
        Resolved Path object.
    """
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(EMBEDDED_LOTUS_BENCHMARK, f, indent=2)
    logger.info(f"Exported curated benchmark to {dest}")
    return dest


# ==============================================================================
# CLI Entry Point & Verification
# ==============================================================================

def main() -> None:
    """CLI demonstration and verification for the LOTUS retriever."""
    print("=" * 105)
    print("DecoctionGraph: Physical Chemistry LOTUS Natural Products Retriever & Preparation Engine")
    print("=" * 105)

    test_species = [
        ("Gentiana lutea L.", "water", "boiling_prolonged", 3170040, 7.0, 30.0, 100.0, "root"),
        ("Hypericum perforatum L.", "lipid_oil", "cold_ambient", 3189486, 7.0, 1440.0, 25.0, "flower"),
        ("Artemisia annua L.", "water", "boiling_prolonged", 8373355, 7.0, 30.0, 100.0, "leaf"),
        ("Symphytum officinale L.", "water", "boiling_prolonged", 7558421, 7.0, 30.0, 100.0, "root"),
        ("Symphytum officinale L.", "water", "boiling_prolonged", 7558421, 7.0, 30.0, 100.0, "leaf"),
        ("Cinchona pubescens Vahl", "vinegar_acidic", "cold_ambient", 3019808, 3.0, 60.0, 25.0, "bark"),
        ("Papaver somniferum L.", "water", "warm_steep", 2888448, 7.0, 30.0, 80.0, "fruit"),
    ]

    for sci_name, solvent, thermal, g_key, ph, dur, temp, part in test_species:
        print(f"\nEvaluating: {sci_name} (GBIF: {g_key})")
        print(f"Preparation: Part={part} | Solvent={solvent} (pH={ph}) | Temp={temp}C | Duration={dur}m")
        print("-" * 105)

        records = enrich_taxon_compounds(
            accepted_name=sci_name,
            solvent_class=solvent,
            thermal_regime=thermal,
            gbif_usage_key=g_key,
            estimated_ph=ph,
            duration_minutes=dur,
            temperature_celsius=temp,
            prepared_part=part,
        )

        header = f"{'Compound Name':<18} | {'Class':<14} | {'logD':<6} | {'Organ':<6} | {'Surv':<6} | {'Plausibility':<12} | {'Verdict':<9} | {'Alerts'}"
        print(header)
        print("-" * 105)

        for rec in records:
            alerts_str = ", ".join(rec.toxicity_alerts) if rec.toxicity_alerts else "None"
            print(
                f"{rec.compound_name:<18} | "
                f"{rec.acid_base_class:<14} | "
                f"{rec.logd_at_recipe_ph:<6.2f} | "
                f"{rec.organ_match_factor:<6.2f} | "
                f"{rec.thermal_survival_fraction:<6.2f} | "
                f"{rec.extraction_plausibility_score:<12.4f} | "
                f"{rec.extraction_verdict:<9} | "
                f"{alerts_str}"
            )

    print("\n" + "=" * 105)
    print("LOTUS Physical Chemistry Engine Demonstration Completed Successfully.")
    print("=" * 105)


if __name__ == "__main__":
    main()
