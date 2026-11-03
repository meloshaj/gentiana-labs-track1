"""DecoctionGraph LOTUS Knowledge Retriever Module.

Maps standardized botanical taxa to known phytochemical constituents using the LOTUS
(Natural Products Online) dataset. Supports dual operating modes:
  1. Offline Curated Benchmark Mode: Fast offline lookup of verified secondary metabolites
     for key benchmark taxa (Hypericum perforatum, Gentiana lutea, Artemisia annua,
     Symphytum officinale).
  2. Parquet/CSV Loading Mode: Streaming/batch querying of local LOTUS extracts (Zenodo dump).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from src.schemas import PhytochemicalRecord, SolventClass, ThermalRegime
from src.cheminformatics import create_phytochemical_record

logger = logging.getLogger(__name__)

DEFAULT_CURATED_JSON_PATH = Path("data/reference/lotus_curated_sample.json")

# ==============================================================================
# Built-In Curated Offline Benchmark Dataset
# ==============================================================================

# Embedded authoritative benchmark dataset for offline operation and deterministic tests
EMBEDDED_LOTUS_BENCHMARK: Dict[str, List[Dict[str, str]]] = {
    "Hypericum perforatum": [
        {
            "compound_name": "Hypericin",
            "smiles": "CC1=CC(=O)C2=C(C3=C(C=C(C4=C3C5=C2C1=C6C(=CC(=O)C7=C(C8=C(C=C(C4=C8C5=C67)O)O)O)C)O)O)O",
            "inchi_key": "YDOIFHVUBCIUHF-UHFFFAOYSA-N",
            "doi": "10.1016/j.phytochem.2004.08.022",
        },
        {
            "compound_name": "Hyperforin",
            "smiles": "CC(C)C(=O)C12C(=O)C(=C(C(C1=O)(CC(C2(C)CCC=C(C)C)CC=C(C)C)CC=C(C)C)O)CC=C(C)C",
            "inchi_key": "KGSZHKRKHXOAMG-HQKKAZOISA-N",
            "doi": "10.1055/s-2003-40654",
        },
        {
            "compound_name": "Rutin",
            "smiles": "CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C=C5)O)O)O)O)O)O)O)O",
            "inchi_key": "IKGXIBQEEMLURG-NVPNHPEKSA-N",
            "doi": "10.1016/j.jep.2011.08.077",
        },
        {
            "compound_name": "Quercetin",
            "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
            "inchi_key": "REFJWTPEDVJJIY-UHFFFAOYSA-N",
            "doi": "10.1016/j.fitote.2019.104251",
        },
        {
            "compound_name": "Chlorogenic acid",
            "smiles": "C1C(C(C(CC1(C(=O)O)O)OC(=O)C=CC2=CC(=C(C=C2)O)O)O)O",
            "inchi_key": "CWVRJTMFETXNAD-JUHZACGLSA-N",
            "doi": "10.1002/ptr.2612",
        },
        {
            "compound_name": "Pseudohypericin",
            "smiles": "CC1=CC(=O)C2=C(C3=C(C=C(C4=C3C5=C2C1=C6C(=CC(=O)C7=C(C8=C(C=C(C4=C8C5=C67)O)O)O)CO)O)O)O",
            "inchi_key": "NODGUBIGZKATOM-UHFFFAOYSA-N",
            "doi": "10.1016/S0031-9422(00)80442-9",
        },
        {
            "compound_name": "Biapigenin",
            "smiles": "C1=CC(=CC=C1C2=CC(=O)C3=C(O2)C(=C(C=C3O)O)C4=C(OC5=CC(=CC(=C5C4=O)O)O)C6=CC=C(C=C6)O)O",
            "inchi_key": "IQAMTZLKUHMPPE-UHFFFAOYSA-N",
            "doi": "10.1055/s-2006-957833",
        },
    ],
    "Gentiana lutea": [
        {
            "compound_name": "Gentiopicroside",
            "smiles": "C=CC1C(OC=C2C1=CCOC2=O)OC3C(C(C(C(O3)CO)O)O)O",
            "inchi_key": "DUAGQYUORDTXOR-GPQRQXLASA-N",
            "doi": "10.1016/j.jep.2017.03.017",
        },
        {
            "compound_name": "Amarogentin",
            "smiles": "C=CC1C2CCOC(=O)C2=COC1OC3C(C(C(C(O3)CO)O)O)OC(=O)C4=C(C=C(C=C4O)O)C5=CC(=CC=C5)O",
            "inchi_key": "DBOVHQOUSDWAPQ-WTONXPSSSA-N",
            "doi": "10.1016/j.phytochem.2016.12.015",
        },
        {
            "compound_name": "Gentisin",
            "smiles": "COC1=CC(=C2C(=C1)OC3=C(C2=O)C=C(C=C3)O)O",
            "inchi_key": "XOXYHGOIRWABTC-UHFFFAOYSA-N",
            "doi": "10.1021/np980182q",
        },
        {
            "compound_name": "Swertiamarin",
            "smiles": "C=CC1C(OC=C2C1(CCOC2=O)O)OC3C(C(C(C(O3)CO)O)O)O",
            "inchi_key": "HEYZWPRKKUGDCR-QBXMEVCASA-N",
            "doi": "10.1016/j.phytochem.2010.05.003",
        },
        {
            "compound_name": "Loganic acid",
            "smiles": "CC1C(CC2C1C(OC=C2C(=O)O)OC3C(C(C(C(O3)CO)O)O)O)O",
            "inchi_key": "JNNGEAWILNVFFD-CDJYTOATSA-N",
            "doi": "10.1016/j.foodchem.2014.07.036",
        },
    ],
    "Artemisia annua": [
        {
            "compound_name": "Artemisinin",
            "smiles": "CC1CCC2C(C(=O)OC3C24C1CCC(O3)(OO4)C)C",
            "inchi_key": "BLUAFEHZUWYNDE-NNWCWBAJSA-N",
            "doi": "10.1016/j.phytochem.2014.04.013",
        },
        {
            "compound_name": "Arteannuin B",
            "smiles": "CC1CCC2C(=C)C(=O)OC23C1CCC4(C3O4)C",
            "inchi_key": "QWQSMEDUZQDVLA-KPHNHPKPSA-N",
            "doi": "10.1016/S0031-9422(00)80540-X",
        },
        {
            "compound_name": "Scopoletin",
            "smiles": "COC1=C(C=C2C(=C1)C=CC(=O)O2)O",
            "inchi_key": "RODXRVNMMDRFIK-UHFFFAOYSA-N",
            "doi": "10.1016/j.phytochem.2017.09.006",
        },
        {
            "compound_name": "Chrysosplenol D",
            "smiles": "COC1=C(C(=C2C(=C1)OC(=C(C2=O)OC)C3=CC(=C(C=C3)O)O)O)OC",
            "inchi_key": "BYWLLSQTJBXAPV-UHFFFAOYSA-N",
            "doi": "10.1016/j.jep.2011.04.051",
        },
    ],
    "Symphytum officinale": [
        {
            "compound_name": "Allantoin",
            "smiles": "C1(C(=O)NC(=O)N1)NC(=O)N",
            "inchi_key": "POJWUDADGALRAB-UHFFFAOYSA-N",
            "doi": "10.1016/j.jep.2012.04.011",
        },
        {
            "compound_name": "Rosmarinic acid",
            "smiles": "C1=CC(=C(C=C1CC(C(=O)O)OC(=O)C=CC2=CC(=C(C=C2)O)O)O)O",
            "inchi_key": "DOUMFZQKYFQNTF-WUTVXBCWSA-N",
            "doi": "10.1016/j.phytochem.2015.02.013",
        },
        {
            "compound_name": "Symphytine",
            "smiles": "CC=C(C)C(=O)OC1CCN2C1C(=CC2)COC(=O)C(C(C)C)(C(C)O)O",
            "inchi_key": "MVWPTZQHBOWRTF-SMLWLWDZSA-N",
            "doi": "10.1016/j.fct.2018.06.024",
        },
        {
            "compound_name": "Echimidine",
            "smiles": "CC=C(C)C(=O)OC1CCN2C1C(=CC2)COC(=O)C(C(C)O)(C(C)(C)O)O",
            "inchi_key": "HRSGCYGUWHGOPY-LYHHMGRNSA-N",
            "doi": "10.1016/j.toxlet.2019.01.008",
        },
    ],
}

# Known GBIF keys mapped to canonical benchmark species
GBIF_BENCHMARK_TAXON_MAP: Dict[int, str] = {
    3170040: "Gentiana lutea",
    3170154: "Gentiana lutea",
    3189486: "Hypericum perforatum",
    8373355: "Artemisia annua",
    7558421: "Symphytum officinale",
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

        record = {
            "compound_name": c_name,
            "smiles": smi,
            "inchi_key": ikey.upper(),
            "doi": doi,
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
        List of chemical dictionaries with keys: compound_name, smiles, inchi_key, doi.
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
        ikey = str(item.get("inchi_key", "")).strip().upper()
        if ikey and ikey not in seen_inchis:
            seen_inchis.add(ikey)
            unique_compounds.append(dict(item))

    return unique_compounds


# ==============================================================================
# Chemical Enrichment & Plausibility Evaluation
# ==============================================================================

def enrich_taxon_compounds(
    accepted_name: str,
    solvent_class: str,
    thermal_regime: str,
    gbif_usage_key: Optional[int] = None,
) -> List[PhytochemicalRecord]:
    """Retrieve secondary metabolites for a taxon and compute physicochemical extraction plausibility.

    Interfaces with `src.cheminformatics` to calculate 2D descriptors, detect toxicity alerts,
    and score continuous extraction plausibility under traditional solvent and thermal conditions.

    Args:
        accepted_name: Botanical accepted scientific name (e.g., 'Gentiana lutea L.').
        solvent_class: Extraction solvent class (e.g., 'water', 'lipid_oil', 'hydroalcoholic').
        thermal_regime: Thermal conditions (e.g., 'boiling_prolonged', 'cold_ambient').
        gbif_usage_key: Optional GBIF backbone usage key.

    Returns:
        List of enriched, validated PhytochemicalRecord instances.
    """
    raw_compounds = get_compounds_for_taxon(accepted_name, gbif_usage_key=gbif_usage_key)
    enriched_records: List[PhytochemicalRecord] = []

    for comp in raw_compounds:
        record = create_phytochemical_record(
            compound_name=comp["compound_name"],
            smiles=comp["smiles"],
            solvent_class=solvent_class,
            thermal_regime=thermal_regime,
            provenance_doi=comp.get("doi"),
        )
        if record is not None:
            enriched_records.append(record)

    return enriched_records


# ==============================================================================
# CLI Entry Point & Verification
# ==============================================================================

def main() -> None:
    """CLI demonstration and verification for the LOTUS retriever."""
    print("=" * 100)
    print("DecoctionGraph: LOTUS Natural Products Retriever & Preparation Filter")
    print("=" * 100)

    test_species = [
        ("Gentiana lutea L.", "water", "boiling_prolonged", 3170040),
        ("Hypericum perforatum L.", "lipid_oil", "cold_ambient", 3189486),
        ("Artemisia annua L.", "water", "boiling_prolonged", 8373355),
        ("Symphytum officinale L.", "water", "boiling_prolonged", 7558421),
    ]

    for sci_name, solvent, thermal, g_key in test_species:
        print(f"\nEvaluating: {sci_name} (GBIF: {g_key})")
        print(f"Preparation Conditions: Solvent={solvent} | Thermal={thermal}")
        print("-" * 100)

        records = enrich_taxon_compounds(
            accepted_name=sci_name,
            solvent_class=solvent,
            thermal_regime=thermal,
            gbif_usage_key=g_key,
        )

        header = f"{'Compound Name':<20} | {'MW':<7} | {'cLogP':<6} | {'TPSA':<6} | {'Plausibility':<12} | {'Verdict':<10} | {'Alerts'}"
        print(header)
        print("-" * 100)

        for rec in records:
            alerts_str = ", ".join(rec.toxicity_alerts) if rec.toxicity_alerts else "None"
            print(
                f"{rec.compound_name:<20} | "
                f"{rec.mw:<7.1f} | "
                f"{rec.clogp:<6.2f} | "
                f"{rec.tpsa:<6.1f} | "
                f"{rec.extraction_plausibility_score:<12.4f} | "
                f"{rec.extraction_verdict:<10} | "
                f"{alerts_str}"
            )

    print("\n" + "=" * 100)
    print("LOTUS Retriever Verification Completed Successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()

