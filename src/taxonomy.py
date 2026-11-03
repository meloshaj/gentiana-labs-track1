"""DecoctionGraph Taxonomic Normalization Module.

Interfaces with the public GBIF Species Backbone API to normalize and resolve
vernacular, archaic, and canonical botanical names to standardized taxonomic
entities (TaxonResolution). Includes local vernacular fallback mapping and
disk/in-memory caching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import unicodedata

import httpx

from src.schemas import TaxonResolution

logger = logging.getLogger(__name__)

# GBIF Species Backbone API endpoint
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CACHE_PATH = Path("data/processed/taxonomy_cache.json")

# ==============================================================================
# Local Vernacular & Historical Regional Fallback Dictionary
# ==============================================================================

LOCAL_FALLBACK_DICTIONARY: Dict[str, Dict[str, Any]] = {
    # Balkan / Mediterranean ethnobotanical folk names
    "kantarion": {
        "scientific_name": "Hypericum perforatum L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3189486,
    },
    "kantarioni": {
        "scientific_name": "Hypericum perforatum L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3189486,
    },
    "st. john's wort": {
        "scientific_name": "Hypericum perforatum L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3189486,
    },
    "lincure": {
        "scientific_name": "Gentiana lutea L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3170040,
    },
    "lincurë": {
        "scientific_name": "Gentiana lutea L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3170040,
    },
    "lincura": {
        "scientific_name": "Gentiana lutea L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3170040,
    },
    "yellow gentian": {
        "scientific_name": "Gentiana lutea L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3170040,
    },
    "pelini": {
        "scientific_name": "Artemisia absinthium L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3120684,
    },
    "pelin": {
        "scientific_name": "Artemisia absinthium L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3120684,
    },
    "wormwood": {
        "scientific_name": "Artemisia absinthium L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3120684,
    },
    "sweet wormwood": {
        "scientific_name": "Artemisia annua L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 8373355,
    },
    "kamomilë": {
        "scientific_name": "Matricaria chamomilla L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3139743,
    },
    "kamomil": {
        "scientific_name": "Matricaria chamomilla L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 3139743,
    },
    "sherbelë": {
        "scientific_name": "Salvia officinalis L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 2926975,
    },
    "hithra": {
        "scientific_name": "Urtica dioica L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 5361762,
    },
    "dëllinja": {
        "scientific_name": "Juniperus communis L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 2684708,
    },
    "dellinja": {
        "scientific_name": "Juniperus communis L.",
        "rank": "species",
        "status": "ACCEPTED",
        "gbif_usage_key": 2684708,
    },
}

# Module-level memory cache
_TAXONOMY_MEMORY_CACHE: Dict[str, TaxonResolution] = {}


# ==============================================================================
# Helper Functions
# ==============================================================================

def normalize_text_key(text: str) -> str:
    """Normalize input string by stripping whitespace and accents for dictionary lookups."""
    clean = text.strip().lower()
    return clean


def lookup_local_fallback(name: str) -> Optional[Dict[str, Any]]:
    """Look up a vernacular or archaic name in the local fallback dictionary.

    Supports direct lowercase and unaccented ASCII fallback matching.
    """
    key = normalize_text_key(name)
    if key in LOCAL_FALLBACK_DICTIONARY:
        return LOCAL_FALLBACK_DICTIONARY[key]

    # Try unaccented key (e.g. "lincure" for "lincurë")
    nfkd = unicodedata.normalize("NFKD", key)
    ascii_key = "".join([c for c in nfkd if not unicodedata.combining(c)])
    if ascii_key in LOCAL_FALLBACK_DICTIONARY:
        return LOCAL_FALLBACK_DICTIONARY[ascii_key]

    return None


def load_taxonomy_cache(cache_path: Union[str, Path] = DEFAULT_CACHE_PATH) -> Dict[str, TaxonResolution]:
    """Load cached TaxonResolution instances from a JSON file."""
    path = Path(cache_path)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            k: TaxonResolution.model_validate(v)
            for k, v in data.items()
        }
    except Exception as exc:
        logger.warning(f"Could not read taxonomy cache from {path}: {exc}")
        return {}


def save_taxonomy_cache(
    cache: Dict[str, TaxonResolution],
    cache_path: Union[str, Path] = DEFAULT_CACHE_PATH,
) -> None:
    """Persist dictionary of TaxonResolution instances to a JSON file."""
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v.model_dump() for k, v in cache.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Could not save taxonomy cache to {path}: {exc}")


# ==============================================================================
# Core Resolution Functions
# ==============================================================================

def resolve_plant_name(
    vernacular_or_archaic_name: str,
    kingdom: str = "Plantae",
    strict: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: Optional[httpx.Client] = None,
) -> TaxonResolution:
    """Resolve a botanical or vernacular plant name against the GBIF Species Backbone API.

    Handles local vernacular fallback mapping, strict HTTP timeouts, confidence score
    normalization [0.0, 1.0], and fallback logic for unresolved taxa.

    Args:
        vernacular_or_archaic_name: Raw botanical binomial, vernacular, or archaic name.
        kingdom: Kingdom constraint for matching (default: "Plantae").
        strict: If True, only exact matches are returned by GBIF.
        timeout: HTTP request timeout in seconds (default: 10.0).
        client: Optional reusable httpx.Client instance.

    Returns:
        TaxonResolution model populated with accepted taxonomy or fallback info.
    """
    clean_name = vernacular_or_archaic_name.strip()
    if not clean_name:
        return TaxonResolution(
            queried_name="Unknown",
            accepted_scientific_name="Unknown plant",
            taxon_rank="unknown",
            taxonomic_status="UNRESOLVED",
            gbif_usage_key=None,
            confidence=0.0,
            match_type="NONE",
            is_resolved=False,
        )

    # Check local fallback dictionary for well-known vernacular or archaic terms
    local_hit = lookup_local_fallback(clean_name)
    query_target = local_hit["scientific_name"] if local_hit else clean_name

    params = {
        "name": query_target,
        "kingdom": kingdom,
        "strict": strict,
    }

    raw_data: Optional[Dict[str, Any]] = None
    http_error: bool = False

    def _execute_request(c: httpx.Client) -> Optional[Dict[str, Any]]:
        resp = c.get(GBIF_MATCH_URL, params=params, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(
                f"GBIF Species Match API returned HTTP {resp.status_code} for '{query_target}'"
            )
            return None
        return resp.json()

    try:
        if client is not None:
            raw_data = _execute_request(client)
        else:
            with httpx.Client(timeout=timeout) as temp_client:
                raw_data = _execute_request(temp_client)
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning(
            f"Network/timeout error querying GBIF API for '{query_target}': {exc}"
        )
        http_error = True

    # Case A: Network failure or HTTP non-200, but we have a local fallback definition
    if raw_data is None:
        if local_hit:
            return TaxonResolution(
                queried_name=clean_name,
                accepted_scientific_name=local_hit["scientific_name"],
                taxon_rank=local_hit.get("rank", "species"),
                taxonomic_status=local_hit.get("status", "ACCEPTED"),
                gbif_usage_key=local_hit.get("gbif_usage_key"),
                confidence=1.0,
                match_type="LOCAL_FALLBACK",
                is_resolved=True,
            )
        return TaxonResolution(
            queried_name=clean_name,
            accepted_scientific_name=clean_name,
            taxon_rank="unknown",
            taxonomic_status="UNRESOLVED",
            gbif_usage_key=None,
            confidence=0.0,
            match_type="NONE",
            is_resolved=False,
        )

    # Case B: Successfully received GBIF JSON payload
    match_type = str(raw_data.get("matchType", "NONE")).upper()
    raw_confidence = float(raw_data.get("confidence", 0.0))
    confidence = round(max(0.0, min(1.0, raw_confidence / 100.0)), 4)
    rank = str(raw_data.get("rank", "unknown")).lower()
    status = str(raw_data.get("status", "UNKNOWN")).upper()

    # Extract usageKey or speciesKey
    usage_key_val = raw_data.get("usageKey") or raw_data.get("speciesKey")
    gbif_usage_key = int(usage_key_val) if usage_key_val and int(usage_key_val) > 0 else None

    # Determine scientific name
    scientific_name = raw_data.get("scientificName") or query_target

    # Detect kingdom-only match (e.g. GBIF couldn't match the name, so it matched kingdom="Plantae")
    is_kingdom_fallback = (match_type == "HIGHERRANK" and rank in ("kingdom", "phylum"))

    # Check unresolved conditions: matchType == "NONE", confidence < 0.50, or kingdom-only
    if match_type == "NONE" or confidence < 0.50 or is_kingdom_fallback:
        # If local fallback is available, rescue with local fallback
        if local_hit:
            return TaxonResolution(
                queried_name=clean_name,
                accepted_scientific_name=local_hit["scientific_name"],
                taxon_rank=local_hit.get("rank", "species"),
                taxonomic_status=local_hit.get("status", "ACCEPTED"),
                gbif_usage_key=local_hit.get("gbif_usage_key"),
                confidence=1.0,
                match_type="LOCAL_FALLBACK",
                is_resolved=True,
            )

        return TaxonResolution(
            queried_name=clean_name,
            accepted_scientific_name=clean_name,
            taxon_rank="unknown",
            taxonomic_status="UNRESOLVED",
            gbif_usage_key=None,
            confidence=confidence if not is_kingdom_fallback else 0.0,
            match_type="NONE" if is_kingdom_fallback else match_type,
            is_resolved=False,
        )

    # If vernacular name was mapped to scientific name before querying GBIF, note match type
    effective_match_type = "VERNACULAR_GBIF" if local_hit else match_type

    return TaxonResolution(
        queried_name=clean_name,
        accepted_scientific_name=scientific_name,
        taxon_rank=rank,
        taxonomic_status=status,
        gbif_usage_key=gbif_usage_key,
        confidence=confidence,
        match_type=effective_match_type,
        is_resolved=True,
    )


def batch_resolve_plants(
    names: List[str],
    cache_path: Union[str, Path] = DEFAULT_CACHE_PATH,
    kingdom: str = "Plantae",
    strict: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, TaxonResolution]:
    """Resolve a batch of botanical names with persistent local disk caching.

    Checks cache first to prevent redundant API queries, resolves uncached names,
    and updates the JSON cache file.

    Args:
        names: List of botanical/vernacular names to resolve.
        cache_path: Path to the JSON cache file (default: data/processed/taxonomy_cache.json).
        kingdom: Kingdom parameter for GBIF query (default: Plantae).
        strict: GBIF strict matching flag.
        timeout: Timeout per HTTP request in seconds.

    Returns:
        Dict mapping each input name to its resolved TaxonResolution.
    """
    path = Path(cache_path)
    disk_cache = load_taxonomy_cache(path)

    # Merge with memory cache
    combined_cache: Dict[str, TaxonResolution] = {**disk_cache, **_TAXONOMY_MEMORY_CACHE}
    results: Dict[str, TaxonResolution] = {}
    cache_modified = False

    # Reuse an HTTP client session across the batch
    with httpx.Client(timeout=timeout) as client:
        for name in names:
            clean_name = name.strip()
            if not clean_name:
                continue

            # Check cache
            if clean_name in combined_cache:
                results[name] = combined_cache[clean_name]
                continue

            # Resolve via API / fallback
            resolution = resolve_plant_name(
                clean_name,
                kingdom=kingdom,
                strict=strict,
                timeout=timeout,
                client=client,
            )

            # Store in caches
            combined_cache[clean_name] = resolution
            _TAXONOMY_MEMORY_CACHE[clean_name] = resolution
            results[name] = resolution
            cache_modified = True

    # Persist updated cache if changes occurred
    if cache_modified:
        save_taxonomy_cache(combined_cache, path)

    return results


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main() -> None:
    """CLI demonstration and verification for the taxonomy normalization module."""
    test_queries = [
        # Canonical scientific binomials
        "Gentiana lutea",
        "Hypericum perforatum",
        "Artemisia annua",
        # Messy archaic synonym
        "Hypericon",
        # Regional vernacular / ethnobotanical folk names
        "Kantarion",
        "Lincurë",
        "Pelini",
        # Fictitious or unresolvable name
        "Nonexistentia herbacea xyz",
    ]

    print("=" * 105)
    print("DecoctionGraph: Taxonomic Normalization CLI (GBIF Backbone + Vernacular Fallbacks)")
    print("=" * 105)

    results = batch_resolve_plants(test_queries)

    header = f"{'Queried Name':<26} | {'Accepted Scientific Name':<28} | {'Rank':<8} | {'Status':<10} | {'UsageKey':<8} | {'Conf':<6} | {'Match':<12} | {'Resolved'}"
    print(header)
    print("-" * 105)

    for query in test_queries:
        res = results.get(query)
        if res is None:
            continue
        key_str = str(res.gbif_usage_key) if res.gbif_usage_key else "None"
        resolved_mark = "[YES]" if res.is_resolved else "[NO]"
        print(
            f"{res.queried_name:<26} | "
            f"{res.accepted_scientific_name:<28} | "
            f"{res.taxon_rank:<8} | "
            f"{res.taxonomic_status:<10} | "
            f"{key_str:<8} | "
            f"{res.confidence:<6.2f} | "
            f"{res.match_type:<12} | "
            f"{resolved_mark}"
        )

    print("=" * 105)
    print(f"Results cached to: {DEFAULT_CACHE_PATH}")


if __name__ == "__main__":
    main()

