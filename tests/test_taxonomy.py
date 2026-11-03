"""Unit tests for DecoctionGraph taxonomic normalization module."""

import json
from pathlib import Path
import pytest
import httpx

from src.schemas import TaxonResolution
from src.taxonomy import (
    LOCAL_FALLBACK_DICTIONARY,
    resolve_plant_name,
    batch_resolve_plants,
    lookup_local_fallback,
    load_taxonomy_cache,
    save_taxonomy_cache,
)


def test_lookup_local_fallback():
    """Test vernacular lookup with exact and accented variants."""
    hit_kantarion = lookup_local_fallback("Kantarion")
    assert hit_kantarion is not None
    assert hit_kantarion["scientific_name"] == "Hypericum perforatum L."

    hit_lincure = lookup_local_fallback("Lincurë")
    assert hit_lincure is not None
    assert hit_lincure["scientific_name"] == "Gentiana lutea L."

    hit_lincure_ascii = lookup_local_fallback("lincure")
    assert hit_lincure_ascii is not None
    assert hit_lincure_ascii["scientific_name"] == "Gentiana lutea L."

    hit_pelini = lookup_local_fallback("Pelini")
    assert hit_pelini is not None
    assert hit_pelini["scientific_name"] == "Artemisia absinthium L."

    assert lookup_local_fallback("UnknownPlantXYZ") is None


def test_resolve_canonical_scientific_name():
    """Verify live GBIF resolution for Gentiana lutea."""
    res = resolve_plant_name("Gentiana lutea")
    assert isinstance(res, TaxonResolution)
    assert res.is_resolved is True
    assert "Gentiana lutea" in res.accepted_scientific_name
    assert res.taxon_rank in ("species", "SPECIES")
    assert res.taxonomic_status == "ACCEPTED"
    assert res.gbif_usage_key == 3170040
    assert res.confidence >= 0.90
    assert res.match_type == "EXACT"


def test_resolve_archaic_synonym():
    """Verify resolution of archaic synonym 'Hypericon'."""
    res = resolve_plant_name("Hypericon")
    assert isinstance(res, TaxonResolution)
    assert res.is_resolved is True
    assert "Hypericon" in res.accepted_scientific_name
    assert res.taxon_rank in ("genus", "GENUS")
    assert res.confidence >= 0.80
    assert res.taxonomic_status == "SYNONYM"


def test_resolve_vernacular_names():
    """Verify resolution of regional folk names mapped through fallback dictionary."""
    for folk_name, expected_scientific in [
        ("Kantarion", "Hypericum perforatum L."),
        ("Lincurë", "Gentiana lutea L."),
        ("Pelini", "Artemisia absinthium L."),
    ]:
        res = resolve_plant_name(folk_name)
        assert res.is_resolved is True
        assert res.queried_name == folk_name
        assert res.accepted_scientific_name == expected_scientific
        assert res.confidence >= 0.80


def test_resolve_unresolvable_taxon():
    """Verify fictitious plant returns is_resolved=False with fallback info."""
    fictitious_name = "Nonexistentia herbacea xyz999"
    res = resolve_plant_name(fictitious_name)
    assert isinstance(res, TaxonResolution)
    assert res.is_resolved is False
    assert res.queried_name == fictitious_name
    assert res.match_type == "NONE"


def test_resolve_empty_name():
    """Verify whitespace or empty query handling."""
    res = resolve_plant_name("   ")
    assert res.is_resolved is False
    assert res.queried_name == "Unknown"


def test_batch_resolve_and_cache(tmp_path: Path):
    """Verify batch resolution, disk cache creation, and subsequent cache hit."""
    test_cache_file = tmp_path / "test_taxonomy_cache.json"
    queries = ["Gentiana lutea", "Artemisia annua", "Lincurë"]

    # First run: queries API/fallback and creates cache
    results_run1 = batch_resolve_plants(queries, cache_path=test_cache_file)
    assert len(results_run1) == 3
    assert test_cache_file.is_file()

    # Verify cache content on disk
    cached_data = load_taxonomy_cache(test_cache_file)
    assert "Gentiana lutea" in cached_data
    assert "Lincurë" in cached_data
    assert cached_data["Gentiana lutea"].is_resolved is True

    # Second run: must load from cache without API error even with invalid URL mock
    results_run2 = batch_resolve_plants(queries, cache_path=test_cache_file)
    assert len(results_run2) == 3
    assert results_run2["Gentiana lutea"].gbif_usage_key == results_run1["Gentiana lutea"].gbif_usage_key


def test_offline_resilience(monkeypatch):
    """Verify graceful handling when GBIF returns HTTP 500 or network error."""
    class Mock500Client:
        def get(self, *args, **kwargs):
            return httpx.Response(status_code=500, content=b"Server error")

    # When querying a vernacular name with network error, local fallback dictionary rescues it
    res_vernacular = resolve_plant_name("Kantarion", client=Mock500Client())
    assert res_vernacular.is_resolved is True
    assert res_vernacular.accepted_scientific_name == "Hypericum perforatum L."
    assert res_vernacular.match_type == "LOCAL_FALLBACK"

    # When querying an unknown name with network error, returns unresolved stub
    res_unknown = resolve_plant_name("UnknownWeed", client=Mock500Client())
    assert res_unknown.is_resolved is False
    assert res_unknown.match_type == "NONE"

