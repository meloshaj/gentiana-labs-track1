"""Unit tests for src/pdf_ingestor.py (Archival PDF & Batch Corpus Ingestion)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from src.pdf_ingestor import (
    extract_text_chunks_from_pdf,
    extract_text_chunks_from_text,
    extract_offline_from_text,
    deduplicate_records,
    process_document_chunks,
    build_atlas_dataset,
    export_atlas_csv,
    export_atlas_jsonld,
    get_sample_corpus_text,
    get_sample_corpus_pdf_bytes,
)
from src.schemas import EthnoPreparationRecord, ExtractionBatch


def test_extract_text_chunks_from_pdf():
    """Verify pypdf extraction preserves page numbers in metadata."""
    pdf_bytes = get_sample_corpus_pdf_bytes()
    chunks = extract_text_chunks_from_pdf(pdf_bytes)

    assert len(chunks) == 3
    page_numbers = [c["page_number"] for c in chunks]
    assert page_numbers == [1, 2, 3]

    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c
        assert len(c["text"]) > 20

    # Page 1 contains Gentiana and Hypericum
    assert "Gentiana lutea" in chunks[0]["text"]
    assert "Hypericum perforatum" in chunks[0]["text"]

    # Page 2 contains Artemisia and Cinchona
    assert "Artemisia annua" in chunks[1]["text"]
    assert "Cinchona pubescens" in chunks[1]["text"]

    # Page 3 contains Symphytum and Papaver
    assert "Symphytum officinale" in chunks[2]["text"]
    assert "Papaver somniferum" in chunks[2]["text"]


def test_extract_text_chunks_from_text_explicit_pages():
    """Verify chunking of text with explicit page markers."""
    raw_text = get_sample_corpus_text()
    chunks = extract_text_chunks_from_text(raw_text)

    assert len(chunks) == 3
    assert [c["page_number"] for c in chunks] == [1, 2, 3]
    assert "Gentiana lutea" in chunks[0]["text"]
    assert "Artemisia annua" in chunks[1]["text"]
    assert "Symphytum officinale" in chunks[2]["text"]


def test_extract_text_chunks_from_text_synthetic_pages():
    """Verify chunking of raw unstructured text into synthetic pages."""
    synthetic_text = "Paragraph one with some ethnobotanical notes.\n\n" * 20
    chunks = extract_text_chunks_from_text(synthetic_text, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    # Page numbers should increment sequentially
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


def test_extract_offline_from_text():
    """Verify deterministic offline extraction of preparation filters from archival text."""
    passage = (
        "Gentiana lutea (Yellow Gentian) dried roots are chopped finely and boiled in mountain "
        "spring water for over an hour to prepare a strong decoction. This bitter draught is "
        "administered orally to cure stomach weakness and chronic dyspepsia."
    )
    citation = "Swiss Alpine Pharmacopoeia (1885) (Page 1)"
    recs = extract_offline_from_text(passage, source_citation=citation, region="Alps")

    assert len(recs) == 1
    r = recs[0]
    assert r.verbatim_plant_name == "Gentiana lutea"
    assert r.plant_part == "root"
    assert r.preparation_type == "decoction"
    assert r.solvent_class == "water"
    assert r.thermal_regime == "boiling_prolonged"
    assert r.administration_route == "oral"
    assert "stomach weakness" in r.verbatim_indication.lower()
    assert r.source_citation == citation


def test_deduplicate_records():
    """Verify deduplication preserves distinct preparations while collapsing true duplicates."""
    r1 = EthnoPreparationRecord(
        verbatim_plant_name="Symphytum officinale",
        plant_part="root",
        preparation_type="poultice",
        solvent_class="none",
        thermal_regime="none",
        verbatim_indication="sprains and broken bones",
        administration_route="topical",
        verbatim_quote="Root poultice bound over sprains.",
        cultural_or_geographic_origin="Europe",
        extraction_confidence=0.85,
    )
    # Higher confidence duplicate of r1
    r2 = r1.model_copy(update={"extraction_confidence": 0.96})

    # Distinct preparation of same plant (leaf tea)
    r3 = EthnoPreparationRecord(
        verbatim_plant_name="Symphytum officinale",
        plant_part="leaf",
        preparation_type="infusion",
        solvent_class="water",
        thermal_regime="warm_steep",
        verbatim_indication="coughs and ulcers",
        administration_route="oral",
        verbatim_quote="Leaf tea for coughs.",
        cultural_or_geographic_origin="Europe",
        extraction_confidence=0.90,
    )

    deduped = deduplicate_records([r1, r2, r3])
    assert len(deduped) == 2
    # The root poultice record should retain the higher confidence score (0.96)
    root_rec = [r for r in deduped if r.plant_part == "root"][0]
    assert root_rec.extraction_confidence == 0.96


def test_process_document_chunks_offline_mode():
    """Verify batch processing across document chunks in fast offline mode."""
    chunks = [
        {
            "page_number": 1,
            "chunk_id": "p1_c1",
            "text": "Gentiana lutea dried roots are boiled in spring water for an hour to prepare a strong decoction for stomach weakness.",
        },
        {
            "page_number": 2,
            "chunk_id": "p2_c1",
            "text": "Artemisia annua fresh leaves are wrung cold through cloth, and the raw juice drunk for intermittent fevers.",
        },
    ]

    progress_log = []
    def on_progress(done, total, msg):
        progress_log.append((done, total, msg))

    batch = process_document_chunks(
        chunks=chunks,
        document_title="Historical Febrifuge Herbal (1890)",
        region="Eurasia",
        progress_callback=on_progress,
        force_offline=True,
    )

    assert isinstance(batch, ExtractionBatch)
    assert len(batch.records) == 2
    assert len(progress_log) == 2

    # Verify that (Page X) is appended to each record's citation for provenance
    assert batch.records[0].source_citation == "Historical Febrifuge Herbal (1890) (Page 1)"
    assert batch.records[1].source_citation == "Historical Febrifuge Herbal (1890) (Page 2)"


def test_build_atlas_dataset():
    """Verify regional ethnopharmacology atlas synthesis and summary metrics."""
    raw_text = get_sample_corpus_text()
    chunks = extract_text_chunks_from_text(raw_text)
    batch = process_document_chunks(chunks, "Alpine Survey (1885)", "European Alps", force_offline=True)

    atlas = build_atlas_dataset(batch)
    assert "atlas_df" in atlas
    assert "metrics" in atlas
    assert "graph_builder" in atlas

    df: pd.DataFrame = atlas["atlas_df"]
    metrics: dict = atlas["metrics"]

    assert not df.empty
    assert metrics["total_pages_scanned"] == 3
    assert metrics["unique_taxa_count"] >= 5
    assert metrics["distinct_preparations_count"] >= 6
    assert metrics["validated_mechanistic_links_count"] >= 1

    # Check key columns
    expected_cols = [
        "Source Page", "Plant Name", "Accepted Taxon", "Plant Part",
        "Preparation", "Solvent", "Thermal Regime", "Historical Indication",
        "Primary Bioactive", "Top Target (Gene)", "Extraction Plausibility",
        "Mechanistic Status", "Toxicity Alerts"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify graph builder contains nodes
    builder = atlas["graph_builder"]
    assert builder.graph.number_of_nodes() > 10
    assert builder.graph.number_of_edges() > 10


def test_export_atlas_csv_and_jsonld():
    """Verify CSV and W3C JSON-LD export formats."""
    raw_text = get_sample_corpus_text()
    chunks = extract_text_chunks_from_text(raw_text)
    batch = process_document_chunks(chunks, "Alpine Survey (1885)", "European Alps", force_offline=True)
    atlas = build_atlas_dataset(batch)

    # 1. CSV export
    csv_str = export_atlas_csv(atlas["atlas_df"])
    assert "Source Page,Plant Name,Accepted Taxon" in csv_str
    assert "Gentiana lutea" in csv_str

    # 2. JSON-LD export
    jsonld_str = export_atlas_jsonld(batch, atlas)
    doc = json.loads(jsonld_str)
    assert "@context" in doc
    assert "@graph" in doc
    assert len(doc["@graph"]) > 5

    # Check for linked entities
    types = [entity.get("@type") for entity in doc["@graph"]]
    assert "schema:CreativeWork" in types
    assert "decoction:PlantTaxon" in types
    assert "decoction:EthnoPreparation" in types
