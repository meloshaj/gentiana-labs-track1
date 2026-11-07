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


def test_four_page_sharr_corpus_preserves_narrative_and_skips_summary_table():
    """Regression coverage for the four-page Sharr Mountains archival layout."""
    corpus = """=== Page 1 ===
Gentiana lutea roots are boiled in water and administered orally 20 minutes before meals to remedy stomach weakness, chronic dyspepsia, sluggish digestion, and complete loss of appetite.
Hypericum perforatum flowering tops are packed into olive oil to treat severe cutaneous burns, deep abrasions, lacerations, and purulent skin wound infections.
The dried leaves and flowers are steeped as a warm aqueous infusion in freshly boiled water at 80°C to soothe internal melancholia, nervous anxiety, mild depression, and restlessness with insomnia. Practitioners never administer the olive oil maceration orally.

=== Page 2 ===
The regional materia medica records several additional preparations and seasonal gathering practices.

=== Page 3 ===
Section 4.1 Cinchona pubescens bark is macerated in vinegar and taken orally to treat shaking chills, intermittent fevers, and ague tremors.
Section 4.2 Papaver somniferum capsules are prepared as a hydroalcoholic tincture and taken orally to arrest intractable postoperative pain, severe visceral colic cramps, and exhausting irritable coughs.
Section 5 Summary Table

=== Page 4 ===
Verbatim & Scientific Taxon | Route & Historical Indication | Temp / Time / pH | Plant Organ | Preparation & Solvent
Cinchona pubescens | shaking chills | 20C | bark | infusion & vinegar
Papaver somniferum | postoperative pain | 25C | capsule | tincture & hydroalcoholic
"""
    chunks = extract_text_chunks_from_text(corpus)
    assert [chunk["page_number"] for chunk in chunks] == [1, 2, 3]
    assert "Verbatim & Scientific Taxon" not in "\n".join(chunk["text"] for chunk in chunks)

    batch = process_document_chunks(chunks, "Traditional Phytotherapeutic Preparations of the Sharr Mountains", force_offline=True)
    records = batch.records
    assert {record.verbatim_plant_name for record in records} >= {
        "Gentiana lutea", "Hypericum perforatum", "Cinchona pubescens", "Papaver somniferum"
    }
    assert any("complete loss of appetite" in record.verbatim_indication for record in records)
    assert any("restlessness with insomnia" in record.verbatim_indication for record in records)
    assert any("ague tremors" in record.verbatim_indication for record in records)
    assert any("exhausting irritable coughs" in record.verbatim_indication for record in records)
    cinchona = next(record for record in records if record.verbatim_plant_name == "Cinchona pubescens")
    assert cinchona.plant_part == "bark"
    assert cinchona.preparation_type == "cold_maceration"
    assert cinchona.solvent_class == "vinegar_acidic"
    tea = next(record for record in records if record.verbatim_plant_name == "Hypericum perforatum" and record.preparation_type == "infusion")
    assert (tea.plant_part, tea.solvent_class, tea.thermal_regime) == ("leaf", "water", "warm_steep")

    atlas = build_atlas_dataset(batch)["atlas_df"]
    tea_row = atlas[(atlas["Plant Name"] == "Hypericum perforatum") & (atlas["Preparation"] == "Infusion")].iloc[0]
    papaver_row = atlas[atlas["Plant Name"] == "Papaver somniferum"].iloc[0]
    assert tea_row["Top Target (Gene)"] in {"MAOA", "SLC6A4"}
    assert tea_row["Mechanistic Status"] == "⭐ Validated Path"
    assert papaver_row["Primary Bioactive"] == "Morphine"
    assert papaver_row["Top Target (Gene)"] == "OPRM1"
    assert papaver_row["Mechanistic Status"] == "⭐ Validated Path"


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


def test_skip_summary_matrix_and_markdown_tables():
    """Verify that summary matrices and markdown reference tables are skipped."""
    text_with_tables = """=== Page 1 ===
Gentiana lutea dried roots are boiled in water for an hour to prepare a decoction for stomach weakness.

=== Page 2 ===
Summary Matrix of Pipeline Results
| Plant | Part | Preparation | Solvent | Thermal | Indication |
| --- | --- | --- | --- | --- | --- |
| Gentiana lutea | root | decoction | water | boiling_prolonged | stomach weakness |
| Hypericum perforatum | flower | oil_maceration | lipid_oil | warm_steep | burns |

=== Page 3 ===
Expected DecoctionGraph Pipeline Outputs:
Taxon \t Preparation \t Target
Gentiana lutea \t decoction \t TAS2R38
"""
    chunks = extract_text_chunks_from_text(text_with_tables)
    # Only Page 1 should be parsed as a chunk; Pages 2 and 3 are summary tables and must be skipped
    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert "Gentiana lutea" in chunks[0]["text"]
    assert "Summary Matrix" not in chunks[0]["text"]


def test_no_cross_sentence_attribute_bleeding_cinchona_and_papaver():
    """Verify Papaver somniferum and Cinchona pubescens extract clean distinct records without indication bleeding."""
    passage = (
        "Cinchona pubescens powdered red bark is steeped in acidic sour wine and vinegar at room temperature "
        "for two days, taken orally against shaking chills and agues.\n\n"
        "Papaver somniferum ripe dried seeds are boiled into a porridge for restless infants. "
        "Conversely, the dried latex harvested from scored green capsules is dissolved in warm wine "
        "as a potent anodyne for intractable colic and insomnia."
    )
    citation = "Herbal Survey (Page 2)"
    recs = extract_offline_from_text(passage, source_citation=citation, region="Europe")

    # Should cleanly extract 3 distinct preparations across the two plants
    assert len(recs) == 3

    cinchona_recs = [r for r in recs if r.verbatim_plant_name == "Cinchona pubescens"]
    papaver_recs = [r for r in recs if r.verbatim_plant_name == "Papaver somniferum"]

    assert len(cinchona_recs) == 1
    assert len(papaver_recs) == 2

    c_rec = cinchona_recs[0]
    assert c_rec.plant_part == "bark"
    assert c_rec.solvent_class == "vinegar_acidic"
    assert "shaking chills and agues" in c_rec.verbatim_indication.lower()
    # Ensure Papaver's colic/insomnia/restless indication did NOT bleed into Cinchona
    assert "colic" not in c_rec.verbatim_indication.lower()
    assert "restless" not in c_rec.verbatim_indication.lower()

    # Papaver seed porridge
    p_seed = [r for r in papaver_recs if r.plant_part == "seed"][0]
    assert p_seed.preparation_type == "decoction"
    assert "restless infants" in p_seed.verbatim_indication.lower()

    # Papaver latex / capsule tincture
    p_latex = [r for r in papaver_recs if r.plant_part == "fruit"][0]
    assert p_latex.preparation_type == "tincture_alcohol"
    assert p_latex.solvent_class == "hydroalcoholic"
    assert "intractable colic and insomnia" in p_latex.verbatim_indication.lower()


def test_clean_indication_leading_preposition_and_hypericum_trpc6_validation():
    """Verify leading prepositional indication extraction and Hypericum perforatum TRPC6 validation."""
    passage = (
        "Hypericum perforatum: For severe thermal burns, deep skin lacerations, abrasions, and purulent cutaneous wounds, "
        "fresh flowering tops are packed into virgin olive oil and set in warm sunlight for forty days."
    )
    citation = "Trauma Herbal (Page 1)"
    recs = extract_offline_from_text(passage, source_citation=citation, region="Balkans")

    assert len(recs) == 1
    r = recs[0]
    assert r.verbatim_plant_name == "Hypericum perforatum"
    assert r.plant_part == "flower"
    assert r.preparation_type == "oil_maceration"
    assert r.solvent_class == "lipid_oil"
    assert r.thermal_regime == "warm_steep"

    # Indication must not fall back to 'Historical illness or ailment' and must strip trailing botanical subject clause
    assert "severe thermal burns" in r.verbatim_indication.lower()
    assert "fresh flowering tops" not in r.verbatim_indication.lower()

    # Verify building atlas dataset yields Validated Path with TRPC6
    batch = ExtractionBatch(source_corpus="Trauma Herbal", records=[r])
    atlas = build_atlas_dataset(batch)
    df = atlas["atlas_df"]
    assert not df.empty
    row = df.iloc[0]
    assert row["Primary Bioactive"] == "Hyperforin"
    assert row["Top Target (Gene)"] == "TRPC6"
    assert row["Mechanistic Status"] == "⭐ Validated Path"


def test_poultice_default_solvent_and_thermal_labels_in_csv():
    """Verify poultice preparations export 'None (Direct/Topical)' and 'Ambient (25°C)' in table/CSV."""
    passage = (
        "Symphytum officinale fresh root is ground into a thick pulp and bound as a poultice "
        "over sprained joints and broken bones to knit the tissues."
    )
    citation = "Bone Setter Manual (Page 1)"
    recs = extract_offline_from_text(passage, source_citation=citation, region="Europe")
    assert len(recs) == 1

    batch = ExtractionBatch(source_corpus="Bone Setter Manual", records=recs)
    atlas = build_atlas_dataset(batch)
    df = atlas["atlas_df"]
    row = df.iloc[0]

    assert row["Preparation"] == "Poultice"
    assert row["Solvent"] == "None (Direct/Topical)"
    assert row["Thermal Regime"] == "Ambient (25°C)"

    # Verify CSV export contains formatted values rather than NaN
    csv_text = export_atlas_csv(df)
    assert "None (Direct/Topical)" in csv_text
    assert "Ambient (25°C)" in csv_text
    assert "NaN" not in csv_text


def test_combined_mechanistic_ranking_artemisia_artemisinin():
    """Verify cold-wrung Artemisia annua highlights Artemisinin (PfATP6) over Scopoletin (IL6)."""
    passage = (
        "Artemisia annua fresh green aerial parts are pounded and wrung cold through silk cloth, "
        "and the raw fresh juice is drunk immediately to break recurring malarial fevers."
    )
    citation = "Ge Hong Herbal (Page 1)"
    recs = extract_offline_from_text(passage, source_citation=citation, region="China")
    assert len(recs) == 1
    r = recs[0]
    assert r.preparation_type == "juice_wringing"

    batch = ExtractionBatch(source_corpus="Ge Hong Herbal", records=[r])
    atlas = build_atlas_dataset(batch)
    df = atlas["atlas_df"]
    row = df.iloc[0]

    # Artemisinin must be selected as Primary Bioactive due to combined mechanistic ranking
    assert row["Primary Bioactive"] == "Artemisinin"
    assert row["Top Target (Gene)"] == "PfATP6"
    assert row["Mechanistic Status"] == "⭐ Validated Path"


