"""Unit tests for DecoctionGraph NLP literature extraction module."""

import json
from unittest.mock import MagicMock
import pytest

from src.schemas import EthnoPreparationRecord, ExtractionBatch
from src.extraction import (
    SYSTEM_PROMPT,
    extract_from_text,
    batch_extract_from_documents,
    mock_extract_hypericum_sample,
    get_llm_client,
)

SAMPLE_PASSAGE = (
    "Hypericum perforatum (St. John's Wort) has long held a revered place in traditional folk medicine "
    "across the Mediterranean and Balkans. The fresh flowering tops and aerial parts are steeped in pure olive oil "
    "and exposed to warm sunlight for forty days until the preparation turns a deep blood-red, yielding a potent oil "
    "applied topically to soothe severe burns, abrasions, and skin wounds. Conversely, for melancholia and nervous "
    "anxiety, the dried leaves and flowers are boiled in fresh spring water for fifteen minutes to prepare a bitter "
    "decoction taken orally twice each day."
)


def test_empty_passage_extraction():
    """Verify empty string returns empty batch without calling LLM."""
    batch = extract_from_text(
        passage="",
        source_citation="Test Citation",
        client=MagicMock(),
    )
    assert isinstance(batch, ExtractionBatch)
    assert len(batch) == 0
    assert batch.source_corpus == "Test Citation"


def test_mock_hypericum_benchmark_extraction():
    """Verify benchmark multi-preparation extraction structure and atomicity."""
    batch = mock_extract_hypericum_sample(
        source_citation="Dioscorides Materia Medica",
        region="Mediterranean",
    )
    assert len(batch) == 2
    assert batch.source_corpus == "Dioscorides Materia Medica"

    rec1, rec2 = batch.records

    # Record 1: Topical Oil Maceration
    assert rec1.verbatim_plant_name == "Hypericum perforatum"
    assert rec1.preparation_type == "oil_maceration"
    assert rec1.solvent_class == "lipid_oil"
    assert rec1.administration_route == "topical"
    assert rec1.verbatim_quote in SAMPLE_PASSAGE

    # Record 2: Oral Aqueous Decoction
    assert rec2.verbatim_plant_name == "Hypericum perforatum"
    assert rec2.preparation_type == "decoction"
    assert rec2.solvent_class == "water"
    assert rec2.thermal_regime == "boiling_prolonged"
    assert rec2.administration_route == "oral"
    assert rec2.verbatim_quote in SAMPLE_PASSAGE


def test_extract_from_text_structured_parse_success():
    """Verify primary extraction path using client.beta.chat.completions.parse."""
    mock_batch = mock_extract_hypericum_sample("Test Source", "Europe")

    mock_client = MagicMock()
    mock_parsed_choice = MagicMock()
    mock_parsed_choice.message.parsed = mock_batch
    mock_client.beta.chat.completions.parse.return_value = MagicMock(choices=[mock_parsed_choice])

    result = extract_from_text(
        passage=SAMPLE_PASSAGE,
        source_citation="Test Source",
        region="Europe",
        client=mock_client,
    )

    assert len(result) == 2
    assert result.source_corpus == "Test Source"
    mock_client.beta.chat.completions.parse.assert_called_once()


def test_extract_from_text_fallback_to_json_mode():
    """Verify fallback to chat.completions.create with JSON mode when parse() fails."""
    mock_batch = mock_extract_hypericum_sample("Test Source", "Europe")
    serialized_batch = mock_batch.model_dump_json()

    mock_client = MagicMock()
    # Simulate parse() failure
    mock_client.beta.chat.completions.parse.side_effect = Exception("Parse failed on provider")

    # Fallback chat.completions.create returns valid JSON
    mock_fallback_choice = MagicMock()
    mock_fallback_choice.message.content = serialized_batch
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_fallback_choice])

    result = extract_from_text(
        passage=SAMPLE_PASSAGE,
        source_citation="Test Source",
        region="Europe",
        client=mock_client,
    )

    assert len(result) == 2
    assert result.source_corpus == "Test Source"
    mock_client.chat.completions.create.assert_called_once()


def test_batch_extract_from_documents():
    """Verify batch processing across multiple documents."""
    mock_client = MagicMock()

    def _mock_parse(**kwargs):
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_extract_hypericum_sample("Herb Book", "Europe")
        return MagicMock(choices=[mock_choice])

    mock_client.beta.chat.completions.parse.side_effect = _mock_parse

    docs = [
        {"text": SAMPLE_PASSAGE, "source": "Source 1", "region": "Mediterranean"},
        {"text": SAMPLE_PASSAGE, "source": "Source 2", "region": "Balkans"},
    ]

    batches = batch_extract_from_documents(docs, max_workers=2, client=mock_client)
    assert len(batches) == 2
    assert batches[0].source_corpus == "Source 1"
    assert batches[1].source_corpus == "Source 2"


def test_get_llm_client_missing_key(monkeypatch):
    """Verify ValueError is raised when no API key is set."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing LLM API key"):
        get_llm_client()
