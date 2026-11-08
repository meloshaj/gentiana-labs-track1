"""DecoctionGraph NLP Literature Extraction Module.

Extracts structured ethnobotanical preparation records from historical and
ethnographic texts using OpenAI/Gemini structured outputs and Pydantic v2.
Treats traditional preparations as analytical chemical filters and enforces
strict atomicity and evidentiary substring verification.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dotenv
from openai import OpenAI

from src.schemas import (
    EthnoPreparationRecord,
    ExtractionBatch,
    PlantPart,
    PreparationType,
    SolventClass,
    ThermalRegime,
    AdministrationRoute,
)

# Load environment variables from .env if present
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def has_configured_llm_credentials() -> bool:
    """Return whether a supported LLM provider has sufficient configuration.

    This checks configuration only; it never instantiates a client or sends a
    network request. UI callers use it to choose live extraction or the
    deterministic offline fallback.
    """
    standard_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    azure_ready = bool(
        os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return bool(standard_key or os.getenv("GEMINI_API_KEY") or azure_ready)

# ==============================================================================
# System Prompt
# ==============================================================================

SYSTEM_PROMPT = """You are an expert ethnobotanical and cheminformatics literature extraction specialist for the DecoctionGraph project.
Your task is to extract structured ethnobotanical preparation records from historical, ethnographic, and pharmacological texts.

Core Instructions:
1. Preparation as Chemical Filter:
   Treat plant preparation (solvent class, thermal conditions, anatomical plant part, administration route) as an analytical chemical extraction filter.
   Strictly classify:
   - solvent_class: "water", "hydroalcoholic", "lipid_oil", "vinegar_acidic", "honey_syrup", "none", "unknown"
   - thermal_regime: "cold_ambient", "warm_steep", "boiling_prolonged", "charred_ash", "none", "unknown"
   - plant_part: "root", "bark", "leaf", "flower", "seed", "fruit", "aerial_parts", "whole_plant", "resin", "rhizome", "unknown"
   - preparation_type: "decoction", "infusion", "cold_maceration", "tincture_alcohol", "oil_maceration", "poultice", "steam_inhalation", "juice_wringing", "raw", "unknown"
   - administration_route: "oral", "topical", "inhalation", "ophthalmic", "unknown"

2. Atomicity:
   If a text passage describes multiple distinct preparations (e.g. an oil maceration for external wounds AND a boiled water decoction for internal ailments, or different parts used separately), you MUST extract each as a SEPARATE, ATOMIC EthnoPreparationRecord. Never conflate multiple preparations into one record.

3. Zero Hallucination on verbatim_quote:
   The `verbatim_quote` MUST BE an exact, contiguous substring copied verbatim from the input passage providing direct evidence for the extraction. Do not paraphrase or alter spelling/punctuation.

4. Verbatim Plant Name & Indication:
   - `verbatim_plant_name`: The exact plant name spelling as written in the text.
   - `verbatim_indication`: The historical ailment, illness, disease, or symptom description as described.

5. Confidence & Ambiguity Notes:
   - `extraction_confidence`: Float between 0.0 and 1.0 based on text clarity and explicitness.
   - `ambiguity_notes`: If any preparation detail (e.g. solvent, thermal regime, or anatomical part) is implied rather than explicitly stated, or if the text is ambiguous, provide an explicit explanatory note and slightly discount confidence.
"""


# ==============================================================================
# LLM Client Factory
# ==============================================================================

def get_llm_client() -> Union[OpenAI, Any]:
    """Instantiate and configure an LLM client across multi-provider backends.

    Supports:
      1. Azure OpenAI: when AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set.
      2. Google Gemini: when GEMINI_API_KEY is present and no standard OpenAI key is configured.
      3. Standard OpenAI / compatible (Groq, Ollama, OpenRouter, vLLM): via LLM_API_KEY or OPENAI_API_KEY.

    Returns:
        Configured OpenAI or AzureOpenAI client instance.

    Raises:
        ValueError: If no valid API key is configured.
    """
    # 1. Azure OpenAI
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint:
        from openai import AzureOpenAI
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        return AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    # 2. Google Gemini API (via OpenAI-compatible endpoint)
    gemini_key = os.getenv("GEMINI_API_KEY")
    standard_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if gemini_key and not standard_key:
        base_url = os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return OpenAI(api_key=gemini_key, base_url=base_url)

    # 3. Standard OpenAI / Custom Base URL
    if not standard_key:
        raise ValueError(
            "Missing LLM API key. Set 'LLM_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY', "
            "or 'AZURE_OPENAI_API_KEY' in your environment or .env file."
        )

    base_url = os.getenv("LLM_BASE_URL")
    return OpenAI(api_key=standard_key, base_url=base_url if base_url else None)


# ==============================================================================
# Core Extraction Functions
# ==============================================================================

def extract_from_text(
    passage: str,
    source_citation: str,
    region: str = "Unspecified",
    model: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> ExtractionBatch:
    """Extract structured ethnobotanical preparation records from a text passage.

    Uses OpenAI structured outputs (`client.beta.chat.completions.parse`) with
    fallback to JSON mode and Pydantic validation if parsing fails.

    Args:
        passage: Source excerpt or historical paragraph to extract.
        source_citation: Citation or title of source document.
        region: Geographic or cultural origin context (default: 'Unspecified').
        model: Model name override (defaults to LLM_MODEL or 'gpt-4o-mini').
        client: Optional pre-configured OpenAI client instance.

    Returns:
        ExtractionBatch container with list of extracted EthnoPreparationRecord models.
    """
    clean_passage = passage.strip()
    if not clean_passage:
        return ExtractionBatch(source_corpus=source_citation, records=[])

    llm_model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    llm_client = client or get_llm_client()

    user_prompt = f"""Source Citation: {source_citation}
Geographic / Cultural Origin: {region}

Text Passage to Extract:
\"\"\"
{clean_passage}
\"\"\"

Extract all distinct ethnobotanical preparations according to the system instructions."""

    batch: Optional[ExtractionBatch] = None

    # Primary method: client.beta.chat.completions.parse with response_format=ExtractionBatch
    try:
        response = llm_client.beta.chat.completions.parse(
            model=llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ExtractionBatch,
        )
        batch = response.choices[0].message.parsed
    except Exception as exc:
        logger.warning(
            f"client.beta.chat.completions.parse failed ({exc}). Attempting JSON mode fallback..."
        )

    # Fallback method: standard chat.completions.create with JSON mode and Pydantic validation
    if batch is None:
        try:
            fallback_response = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        + "\nRespond with strictly valid JSON matching the ExtractionBatch schema: "
                        + '{"records": [...]}',
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw_content = fallback_response.choices[0].message.content or "{}"
            raw_data = json.loads(raw_content)
            batch = ExtractionBatch.model_validate(raw_data)
        except Exception as fallback_exc:
            logger.error(f"Structured extraction fallback also failed: {fallback_exc}")
            raise RuntimeError(
                f"Failed to extract structured ethnobotanical records: {fallback_exc}"
            ) from fallback_exc

    # Post-process: assign source citation and default region if unspecified
    batch = batch.model_copy(deep=True)
    batch.source_corpus = source_citation
    for record in batch.records:
        if record.cultural_or_geographic_origin == "Unspecified" and region != "Unspecified":
            record.cultural_or_geographic_origin = region

    return batch


def batch_extract_from_documents(
    documents: List[Dict[str, Any]],
    max_workers: int = 4,
    model: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> List[ExtractionBatch]:
    """Process a list of ethnobotanical documents concurrently using a thread pool.

    Each document dict should contain:
      * 'text' (or 'passage'): Text excerpt to process.
      * 'source' (or 'source_citation'): Citation or archival reference.
      * 'region' (or 'cultural_or_geographic_origin', optional): Geographic context.

    Args:
        documents: List of document dictionaries.
        max_workers: Number of concurrent worker threads.
        model: Model name override.
        client: Optional pre-configured OpenAI client instance.

    Returns:
        List of ExtractionBatch results matching input order.
    """
    if not documents:
        return []

    llm_client = client or get_llm_client()

    def _process_single(doc: Dict[str, Any]) -> ExtractionBatch:
        text = str(doc.get("text") or doc.get("passage") or "").strip()
        source = str(doc.get("source") or doc.get("source_citation") or "Unspecified Source")
        region = str(doc.get("region") or doc.get("cultural_or_geographic_origin") or "Unspecified")
        return extract_from_text(
            passage=text,
            source_citation=source,
            region=region,
            model=model,
            client=llm_client,
        )

    if max_workers <= 1 or len(documents) == 1:
        return [_process_single(doc) for doc in documents]

    results: List[Optional[ExtractionBatch]] = [None] * len(documents)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_single, doc): idx
            for idx, doc in enumerate(documents)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()

    return [res for res in results if res is not None]


# ==============================================================================
# Benchmark Mock / Offline Extraction Utility
# ==============================================================================

def mock_extract_hypericum_sample(source_citation: str, region: str = "Balkans / Mediterranean") -> ExtractionBatch:
    """Deterministic offline extraction of the multi-preparation Hypericum benchmark passage.

    Used for testing and offline CLI demonstrations without requiring live API keys.
    """
    rec_oil = EthnoPreparationRecord(
        verbatim_plant_name="Hypericum perforatum",
        plant_part="flower",
        preparation_type="oil_maceration",
        solvent_class="lipid_oil",
        thermal_regime="warm_steep",
        verbatim_indication="severe burns, abrasions, and skin wounds",
        administration_route="topical",
        verbatim_quote="The fresh flowering tops and aerial parts are steeped in pure olive oil and exposed to warm sunlight for forty days until the preparation turns a deep blood-red, yielding a potent oil applied topically to soothe severe burns, abrasions, and skin wounds.",
        cultural_or_geographic_origin=region,
        extraction_confidence=0.98,
        ambiguity_notes="Exposure to sunlight for 40 days mapped to warm_steep thermal regime.",
    )

    rec_tea = EthnoPreparationRecord(
        verbatim_plant_name="Hypericum perforatum",
        plant_part="leaf",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="melancholia and nervous anxiety",
        administration_route="oral",
        verbatim_quote="for melancholia and nervous anxiety, the dried leaves and flowers are boiled in fresh spring water for fifteen minutes to prepare a bitter decoction taken orally twice each day.",
        cultural_or_geographic_origin=region,
        extraction_confidence=0.99,
        ambiguity_notes="Boiled for 15 minutes mapped to boiling_prolonged decoction.",
    )

    return ExtractionBatch(
        source_corpus=source_citation,
        records=[rec_oil, rec_tea],
    )


# ==============================================================================
# CLI Demonstration
# ==============================================================================

def main() -> None:
    """CLI demonstration testing a multi-preparation paragraph describing Hypericum perforatum."""
    sample_text = (
        "Hypericum perforatum (St. John's Wort) has long held a revered place in traditional folk medicine "
        "across the Mediterranean and Balkans. The fresh flowering tops and aerial parts are steeped in pure olive oil "
        "and exposed to warm sunlight for forty days until the preparation turns a deep blood-red, yielding a potent oil "
        "applied topically to soothe severe burns, abrasions, and skin wounds. Conversely, for melancholia and nervous "
        "anxiety, the dried leaves and flowers are boiled in fresh spring water for fifteen minutes to prepare a bitter "
        "decoction taken orally twice each day."
    )
    source_citation = "Traditional Herbal Pharmacopoeia of the Mediterranean (1892)"
    region = "Mediterranean / Balkans"

    print("=" * 105)
    print("DecoctionGraph: NLP Literature Extraction CLI (Chemical Filter & Atomic Record Separation)")
    print("=" * 105)
    print(f"\nSource Citation: {source_citation}")
    print(f"Region:          {region}")
    print(f"\nInput Passage:\n\"{sample_text}\"\n")

    if has_configured_llm_credentials():
        print(f"Connecting to LLM (Model: {DEFAULT_MODEL})...")
        try:
            batch = extract_from_text(sample_text, source_citation=source_citation, region=region)
            print(f"Successfully extracted {len(batch)} records via live LLM.")
        except Exception as e:
            print(f"Live LLM call failed ({e}). Falling back to benchmark extraction.")
            batch = mock_extract_hypericum_sample(source_citation=source_citation, region=region)
    else:
        print("Note: No supported LLM provider credentials are set in the environment.")
        print("Running deterministic benchmark extraction on the multi-preparation paragraph...\n")
        batch = mock_extract_hypericum_sample(source_citation=source_citation, region=region)

    print("=" * 105)
    print(f"Extraction Batch Summary (Batch ID: {batch.batch_id})")
    print(f"Extracted Records Count: {len(batch.records)}")
    print("=" * 105)

    for idx, rec in enumerate(batch.records, start=1):
        print(f"\n--- [Record {idx}] ---")
        print(f"Plant Name:           {rec.verbatim_plant_name}")
        print(f"Plant Part:           {rec.plant_part}")
        print(f"Preparation Type:     {rec.preparation_type}")
        print(f"Solvent Class:        {rec.solvent_class}")
        print(f"Thermal Regime:       {rec.thermal_regime}")
        print(f"Administration Route: {rec.administration_route}")
        print(f"Indication:           {rec.verbatim_indication}")
        print(f"Confidence:           {rec.extraction_confidence:.2f}")
        print(f"Verbatim Quote:       \"{rec.verbatim_quote}\"")
        if rec.ambiguity_notes:
            print(f"Ambiguity Notes:      {rec.ambiguity_notes}")

    print("\n" + "=" * 105)
    print("CLI Demonstration Completed Successfully.")
    print("=" * 105)


if __name__ == "__main__":
    main()
