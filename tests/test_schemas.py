"""Unit tests for DecoctionGraph Pydantic v2 schemas."""

import json
import pytest
from pydantic import ValidationError

from src.schemas import (
    PLANT_PARTS,
    PREPARATION_TYPES,
    SOLVENT_CLASSES,
    THERMAL_REGIMES,
    ADMINISTRATION_ROUTES,
    EXTRACTION_VERDICTS,
    PlantPart,
    plant_part,
    PreparationType,
    preparation_type,
    SolventClass,
    solvent_class,
    ThermalRegime,
    thermal_regime,
    AdministrationRoute,
    administration_route,
    ExtractionVerdict,
    extraction_verdict,
    EthnoPreparationRecord,
    TaxonResolution,
    PhytochemicalRecord,
    ExtractionBatch,
    EnrichedPlantGraph,
    get_module_json_schemas,
)


# ==============================================================================
# Requirement 1: Strict Literal Types Tests
# ==============================================================================

def test_literal_values():
    """Verify exact literal memberships requested in specification."""
    expected_plant_parts = {
        "root", "bark", "leaf", "flower", "seed", "fruit",
        "aerial_parts", "whole_plant", "resin", "rhizome", "unknown"
    }
    assert set(PLANT_PARTS) == expected_plant_parts

    expected_prep_types = {
        "decoction", "infusion", "cold_maceration", "tincture_alcohol",
        "oil_maceration", "poultice", "steam_inhalation", "juice_wringing",
        "raw", "unknown"
    }
    assert set(PREPARATION_TYPES) == expected_prep_types

    expected_solvents = {
        "water", "hydroalcoholic", "lipid_oil", "vinegar_acidic",
        "honey_syrup", "none", "unknown"
    }
    assert set(SOLVENT_CLASSES) == expected_solvents

    expected_thermal = {
        "cold_ambient", "warm_steep", "boiling_prolonged", "charred_ash",
        "none", "unknown"
    }
    assert set(THERMAL_REGIMES) == expected_thermal

    expected_routes = {
        "oral", "topical", "inhalation", "ophthalmic", "unknown"
    }
    assert set(ADMINISTRATION_ROUTES) == expected_routes

    expected_verdicts = {"high", "moderate", "negligible", "excluded"}
    assert set(EXTRACTION_VERDICTS) == expected_verdicts


# ==============================================================================
# Requirement 2: EthnoPreparationRecord Tests
# ==============================================================================

def test_ethno_preparation_record_valid():
    """Test valid EthnoPreparationRecord construction and serialization."""
    record = EthnoPreparationRecord(
        verbatim_plant_name="Gentiana lutea",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="digestive complaints and intermittent fever",
        administration_route="oral",
        verbatim_quote="Take the roots of Gentian, boil them in pure spring water for one hour and drink upon waking.",
        cultural_or_geographic_origin="Alpine European",
        extraction_confidence=0.95,
        ambiguity_notes="Species name confirmed in text; no synonym ambiguity.",
    )

    assert record.verbatim_plant_name == "Gentiana lutea"
    assert record.plant_part == "root"
    assert record.preparation_type == "decoction"
    assert record.solvent_class == "water"
    assert record.thermal_regime == "boiling_prolonged"
    assert record.verbatim_indication == "digestive complaints and intermittent fever"
    assert record.administration_route == "oral"
    assert record.cultural_or_geographic_origin == "Alpine European"
    assert record.extraction_confidence == 0.95

    # Test JSON round-trip
    json_data = record.to_json()
    roundtrip = EthnoPreparationRecord.model_validate_json(json_data)
    assert roundtrip.verbatim_plant_name == record.verbatim_plant_name
    assert roundtrip.extraction_confidence == record.extraction_confidence


def test_ethno_preparation_record_defaults():
    """Verify default values when optional fields are omitted."""
    record = EthnoPreparationRecord(
        verbatim_plant_name="Mentha piperita",
        plant_part="leaf",
        preparation_type="infusion",
        solvent_class="water",
        thermal_regime="warm_steep",
        verbatim_indication="headache and nausea",
        administration_route="oral",
        verbatim_quote="Infuse the dried peppermint leaves in warm water.",
        extraction_confidence=0.88,
    )
    assert record.cultural_or_geographic_origin == "Unspecified"
    assert record.ambiguity_notes is None


def test_ethno_preparation_record_invalid_literal():
    """Verify that invalid Literal choices raise ValidationError."""
    with pytest.raises(ValidationError):
        EthnoPreparationRecord(
            verbatim_plant_name="Gentiana lutea",
            plant_part="invalid_part",  # Not in PlantPart
            preparation_type="decoction",
            solvent_class="water",
            thermal_regime="boiling_prolonged",
            verbatim_indication="fever",
            administration_route="oral",
            verbatim_quote="Boil roots.",
            extraction_confidence=0.9,
        )


def test_ethno_preparation_record_bounds():
    """Verify confidence score bounds [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        EthnoPreparationRecord(
            verbatim_plant_name="Gentiana lutea",
            plant_part="root",
            preparation_type="decoction",
            solvent_class="water",
            thermal_regime="boiling_prolonged",
            verbatim_indication="fever",
            administration_route="oral",
            verbatim_quote="Boil roots.",
            extraction_confidence=1.5,  # > 1.0
        )

    with pytest.raises(ValidationError):
        EthnoPreparationRecord(
            verbatim_plant_name="Gentiana lutea",
            plant_part="root",
            preparation_type="decoction",
            solvent_class="water",
            thermal_regime="boiling_prolonged",
            verbatim_indication="fever",
            administration_route="oral",
            verbatim_quote="Boil roots.",
            extraction_confidence=-0.1,  # < 0.0
        )


# ==============================================================================
# Requirement 3: TaxonResolution Tests
# ==============================================================================

def test_taxon_resolution_valid():
    """Test TaxonResolution construction with GBIF backbone keys."""
    taxon = TaxonResolution(
        queried_name="Gentiana lutea",
        accepted_scientific_name="Gentiana lutea L.",
        taxon_rank="species",
        taxonomic_status="ACCEPTED",
        gbif_usage_key=3170154,
        confidence=0.98,
        match_type="EXACT",
        is_resolved=True,
    )
    assert taxon.gbif_usage_key == 3170154
    assert taxon.is_resolved is True
    assert taxon.confidence == 0.98

    # Optional gbif_usage_key
    taxon_unmatched = TaxonResolution(
        queried_name="Mythical herb",
        accepted_scientific_name="Unknown plant",
        taxon_rank="unknown",
        taxonomic_status="DOUBTFUL",
        confidence=0.0,
        match_type="NONE",
        is_resolved=False,
    )
    assert taxon_unmatched.gbif_usage_key is None
    assert taxon_unmatched.is_resolved is False


def test_taxon_resolution_confidence_bounds():
    """Verify confidence score bounds for TaxonResolution."""
    with pytest.raises(ValidationError):
        TaxonResolution(
            queried_name="Plant",
            accepted_scientific_name="Plant sp.",
            taxon_rank="species",
            taxonomic_status="ACCEPTED",
            confidence=1.1,  # Invalid
            match_type="EXACT",
            is_resolved=True,
        )


# ==============================================================================
# Requirement 4: PhytochemicalRecord Tests
# ==============================================================================

def test_phytochemical_record_valid():
    """Test PhytochemicalRecord creation, InChIKey normalization, and serialization."""
    phyto = PhytochemicalRecord(
        compound_name="Gentiopicroside",
        smiles="C1=COC2(C1C(C(=O)O2)C=C)OC3C(C(C(C(O3)CO)O)O)O",
        inchi_key="bwdnngsfwzupex-uhfffaoysa-n",  # Test lowercase normalization
        molecular_formula="C16H20O9",
        mw=356.32,
        clogp=-0.72,
        tpsa=135.29,
        h_donors=4,
        h_acceptors=9,
        rotatable_bonds=3,
        is_thermolabile=False,
        is_volatile=False,
        toxicity_alerts=[],
        extraction_plausibility_score=0.92,
        extraction_verdict="high",
        provenance_doi="10.1016/j.phytochem.2020.112456",
    )
    assert phyto.inchi_key == "BWDNNGSFWZUPEX-UHFFFAOYSA-N"  # Uppercase normalized
    assert phyto.extraction_verdict == "high"
    assert phyto.extraction_plausibility_score == 0.92
    assert phyto.mw == 356.32
    assert phyto.provenance_doi == "10.1016/j.phytochem.2020.112456"


def test_phytochemical_record_invalid_verdict():
    """Verify invalid extraction_verdict fails validation."""
    with pytest.raises(ValidationError):
        PhytochemicalRecord(
            compound_name="Test",
            smiles="CC",
            inchi_key="ABC",
            molecular_formula="C2H6",
            mw=30.07,
            clogp=1.0,
            tpsa=0.0,
            h_donors=0,
            h_acceptors=0,
            rotatable_bonds=0,
            is_thermolabile=False,
            is_volatile=False,
            extraction_plausibility_score=0.5,
            extraction_verdict="super_high",  # Invalid
        )


# ==============================================================================
# Requirement 5: Container Models Tests
# ==============================================================================

def test_extraction_batch():
    """Test ExtractionBatch container behaviors, indexing, and iteration."""
    rec1 = EthnoPreparationRecord(
        verbatim_plant_name="Gentiana lutea",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="fever",
        administration_route="oral",
        verbatim_quote="Boil roots.",
        extraction_confidence=0.95,
    )
    rec2 = EthnoPreparationRecord(
        verbatim_plant_name="Salix alba",
        plant_part="bark",
        preparation_type="infusion",
        solvent_class="water",
        thermal_regime="warm_steep",
        verbatim_indication="rheumatic pain",
        administration_route="oral",
        verbatim_quote="Steep willow bark in hot water.",
        extraction_confidence=0.90,
    )

    batch = ExtractionBatch(
        source_corpus="Pharmacopoeia Londinensis",
        records=[rec1],
    )
    assert len(batch) == 1
    assert batch.total_records == 1
    assert batch[0].verbatim_plant_name == "Gentiana lutea"

    batch.add_record(rec2)
    assert len(batch) == 2
    assert batch.total_records == 2
    assert [r.verbatim_plant_name for r in batch] == ["Gentiana lutea", "Salix alba"]

    # Verify JSON round-trip
    batch_json = batch.to_json()
    restored = ExtractionBatch.model_validate_json(batch_json)
    assert len(restored) == 2
    assert restored.source_corpus == "Pharmacopoeia Londinensis"


def test_enriched_plant_graph():
    """Test EnrichedPlantGraph integration, properties, and compound filtering."""
    prep = EthnoPreparationRecord(
        verbatim_plant_name="Gentiana lutea",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="stomach weakness",
        administration_route="oral",
        verbatim_quote="Decoct gentian root in boiling water.",
        extraction_confidence=0.96,
    )
    taxon = TaxonResolution(
        queried_name="Gentiana lutea",
        accepted_scientific_name="Gentiana lutea L.",
        taxon_rank="species",
        taxonomic_status="ACCEPTED",
        gbif_usage_key=3170154,
        confidence=0.99,
        match_type="EXACT",
        is_resolved=True,
    )
    phyto_high = PhytochemicalRecord(
        compound_name="Gentiopicroside",
        smiles="C1=COC2(C1C(C(=O)O2)C=C)OC3C(C(C(C(O3)CO)O)O)O",
        inchi_key="BWDNNGSFWZUPEX-UHFFFAOYSA-N",
        molecular_formula="C16H20O9",
        mw=356.32,
        clogp=-0.72,
        tpsa=135.29,
        h_donors=4,
        h_acceptors=9,
        rotatable_bonds=3,
        is_thermolabile=False,
        is_volatile=False,
        toxicity_alerts=[],
        extraction_plausibility_score=0.95,
        extraction_verdict="high",
    )
    phyto_thermolabile = PhytochemicalRecord(
        compound_name="Heat-Sensitive Glycoside",
        smiles="OCC1OC(OC2...)C1",
        inchi_key="AAAAAAAAAAAAAAAAAAAAAAAAAA-N",
        molecular_formula="C20H30O10",
        mw=430.45,
        clogp=0.5,
        tpsa=140.0,
        h_donors=5,
        h_acceptors=10,
        rotatable_bonds=4,
        is_thermolabile=True,
        is_volatile=False,
        toxicity_alerts=[],
        extraction_plausibility_score=0.35,
        extraction_verdict="negligible",
    )

    graph = EnrichedPlantGraph(
        preparation_record=prep,
        taxon_resolution=taxon,
        candidate_phytochemicals=[phyto_high, phyto_thermolabile],
        pipeline_metadata={"model": "decoction-graph-v1", "run_id": "test-001"},
    )

    assert graph.is_taxonomically_resolved is True
    assert len(graph.candidate_phytochemicals) == 2
    assert len(graph.high_plausibility_compounds) == 1
    assert graph.high_plausibility_compounds[0].compound_name == "Gentiopicroside"

    # Filter excluding thermolabile
    stable_compounds = graph.filter_phytochemicals(exclude_thermolabile=True)
    assert len(stable_compounds) == 1
    assert stable_compounds[0].compound_name == "Gentiopicroside"

    # Filter by minimum score
    scored_compounds = graph.filter_phytochemicals(min_score=0.5)
    assert len(scored_compounds) == 1
    assert scored_compounds[0].compound_name == "Gentiopicroside"

    # Serialization round-trip
    graph_json = graph.to_json()
    graph_restored = EnrichedPlantGraph.model_validate_json(graph_json)
    assert graph_restored.preparation_record.verbatim_plant_name == "Gentiana lutea"
    assert graph_restored.taxon_resolution.accepted_scientific_name == "Gentiana lutea L."
    assert len(graph_restored.candidate_phytochemicals) == 2


# ==============================================================================
# Requirement 6: JSON Schema Serialization Tests
# ==============================================================================

def test_json_schema_export():
    """Verify that all models generate valid JSON schema specifications."""
    schemas = get_module_json_schemas()

    expected_models = {
        "EthnoPreparationRecord",
        "TaxonResolution",
        "PhytochemicalRecord",
        "ExtractionBatch",
        "EnrichedPlantGraph",
    }
    assert set(schemas.keys()) == expected_models

    for model_name, schema in schemas.items():
        assert "properties" in schema or "$defs" in schema
        assert schema.get("title") == model_name
        # Test serializability of schema itself
        serialized = json.dumps(schema)
        assert len(serialized) > 50

    # Test EthnoPreparationRecord schema specific fields
    ethno_schema = schemas["EthnoPreparationRecord"]
    props = ethno_schema["properties"]
    assert "verbatim_plant_name" in props
    assert "plant_part" in props
    assert "preparation_type" in props
    assert "solvent_class" in props
    assert "thermal_regime" in props
    assert "administration_route" in props
    assert "extraction_confidence" in props
    assert props["extraction_confidence"].get("maximum") == 1.0
    assert props["extraction_confidence"].get("minimum") == 0.0

