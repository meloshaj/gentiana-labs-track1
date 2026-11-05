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
    TARGET_ACTION_TYPES,
    ACID_BASE_CLASSES,
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
    TargetActionType,
    target_action_type,
    AcidBaseClass,
    acid_base_class,
    EthnoPreparationRecord,
    TaxonResolution,
    TargetMechanismRecord,
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

    expected_target_actions = {
        "agonist", "antagonist", "inhibitor", "modulator", "substrate", "unknown"
    }
    assert set(TARGET_ACTION_TYPES) == expected_target_actions

    expected_acid_base = {
        "neutral", "basic_alkaloid", "weak_acid", "phenolic", "zwitterion"
    }
    assert set(ACID_BASE_CLASSES) == expected_acid_base


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
    assert record.estimated_ph == 7.0
    assert record.duration_minutes == 15.0
    assert record.temperature_celsius == 25.0


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
    """Verify confidence score, pH, duration, and temperature bounds."""
    # confidence bounds
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

    # pH bounds [1.0, 12.0]
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
            extraction_confidence=0.9,
            estimated_ph=0.5,  # < 1.0
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
            extraction_confidence=0.9,
            estimated_ph=13.0,  # > 12.0
        )

    # Duration bounds [1.0, 10080.0]
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
            extraction_confidence=0.9,
            duration_minutes=0.0,  # < 1.0
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
            extraction_confidence=0.9,
            duration_minutes=20000.0,  # > 10080.0
        )

    # Temperature bounds [0.0, 250.0]
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
            extraction_confidence=0.9,
            temperature_celsius=-5.0,  # < 0.0
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
            extraction_confidence=0.9,
            temperature_celsius=300.0,  # > 250.0
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
# Requirement: TargetMechanismRecord Tests
# ==============================================================================

def test_target_mechanism_record_valid():
    """Test TargetMechanismRecord creation, optional fields, and serialization."""
    target = TargetMechanismRecord(
        target_name="Transient receptor potential cation channel subfamily C member 6",
        gene_symbol="TRPC6",
        chembl_target_id="CHEMBL4822",
        action_type="agonist",
        bioactivity_potency_um=0.45,
        therapeutic_pathway="Keratinocyte proliferation & wound healing",
        symptom_relevance_score=0.92,
    )
    assert target.target_name == "Transient receptor potential cation channel subfamily C member 6"
    assert target.gene_symbol == "TRPC6"
    assert target.chembl_target_id == "CHEMBL4822"
    assert target.action_type == "agonist"
    assert target.bioactivity_potency_um == 0.45
    assert target.therapeutic_pathway == "Keratinocyte proliferation & wound healing"
    assert target.symptom_relevance_score == 0.92

    # Test JSON round-trip
    target_json = target.to_json()
    restored = TargetMechanismRecord.model_validate_json(target_json)
    assert restored.gene_symbol == "TRPC6"
    assert restored.action_type == "agonist"


def test_target_mechanism_record_defaults_and_optionals():
    """Verify optional fields and defaults for TargetMechanismRecord."""
    target = TargetMechanismRecord(
        target_name="Cyclooxygenase-2",
        gene_symbol="PTGS2",
        action_type="inhibitor",
        therapeutic_pathway="Anti-inflammatory pathway",
        symptom_relevance_score=0.85,
    )
    assert target.chembl_target_id is None
    assert target.bioactivity_potency_um is None


def test_target_mechanism_record_bounds_and_invalid_action():
    """Verify validation on action_type and symptom_relevance_score bounds [0.0, 1.0]."""
    # Invalid action_type
    with pytest.raises(ValidationError):
        TargetMechanismRecord(
            target_name="Target",
            gene_symbol="TGT",
            action_type="invalid_action",  # Not in TargetActionType
            therapeutic_pathway="Pathway",
            symptom_relevance_score=0.5,
        )

    # symptom_relevance_score > 1.0
    with pytest.raises(ValidationError):
        TargetMechanismRecord(
            target_name="Target",
            gene_symbol="TGT",
            action_type="inhibitor",
            therapeutic_pathway="Pathway",
            symptom_relevance_score=1.2,
        )

    # symptom_relevance_score < 0.0
    with pytest.raises(ValidationError):
        TargetMechanismRecord(
            target_name="Target",
            gene_symbol="TGT",
            action_type="inhibitor",
            therapeutic_pathway="Pathway",
            symptom_relevance_score=-0.1,
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


def test_phytochemical_record_new_fields_defaults():
    """Verify defaults for newly added organ, pH/logD, thermal, and target mechanism fields."""
    phyto = PhytochemicalRecord(
        compound_name="Quercetin",
        smiles="O=C1C(O)=C(C2=CC=C(O)C(O)=C2)OC3=CC(O)=CC(O)=C13",
        inchi_key="REFJWTPEDVJJIY-UHFFFAOYSA-N",
        molecular_formula="C15H10O7",
        mw=302.24,
        clogp=1.48,
        tpsa=131.36,
        h_donors=5,
        h_acceptors=7,
        rotatable_bonds=1,
        is_thermolabile=False,
        is_volatile=False,
        extraction_plausibility_score=0.88,
        extraction_verdict="high",
    )
    assert phyto.documented_organs == []
    assert phyto.organ_match_factor == 1.0
    assert phyto.acid_base_class == "neutral"
    assert phyto.estimated_pka is None
    assert phyto.logd_at_recipe_ph == 0.0
    assert phyto.thermal_half_life_min_at_100c is None
    assert phyto.thermal_survival_fraction == 1.0
    assert phyto.known_targets == []
    assert phyto.mechanistic_alignment_score == 0.0


def test_phytochemical_record_new_fields_explicit():
    """Verify explicit configuration of organ, pKa/logD, Arrhenius kinetics, and targets."""
    target = TargetMechanismRecord(
        target_name="Transient receptor potential cation channel subfamily C member 6",
        gene_symbol="TRPC6",
        chembl_target_id="CHEMBL4822",
        action_type="agonist",
        bioactivity_potency_um=0.15,
        therapeutic_pathway="Keratinocyte proliferation & wound healing",
        symptom_relevance_score=0.95,
    )
    phyto = PhytochemicalRecord(
        compound_name="Hyperforin",
        smiles="CC(=CCC1(C(=O)C2(C(=O)C(C(=O)C1(C2CC=C(C)C)CC=C(C)C)C(=O)C(C)C)CC=C(C)C)C)C",
        inchi_key="CRDYGABRMFGNOF-UHFFFAOYSA-N",
        molecular_formula="C35H52O4",
        mw=536.79,
        clogp=6.27,
        tpsa=74.60,
        h_donors=1,
        h_acceptors=4,
        rotatable_bonds=8,
        is_thermolabile=True,
        is_volatile=False,
        extraction_plausibility_score=0.91,
        extraction_verdict="high",
        documented_organs=["flower", "aerial_parts"],
        organ_match_factor=1.0,
        acid_base_class="weak_acid",
        estimated_pka=4.8,
        logd_at_recipe_ph=4.5,
        thermal_half_life_min_at_100c=18.5,
        thermal_survival_fraction=0.72,
        known_targets=[target],
        mechanistic_alignment_score=0.95,
    )
    assert phyto.documented_organs == ["flower", "aerial_parts"]
    assert phyto.organ_match_factor == 1.0
    assert phyto.acid_base_class == "weak_acid"
    assert phyto.estimated_pka == 4.8
    assert phyto.logd_at_recipe_ph == 4.5
    assert phyto.thermal_half_life_min_at_100c == 18.5
    assert phyto.thermal_survival_fraction == 0.72
    assert len(phyto.known_targets) == 1
    assert phyto.known_targets[0].gene_symbol == "TRPC6"
    assert phyto.mechanistic_alignment_score == 0.95


def test_phytochemical_record_new_fields_bounds():
    """Verify validation bounds on organ_match_factor, thermal_survival_fraction, and scores."""
    base_kw = dict(
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
        extraction_verdict="high",
    )

    # organ_match_factor bounds [0.0, 1.0]
    with pytest.raises(ValidationError):
        PhytochemicalRecord(**base_kw, organ_match_factor=1.5)
    with pytest.raises(ValidationError):
        PhytochemicalRecord(**base_kw, organ_match_factor=-0.1)

    # thermal_survival_fraction bounds [0.0, 1.0]
    with pytest.raises(ValidationError):
        PhytochemicalRecord(**base_kw, thermal_survival_fraction=1.2)
    with pytest.raises(ValidationError):
        PhytochemicalRecord(**base_kw, thermal_survival_fraction=-0.05)

    # mechanistic_alignment_score bounds [0.0, 1.0]
    with pytest.raises(ValidationError):
        PhytochemicalRecord(**base_kw, mechanistic_alignment_score=1.1)
    with pytest.raises(ValidationError):
        PhytochemicalRecord(**base_kw, mechanistic_alignment_score=-0.2)

    # acid_base_class literal validation
    with pytest.raises(ValidationError):
        PhytochemicalRecord(**base_kw, acid_base_class="invalid_class")


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

    assert graph.overall_mechanistic_plausibility == 0.0

    # Test explicit overall_mechanistic_plausibility and bounds
    graph_scored = EnrichedPlantGraph(
        preparation_record=prep,
        taxon_resolution=taxon,
        candidate_phytochemicals=[phyto_high],
        overall_mechanistic_plausibility=0.88,
    )
    assert graph_scored.overall_mechanistic_plausibility == 0.88

    with pytest.raises(ValidationError):
        EnrichedPlantGraph(
            preparation_record=prep,
            overall_mechanistic_plausibility=1.5,  # > 1.0
        )
    with pytest.raises(ValidationError):
        EnrichedPlantGraph(
            preparation_record=prep,
            overall_mechanistic_plausibility=-0.1,  # < 0.0
        )

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
        "TargetMechanismRecord",
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
    assert "estimated_ph" in props
    assert "duration_minutes" in props
    assert "temperature_celsius" in props
    assert props["extraction_confidence"].get("maximum") == 1.0
    assert props["extraction_confidence"].get("minimum") == 0.0
    assert props["estimated_ph"].get("maximum") == 12.0
    assert props["estimated_ph"].get("minimum") == 1.0

    # Test TargetMechanismRecord schema specific fields
    target_schema = schemas["TargetMechanismRecord"]
    t_props = target_schema["properties"]
    assert "target_name" in t_props
    assert "gene_symbol" in t_props
    assert "action_type" in t_props
    assert "symptom_relevance_score" in t_props

    # Test PhytochemicalRecord schema specific fields
    phyto_schema = schemas["PhytochemicalRecord"]
    p_props = phyto_schema["properties"]
    assert "documented_organs" in p_props
    assert "organ_match_factor" in p_props
    assert "acid_base_class" in p_props
    assert "estimated_pka" in p_props
    assert "logd_at_recipe_ph" in p_props
    assert "thermal_half_life_min_at_100c" in p_props
    assert "thermal_survival_fraction" in p_props
    assert "known_targets" in p_props
    assert "mechanistic_alignment_score" in p_props
    assert props["extraction_confidence"].get("maximum") == 1.0
    assert props["extraction_confidence"].get("minimum") == 0.0

