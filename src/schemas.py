"""DecoctionGraph Core Data Schemas.

Production-grade Pydantic v2 models and strict Literal domain types for
an ethnobotanical text-mining, taxonomic resolution, and cheminformatics
knowledge pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, get_args
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ==============================================================================
# Strict Domain Literal Types & Sets
# ==============================================================================

PlantPart = Literal[
    "root",
    "bark",
    "leaf",
    "flower",
    "seed",
    "fruit",
    "aerial_parts",
    "whole_plant",
    "resin",
    "rhizome",
    "unknown",
]
"""Permitted anatomical parts of the plant recorded in ethnobotanical texts."""
plant_part = PlantPart  # Alias conforming to lowercase type notation

PreparationType = Literal[
    "decoction",
    "infusion",
    "cold_maceration",
    "tincture_alcohol",
    "oil_maceration",
    "poultice",
    "steam_inhalation",
    "juice_wringing",
    "raw",
    "unknown",
]
"""Permitted traditional extraction procedures and preparation modalities."""
preparation_type = PreparationType  # Alias conforming to lowercase type notation

SolventClass = Literal[
    "water",
    "hydroalcoholic",
    "lipid_oil",
    "vinegar_acidic",
    "honey_syrup",
    "none",
    "unknown",
]
"""Permitted solvent or vehicle classes used in ethnobotanical preparations."""
solvent_class = SolventClass  # Alias conforming to lowercase type notation

ThermalRegime = Literal[
    "cold_ambient",
    "warm_steep",
    "boiling_prolonged",
    "charred_ash",
    "none",
    "unknown",
]
"""Thermal regimes applied to the plant material during preparation."""
thermal_regime = ThermalRegime  # Alias conforming to lowercase type notation

AdministrationRoute = Literal[
    "oral",
    "topical",
    "inhalation",
    "ophthalmic",
    "unknown",
]
"""Anatomical route of delivery or administration for the prepared remedy."""
administration_route = AdministrationRoute  # Alias conforming to lowercase type notation

ExtractionVerdict = Literal[
    "high",
    "moderate",
    "negligible",
    "excluded",
]
"""Physicochemical plausibility verdict regarding compound extractability."""
extraction_verdict = ExtractionVerdict  # Alias conforming to lowercase type notation

# Runtime validation tuples / introspection sets
PLANT_PARTS: Tuple[str, ...] = get_args(PlantPart)
PREPARATION_TYPES: Tuple[str, ...] = get_args(PreparationType)
SOLVENT_CLASSES: Tuple[str, ...] = get_args(SolventClass)
THERMAL_REGIMES: Tuple[str, ...] = get_args(ThermalRegime)
ADMINISTRATION_ROUTES: Tuple[str, ...] = get_args(AdministrationRoute)
EXTRACTION_VERDICTS: Tuple[str, ...] = get_args(ExtractionVerdict)


# ==============================================================================
# Base Schema Configuration
# ==============================================================================

class DecoctionBaseModel(BaseModel):
    """Base model applying strict validation, whitespace stripping, and JSON schema hooks."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
        extra="ignore",
        populate_by_name=True,
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize model instance to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model instance to standard Python dictionary."""
        return self.model_dump(mode="python")


# ==============================================================================
# Domain Models
# ==============================================================================

class EthnoPreparationRecord(DecoctionBaseModel):
    """Structured extraction of an ethnobotanical preparation from historical or ethnographic literature.

    Captures verbatim botanical descriptions, preparation parameters (plant part,
    solvent, thermal regime), indication, and supporting textual citations.
    """

    verbatim_plant_name: str = Field(
        ...,
        min_length=1,
        description="Exact botanical or vernacular plant name as spelled in the source historical/ethnographic text.",
        examples=["Gentiana lutea", "Yellow Gentian", "Wormwood"],
    )
    plant_part: PlantPart = Field(
        ...,
        description="Anatomical plant part utilized in the preparation.",
        examples=["root", "leaf", "rhizome"],
    )
    preparation_type: PreparationType = Field(
        ...,
        description="Specific preparation or extraction technique reported (e.g., decoction, infusion, tincture).",
        examples=["decoction", "infusion", "tincture_alcohol"],
    )
    solvent_class: SolventClass = Field(
        ...,
        description="Chemical class of the extraction vehicle or solvent employed.",
        examples=["water", "hydroalcoholic", "lipid_oil"],
    )
    thermal_regime: ThermalRegime = Field(
        ...,
        description="Thermal energy regime applied during extraction (e.g., boiling_prolonged, cold_ambient).",
        examples=["boiling_prolonged", "warm_steep", "cold_ambient"],
    )
    verbatim_indication: str = Field(
        ...,
        min_length=1,
        description="Historical ailment, illness, disease, or symptom description explicitly mentioned in the text.",
        examples=["tertian fever", "stomach weaknesses", "wound cleansing"],
    )
    administration_route: AdministrationRoute = Field(
        ...,
        description="Anatomical route of administration for the preparation.",
        examples=["oral", "topical"],
    )
    verbatim_quote: str = Field(
        ...,
        min_length=1,
        description="Exact verbatim sentence or paragraph excerpt from the historical corpus supporting this extraction.",
        examples=["Boile the roots of Gentian in fountain water for two hours to cure the ague."],
    )
    cultural_or_geographic_origin: Optional[str] = Field(
        default="Unspecified",
        description="Ethnolinguistic culture, indigenous group, or geographic region where this use was recorded.",
        examples=["European Renaissance", "Traditional Chinese Medicine", "Alpine Herbalism", "Unspecified"],
    )
    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="NLP pipeline or annotator extraction confidence score bounded strictly between 0.0 and 1.0.",
        examples=[0.95],
    )
    ambiguity_notes: Optional[str] = Field(
        default=None,
        description="Optional notes documenting orthographic, botanical, linguistic, or methodological ambiguities.",
        examples=["Name may refer to Gentiana purpurea in regional dialects."],
    )


class TaxonResolution(DecoctionBaseModel):
    """Taxonomic entity resolution resolving a verbatim or queried botanical name to accepted taxonomy.

    Standardized against international biodiversity backbones (such as GBIF Backbone Taxonomy
    or World Flora Online).
    """

    queried_name: str = Field(
        ...,
        min_length=1,
        description="Raw input botanical or vernacular name submitted for taxonomic resolution.",
        examples=["Gentiana lutea L.", "Yellow Gentian"],
    )
    accepted_scientific_name: str = Field(
        ...,
        min_length=1,
        description="Resolved, accepted binomial or trinomial scientific name according to taxonomic backbone.",
        examples=["Gentiana lutea L."],
    )
    taxon_rank: str = Field(
        ...,
        min_length=1,
        description="Taxonomic rank of the resolved entity (e.g., 'species', 'genus', 'subspecies').",
        examples=["species", "genus"],
    )
    taxonomic_status: str = Field(
        ...,
        min_length=1,
        description="Nomenclatural or taxonomic status of the resolved taxon (e.g., 'ACCEPTED', 'SYNONYM', 'DOUBTFUL').",
        examples=["ACCEPTED", "SYNONYM"],
    )
    gbif_usage_key: Optional[int] = Field(
        default=None,
        ge=1,
        description="Unique identifier (usageKey / taxonKey) in the GBIF Backbone Taxonomy, if matched.",
        examples=[3170154],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Taxonomic alignment confidence score [0.0, 1.0] from fuzzy/canonical matching algorithms.",
        examples=[0.98],
    )
    match_type: str = Field(
        ...,
        min_length=1,
        description="Strategy or algorithm applied by the taxonomic backbone matcher (e.g., 'EXACT', 'FUZZY', 'HIGHERRANK', 'NONE').",
        examples=["EXACT", "FUZZY"],
    )
    is_resolved: bool = Field(
        ...,
        description="Boolean flag indicating whether the queried name was successfully aligned to an accepted scientific taxon.",
        examples=[True],
    )


class PhytochemicalRecord(DecoctionBaseModel):
    """Cheminformatics and physicochemical profile of a secondary metabolite or phytochemical constituent.

    Contains structural descriptors (SMILES, InChIKey), calculated Lipinski/physicochemical
    properties, stability markers, and extraction plausibility scores under ethnobotanical regimes.
    """

    compound_name: str = Field(
        ...,
        min_length=1,
        description="Common, trivial, or IUPAC name of the phytochemical constituent.",
        examples=["Gentiopicroside", "Amarogentin"],
    )
    smiles: str = Field(
        ...,
        min_length=1,
        description="Canonical or isomeric SMILES (Simplified Molecular Input Line Entry System) string.",
        examples=["C1=COC2(C1C(C(=O)O2)C=C)OC3C(C(C(C(O3)CO)O)O)O"],
    )
    inchi_key: str = Field(
        ...,
        min_length=1,
        description="Standard hashed 27-character International Chemical Identifier Key (InChIKey).",
        examples=["BWDNNGSFWZUPEX-UHFFFAOYSA-N"],
    )
    molecular_formula: str = Field(
        ...,
        min_length=1,
        description="Hill-system molecular formula of the compound.",
        examples=["C16H20O9"],
    )
    mw: float = Field(
        ...,
        ge=0.0,
        description="Molecular weight in Daltons (g/mol).",
        examples=[356.32],
    )
    clogp: float = Field(
        ...,
        description="Calculated octanol-water partition coefficient (logP), reflecting lipophilicity.",
        examples=[-0.72],
    )
    tpsa: float = Field(
        ...,
        ge=0.0,
        description="Topological Polar Surface Area in square Angstroms (Å²).",
        examples=[135.29],
    )
    h_donors: int = Field(
        ...,
        ge=0,
        description="Count of Lipinski hydrogen bond donors (-OH and -NH groups).",
        examples=[4],
    )
    h_acceptors: int = Field(
        ...,
        ge=0,
        description="Count of Lipinski hydrogen bond acceptors (O and N atoms).",
        examples=[9],
    )
    rotatable_bonds: int = Field(
        ...,
        ge=0,
        description="Count of non-terminal single rotatable bonds.",
        examples=[3],
    )
    is_thermolabile: bool = Field(
        ...,
        description="Flag indicating compound susceptibility to thermal degradation or hydrolysis during boiling/heat.",
        examples=[False],
    )
    is_volatile: bool = Field(
        ...,
        description="Flag indicating compound volatility (e.g. monoterpenes) lost or collected in open/closed boiling.",
        examples=[False],
    )
    toxicity_alerts: List[str] = Field(
        default_factory=list,
        description="List of detected toxicological structural alerts (e.g., pyrrolizidine alkaloid core, PAINS alerts).",
        examples=[[]],
    )
    extraction_plausibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Physicochemically derived probability score [0.0, 1.0] that this molecule partitions into the preparation solvent and survives the thermal regime.",
        examples=[0.89],
    )
    extraction_verdict: ExtractionVerdict = Field(
        ...,
        description="Categorical extractability verdict under the specific traditional preparation regime.",
        examples=["high"],
    )
    provenance_doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier (DOI) or literature reference grounding this compound's presence in the botanical taxon.",
        examples=["10.1016/j.phytochem.2020.112456"],
    )

    @field_validator("inchi_key")
    @classmethod
    def normalize_inchi_key(cls, v: str) -> str:
        """Strip whitespace and enforce uppercase representation for InChIKey."""
        return v.strip().upper()


# ==============================================================================
# Container Models
# ==============================================================================

class ExtractionBatch(DecoctionBaseModel):
    """Container model representing a collection of ethnobotanical preparation records.

    Provides collection indexing, batch metadata, and serialization utilities for high-throughput
    NLP literature extraction pipelines.
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4) for this extraction batch.",
    )
    source_corpus: Optional[str] = Field(
        default=None,
        description="Title, archival identifier, or citation of the historical text corpus processed.",
        examples=["Culpeper's Complete Herbal (1653)"],
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp marking when this batch was parsed and instantiated.",
    )
    records: List[EthnoPreparationRecord] = Field(
        default_factory=list,
        description="Collection of ethnobotanical preparation records extracted from the text.",
    )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index: int) -> EthnoPreparationRecord:
        return self.records[index]

    def add_record(self, record: EthnoPreparationRecord) -> None:
        """Append an ethnobotanical preparation record to the batch."""
        self.records.append(record)

    @property
    def total_records(self) -> int:
        """Return total number of extracted records in batch."""
        return len(self.records)


class EnrichedPlantGraph(DecoctionBaseModel):
    """Holistic knowledge graph node linking an ethnobotanical preparation to taxonomic resolution and candidate phytochemicals.

    Represents the unified tripartite graph:
    [Ethnobotanical Record] <--> [Standardized Taxon] <--> [Physicochemically Filtered Phytochemicals]
    """

    graph_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4) for this enriched ethnobotanical plant graph instance.",
    )
    preparation_record: EthnoPreparationRecord = Field(
        ...,
        description="The primary historical or ethnographic preparation record extracted from literature.",
    )
    taxon_resolution: Optional[TaxonResolution] = Field(
        default=None,
        description="Taxonomic backbone resolution resolving the plant entity.",
    )
    candidate_phytochemicals: List[PhytochemicalRecord] = Field(
        default_factory=list,
        description="Candidate phytochemical constituents linked to the plant taxon, scored for extraction plausibility.",
    )
    pipeline_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary pipeline execution metadata, model telemetry, or filter threshold parameters.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this enriched graph node was assembled.",
    )

    @property
    def is_taxonomically_resolved(self) -> bool:
        """Check whether the underlying plant has been validated and resolved to an accepted scientific taxon."""
        return self.taxon_resolution is not None and self.taxon_resolution.is_resolved

    def filter_phytochemicals(
        self,
        min_score: float = 0.0,
        verdicts: Optional[Sequence[ExtractionVerdict]] = None,
        exclude_thermolabile: bool = False,
    ) -> List[PhytochemicalRecord]:
        """Filter candidate phytochemicals by extraction plausibility score, verdict, and thermal properties."""
        filtered: List[PhytochemicalRecord] = []
        verdict_set: Optional[Set[ExtractionVerdict]] = set(verdicts) if verdicts is not None else None

        for compound in self.candidate_phytochemicals:
            if compound.extraction_plausibility_score < min_score:
                continue
            if verdict_set and compound.extraction_verdict not in verdict_set:
                continue
            if exclude_thermolabile and compound.is_thermolabile:
                continue
            filtered.append(compound)
        return filtered

    @property
    def high_plausibility_compounds(self) -> List[PhytochemicalRecord]:
        """Convenience property returning all candidate compounds with verdict 'high'."""
        return [c for c in self.candidate_phytochemicals if c.extraction_verdict == "high"]


# ==============================================================================
# JSON Schema Export Utilities
# ==============================================================================

def get_module_json_schemas() -> Dict[str, Any]:
    """Generate OpenAPI / JSON Schema definitions for all DecoctionGraph Pydantic models.

    Returns:
        Dict mapping model names to their complete JSON Schema (Draft 2020-12 / Draft-07 compatible).
    """
    models = [
        EthnoPreparationRecord,
        TaxonResolution,
        PhytochemicalRecord,
        ExtractionBatch,
        EnrichedPlantGraph,
    ]
    return {model.__name__: model.model_json_schema() for model in models}


# Module public exports
__all__ = [
    # Literal types and aliases
    "PlantPart",
    "plant_part",
    "PreparationType",
    "preparation_type",
    "SolventClass",
    "solvent_class",
    "ThermalRegime",
    "thermal_regime",
    "AdministrationRoute",
    "administration_route",
    "ExtractionVerdict",
    "extraction_verdict",
    # Introspection tuples
    "PLANT_PARTS",
    "PREPARATION_TYPES",
    "SOLVENT_CLASSES",
    "THERMAL_REGIMES",
    "ADMINISTRATION_ROUTES",
    "EXTRACTION_VERDICTS",
    # Models
    "DecoctionBaseModel",
    "EthnoPreparationRecord",
    "TaxonResolution",
    "PhytochemicalRecord",
    "ExtractionBatch",
    "EnrichedPlantGraph",
    # Utilities
    "get_module_json_schemas",
]

