"""DecoctionGraph Knowledge Graph Builder Module.

Synthesizes the outputs of ethnobotanical literature extraction, taxonomic resolution,
LOTUS natural product retrieval, and cheminformatics extraction plausibility filtering
into a bipartite / multipartite knowledge graph using NetworkX and PyVis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from pyvis.network import Network

from src.schemas import (
    EthnoPreparationRecord,
    PhytochemicalRecord,
    TaxonResolution,
)
from src.taxonomy import resolve_plant_name
from src.cheminformatics import calculate_extraction_plausibility, map_plausibility_to_verdict
from src.lotus_retriever import enrich_taxon_compounds

logger = logging.getLogger(__name__)

# ==============================================================================
# Visualization Palette Constants
# ==============================================================================

COLOR_PLANT = "#2E7D32"        # Green
COLOR_PREPARATION = "#1565C0"  # Blue
COLOR_INDICATION = "#E65100"   # Orange
COLOR_COMPOUND_PLAUSIBLE = "#7B1FA2"  # Violet
COLOR_COMPOUND_EXCLUDED = "#757575"   # Grey
COLOR_COMPOUND_TOXIN = "#C62828"      # Red

# ==============================================================================
# MeSH Semantic Indication Mapper
# ==============================================================================

def map_indication_to_mesh(verbatim_indication: str) -> str:
    """Map historical vernacular indication phrases to standard MeSH descriptors."""
    lower = verbatim_indication.lower().strip()
    if any(k in lower for k in ["wound", "burn", "abrasion", "cut", "ulcer", "sore", "lesion"]):
        return "Wounds and Injuries [MeSH: D014947]"
    if any(k in lower for k in ["anxiety", "melancholia", "depression", "nervous", "hysteria"]):
        return "Anxiety Disorders [MeSH: D001008] / Depressive Disorder [MeSH: D003866]"
    if any(k in lower for k in ["fever", "ague", "pyrexia", "tertian", "intermittent"]):
        return "Fever [MeSH: D005334]"
    if any(k in lower for k in ["stomach", "digestive", "dyspepsia", "gastric", "colic"]):
        return "Dyspepsia [MeSH: D004415] / Gastrointestinal Diseases [MeSH: D005767]"
    if any(k in lower for k in ["rheumatic", "rheumatism", "gout", "arthritis", "joint"]):
        return "Rheumatic Diseases [MeSH: D012216] / Arthralgia [MeSH: D018771]"
    if any(k in lower for k in ["cough", "bronchitis", "phthisis", "catarrh", "chest"]):
        return "Respiratory Tract Diseases [MeSH: D012140]"
    if any(k in lower for k in ["headache", "cephalalgia", "migraine"]):
        return "Headache [MeSH: D006261]"
    return "Historical Medical Indication [Unclassified]"


# ==============================================================================
# Knowledge Graph Builder
# ==============================================================================

class DecoctionGraphBuilder:
    """Synthesizes ethnobotanical extractions, taxonomy, and chemistry into a multipartite graph.

    Graph Architecture:
      Nodes:
        - Plant: Standardized scientific taxon with GBIF backbone resolution
        - Preparation: Chemical extraction protocol (solvent, thermal, plant part, route)
        - Indication: Historical ailment mapped to MeSH vocabulary
        - Compound: Secondary metabolites evaluated for physicochemical plausibility
      Edges:
        - (Plant) -> [HAS_PREPARATION] -> (Preparation)
        - (Preparation) -> [TREATS] -> (Indication)
        - (Preparation) -> [SOLUBILIZES] -> (Compound)
    """

    def __init__(self) -> None:
        """Initialize empty NetworkX MultiDiGraph."""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_extraction_record(
        self,
        record: EthnoPreparationRecord,
        taxon_resolution: Optional[TaxonResolution] = None,
    ) -> Tuple[str, str, str, List[str]]:
        """Add an ethnobotanical preparation record to the knowledge graph.

        Resolves taxonomy (if not supplied), enriches secondary metabolites via LOTUS,
        evaluates extraction plausibility under preparation conditions, and connects
        plants, preparations, indications, and compounds.

        Args:
            record: Extracted ethnobotanical preparation record.
            taxon_resolution: Optional pre-resolved TaxonResolution instance.

        Returns:
            Tuple of (plant_node_id, prep_node_id, indication_node_id, compound_node_ids).
        """
        # 1. Resolve taxonomy if not provided
        if taxon_resolution is None:
            taxon_resolution = resolve_plant_name(record.verbatim_plant_name)

        scientific_name = taxon_resolution.accepted_scientific_name
        gbif_key = taxon_resolution.gbif_usage_key

        # 2. Plant Node
        plant_node_id = f"plant:{scientific_name}"
        if plant_node_id not in self.graph:
            self.graph.add_node(
                plant_node_id,
                node_type="plant",
                label=scientific_name,
                scientific_name=scientific_name,
                gbif_key=gbif_key,
                vernacular_name=record.verbatim_plant_name,
                confidence=taxon_resolution.confidence,
                taxon_rank=taxon_resolution.taxon_rank,
                taxonomic_status=taxon_resolution.taxonomic_status,
                is_resolved=taxon_resolution.is_resolved,
            )

        # 3. Preparation Node
        prep_node_id = (
            f"prep:{scientific_name}:{record.plant_part}:{record.preparation_type}:"
            f"{record.solvent_class}:{record.thermal_regime}"
        )
        if prep_node_id not in self.graph:
            self.graph.add_node(
                prep_node_id,
                node_type="preparation",
                label=f"{record.preparation_type}\n({record.solvent_class}, {record.thermal_regime})",
                prep_type=record.preparation_type,
                solvent=record.solvent_class,
                thermal=record.thermal_regime,
                part=record.plant_part,
                route=record.administration_route,
                quote=record.verbatim_quote,
                confidence=record.extraction_confidence,
                origin=record.cultural_or_geographic_origin,
                ambiguity_notes=record.ambiguity_notes,
                estimated_ph=record.estimated_ph,
                duration_minutes=record.duration_minutes,
                temperature_celsius=record.temperature_celsius,
            )

        # 4. Indication Node
        clean_indication = record.verbatim_indication.strip()
        mesh_mapping = map_indication_to_mesh(clean_indication)
        indication_node_id = f"indication:{clean_indication.lower()}"
        if indication_node_id not in self.graph:
            self.graph.add_node(
                indication_node_id,
                node_type="indication",
                label=clean_indication,
                verbatim_indication=clean_indication,
                modern_mesh_mapping=mesh_mapping,
            )

        # 5. Core Topology Edges
        # (Plant) -> [HAS_PREPARATION] -> (Preparation)
        self.graph.add_edge(
            plant_node_id,
            prep_node_id,
            key=f"HAS_PREP_{prep_node_id}",
            edge_type="HAS_PREPARATION",
            quote=record.verbatim_quote,
            source=record.cultural_or_geographic_origin or "Unspecified",
            confidence=record.extraction_confidence,
        )

        # (Preparation) -> [TREATS] -> (Indication)
        self.graph.add_edge(
            prep_node_id,
            indication_node_id,
            key=f"TREATS_{indication_node_id}",
            edge_type="TREATS",
            route=record.administration_route,
        )

        # 6. LOTUS Secondary Metabolite Enrichment & Cheminformatics Filtering
        enriched_compounds: List[PhytochemicalRecord] = enrich_taxon_compounds(
            accepted_name=scientific_name,
            solvent_class=record.solvent_class,
            thermal_regime=record.thermal_regime,
            gbif_usage_key=gbif_key,
            estimated_ph=record.estimated_ph,
            duration_minutes=record.duration_minutes,
            temperature_celsius=record.temperature_celsius,
            prepared_part=record.plant_part,
        )

        compound_node_ids: List[str] = []
        for comp in enriched_compounds:
            comp_node_id = f"compound:{comp.inchi_key}"
            compound_node_ids.append(comp_node_id)

            if comp_node_id not in self.graph:
                self.graph.add_node(
                    comp_node_id,
                    node_type="compound",
                    label=comp.compound_name,
                    compound_name=comp.compound_name,
                    smiles=comp.smiles,
                    inchi_key=comp.inchi_key,
                    molecular_formula=comp.molecular_formula,
                    mw=comp.mw,
                    clogp=comp.clogp,
                    tpsa=comp.tpsa,
                    is_thermolabile=comp.is_thermolabile,
                    is_volatile=comp.is_volatile,
                    toxicity_alerts=comp.toxicity_alerts,
                    has_toxicity_alerts=len(comp.toxicity_alerts) > 0,
                    doi=comp.provenance_doi,
                    acid_base_class=comp.acid_base_class,
                    estimated_pka=comp.estimated_pka,
                    logd_at_recipe_ph=comp.logd_at_recipe_ph,
                    thermal_half_life_min_at_100c=comp.thermal_half_life_min_at_100c,
                    thermal_survival_fraction=comp.thermal_survival_fraction,
                    documented_organs=comp.documented_organs,
                    organ_match_factor=comp.organ_match_factor,
                    mechanistic_alignment_score=comp.mechanistic_alignment_score,
                    known_targets=[t.model_dump() for t in comp.known_targets],
                )

            # (Preparation) -> [SOLUBILIZES] -> (Compound)
            self.graph.add_edge(
                prep_node_id,
                comp_node_id,
                key=f"SOLUBILIZES_{comp.inchi_key}",
                edge_type="SOLUBILIZES",
                plausibility_score=comp.extraction_plausibility_score,
                status=comp.extraction_verdict,
                verdict=comp.extraction_verdict,
                organ_match_factor=comp.organ_match_factor,
                thermal_survival_fraction=comp.thermal_survival_fraction,
                logd_at_recipe_ph=comp.logd_at_recipe_ph,
            )

        return plant_node_id, prep_node_id, indication_node_id, compound_node_ids

    def get_filtered_subgraph(
        self,
        min_plausibility: float = 0.40,
        exclude_toxic: bool = False,
    ) -> nx.MultiDiGraph:
        """Return a pruned subgraph filtering compounds by plausibility and toxicity alerts.

        Retains only compounds meeting or exceeding min_plausibility for the specified
        preparation regime. If exclude_toxic is True, prunes compounds with active toxicity alerts.

        Args:
            min_plausibility: Minimum extraction plausibility score required for compound retention (default: 0.40).
            exclude_toxic: If True, completely prunes compounds with active structural toxicity alerts.

        Returns:
            Filtered networkx.MultiDiGraph containing qualifying entities and active edges.
        """
        subgraph = nx.MultiDiGraph()

        # 1. Add non-compound nodes (Plants, Preparations, Indications)
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") in ("plant", "preparation", "indication"):
                subgraph.add_node(node, **attrs)

        # 2. Add structural topology edges (HAS_PREPARATION, TREATS)
        for u, v, key, edge_data in self.graph.edges(keys=True, data=True):
            if edge_data.get("edge_type") != "SOLUBILIZES":
                if u in subgraph and v in subgraph:
                    subgraph.add_edge(u, v, key=key, **edge_data)

        # 3. Filter SOLUBILIZES edges and compound nodes
        for u, v, key, edge_data in self.graph.edges(keys=True, data=True):
            if edge_data.get("edge_type") == "SOLUBILIZES":
                score = float(edge_data.get("plausibility_score", 0.0))
                if score < min_plausibility:
                    continue

                comp_attrs = self.graph.nodes[v]
                alerts = comp_attrs.get("toxicity_alerts", [])
                if exclude_toxic and len(alerts) > 0:
                    continue

                # Add qualifying compound node and edge
                if v not in subgraph:
                    subgraph.add_node(v, **comp_attrs)
                subgraph.add_edge(u, v, key=key, **edge_data)

        return subgraph

    def to_dict(self, subgraph: Optional[nx.MultiDiGraph] = None) -> Dict[str, Any]:
        """Return node-link JSON structure for frontend visualization."""
        target_graph = subgraph if subgraph is not None else self.graph
        data = nx.node_link_data(target_graph)

        # Standardize 'edges' and 'links' for cross-version frontend compatibility
        if "edges" in data and "links" not in data:
            data["links"] = data["edges"]
        elif "links" in data and "edges" not in data:
            data["edges"] = data["links"]

        return data

    def to_pyvis(
        self,
        output_html_path: str = "decoction_graph.html",
        subgraph: Optional[nx.MultiDiGraph] = None,
        min_plausibility: float = 0.40,
    ) -> str:
        """Generate an interactive, color-coded HTML visualization using PyVis.

        Color Palette:
          * Plants: Green (#2E7D32)
          * Preparations: Blue (#1565C0)
          * Indications: Orange (#E65100)
          * Toxins: Red (#C62828)
          * Plausible Compounds: Violet (#7B1FA2)
          * Excluded Compounds: Grey (#757575)

        Args:
            output_html_path: Target path for the generated HTML file.
            subgraph: Optional filtered subgraph to render (defaults to full graph).
            min_plausibility: Score threshold distinguishing plausible from excluded compounds.

        Returns:
            Resolved absolute path to the generated HTML file.
        """
        target_graph = subgraph if subgraph is not None else self.graph

        net = Network(height="850px", width="100%", directed=True, notebook=False)
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)

        # 1. Add Nodes with Custom Styling & Rich Tooltips
        for node_id, attrs in target_graph.nodes(data=True):
            n_type = attrs.get("node_type", "unknown")
            label = attrs.get("label", str(node_id))

            if n_type == "plant":
                color = COLOR_PLANT
                size = 28
                shape = "dot"
                title = (
                    f"<b>Botanical Taxon</b><br>"
                    f"Scientific Name: {attrs.get('scientific_name')}<br>"
                    f"Vernacular Name: {attrs.get('vernacular_name')}<br>"
                    f"GBIF Key: {attrs.get('gbif_key')}<br>"
                    f"Taxon Rank: {attrs.get('taxon_rank')}<br>"
                    f"Matching Confidence: {attrs.get('confidence', 1.0):.2f}"
                )

            elif n_type == "preparation":
                color = COLOR_PREPARATION
                size = 24
                shape = "diamond"
                title = (
                    f"<b>Ethnobotanical Preparation</b><br>"
                    f"Type: {attrs.get('prep_type')}<br>"
                    f"Solvent Class: {attrs.get('solvent')}<br>"
                    f"Thermal Regime: {attrs.get('thermal')}<br>"
                    f"Plant Part: {attrs.get('part')}<br>"
                    f"Route: {attrs.get('route')}<br>"
                    f"Quote: <i>\"{attrs.get('quote')}\"</i>"
                )

            elif n_type == "indication":
                color = COLOR_INDICATION
                size = 22
                shape = "star"
                title = (
                    f"<b>Medical Indication</b><br>"
                    f"Ailment: {attrs.get('verbatim_indication')}<br>"
                    f"MeSH: {attrs.get('modern_mesh_mapping')}"
                )

            elif n_type == "compound":
                alerts = attrs.get("toxicity_alerts", [])
                is_toxic = len(alerts) > 0

                # Determine max extraction score to this compound in graph
                scores = [
                    d.get("plausibility_score", 0.0)
                    for _, _, d in target_graph.in_edges(node_id, data=True)
                    if d.get("edge_type") == "SOLUBILIZES"
                ]
                max_score = max(scores) if scores else 0.0

                if is_toxic:
                    color = COLOR_COMPOUND_TOXIN
                    size = 18
                    shape = "triangle"
                elif max_score >= min_plausibility:
                    color = COLOR_COMPOUND_PLAUSIBLE
                    size = 16
                    shape = "dot"
                else:
                    color = COLOR_COMPOUND_EXCLUDED
                    size = 12
                    shape = "dot"

                alerts_text = f"<span style='color:red;'>ALERTS: {', '.join(alerts)}</span><br>" if is_toxic else ""
                title = (
                    f"<b>Secondary Metabolite</b><br>"
                    f"Compound: {attrs.get('compound_name')}<br>"
                    f"Formula: {attrs.get('molecular_formula')}<br>"
                    f"MW: {attrs.get('mw')} Da | cLogP: {attrs.get('clogp')} | TPSA: {attrs.get('tpsa')} Å²<br>"
                    f"Max Plausibility: {max_score:.4f}<br>"
                    f"{alerts_text}"
                    f"InChIKey: {attrs.get('inchi_key')}"
                )

            else:
                color = "#9E9E9E"
                size = 15
                shape = "dot"
                title = str(node_id)

            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                size=size,
                shape=shape,
            )

        # 2. Add Edges with Styling & Labels
        for u, v, key, edge_data in target_graph.edges(keys=True, data=True):
            e_type = edge_data.get("edge_type", "")
            if e_type == "HAS_PREPARATION":
                net.add_edge(
                    u,
                    v,
                    title=f"HAS_PREPARATION (Confidence: {edge_data.get('confidence', 1.0):.2f})",
                    label="HAS_PREPARATION",
                    color=COLOR_PREPARATION,
                    arrows="to",
                )
            elif e_type == "TREATS":
                net.add_edge(
                    u,
                    v,
                    title=f"TREATS ({edge_data.get('route')})",
                    label=f"TREATS ({edge_data.get('route')})",
                    color=COLOR_INDICATION,
                    arrows="to",
                )
            elif e_type == "SOLUBILIZES":
                score = edge_data.get("plausibility_score", 0.0)
                status = edge_data.get("status", "unknown")
                edge_color = COLOR_COMPOUND_PLAUSIBLE if score >= min_plausibility else COLOR_COMPOUND_EXCLUDED
                net.add_edge(
                    u,
                    v,
                    title=f"SOLUBILIZES (Score: {score:.4f}, Verdict: {status})",
                    label=f"{score:.2f} [{status}]",
                    color=edge_color,
                    arrows="to",
                )
            else:
                net.add_edge(u, v, arrows="to")

        # 3. Output HTML
        out_path = Path(output_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        net.write_html(str(out_path))
        logger.info(f"Generated interactive knowledge graph visualization at: {out_path.resolve()}")
        return str(out_path.resolve())


# ==============================================================================
# CLI Demonstration
# ==============================================================================

def main() -> None:
    """CLI demonstration generating and filtering a multi-preparation knowledge graph."""
    print("=" * 105)
    print("DecoctionGraph: Knowledge Graph Construction & Preparation Filtering CLI")
    print("=" * 105)

    builder = DecoctionGraphBuilder()

    # 1. Hypericum perforatum - Topical Oil Maceration
    rec_oil = EthnoPreparationRecord(
        verbatim_plant_name="Hypericum perforatum",
        plant_part="flower",
        preparation_type="oil_maceration",
        solvent_class="lipid_oil",
        thermal_regime="warm_steep",
        verbatim_indication="severe burns, abrasions, and skin wounds",
        administration_route="topical",
        verbatim_quote="Flowering tops steeped in olive oil under sunlight for 40 days to heal skin wounds and burns.",
        cultural_or_geographic_origin="Mediterranean",
        extraction_confidence=0.98,
    )

    # 2. Hypericum perforatum - Oral Aqueous Decoction
    rec_tea = EthnoPreparationRecord(
        verbatim_plant_name="Hypericum perforatum",
        plant_part="leaf",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="melancholia and nervous anxiety",
        administration_route="oral",
        verbatim_quote="Dried leaves boiled in water for 15 minutes to treat nervous anxiety and melancholy.",
        cultural_or_geographic_origin="Mediterranean",
        extraction_confidence=0.99,
    )

    # 3. Symphytum officinale - Boiled Root Decoction (Contains Pyrrolizidine Alkaloid Toxins)
    rec_comfrey = EthnoPreparationRecord(
        verbatim_plant_name="Symphytum officinale",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="bone fractures and severe bruising",
        administration_route="topical",
        verbatim_quote="Comfrey roots boiled in water to form a viscous glue applied over broken bones.",
        cultural_or_geographic_origin="European Folk Medicine",
        extraction_confidence=0.96,
    )

    print("\n[Step 1] Adding Extracted Ethnobotanical Records...")
    for r in [rec_oil, rec_tea, rec_comfrey]:
        builder.add_extraction_record(r)
        print(f"  + Added: {r.verbatim_plant_name} ({r.preparation_type} in {r.solvent_class}) -> {r.verbatim_indication}")

    full_graph = builder.graph
    print(f"\n[Step 2] Full Knowledge Graph Topology:")
    print(f"  * Nodes: {full_graph.number_of_nodes()}")
    print(f"  * Edges: {full_graph.number_of_edges()}")

    # 4. Filtered Subgraph: Plausibility >= 0.40
    filtered_subgraph = builder.get_filtered_subgraph(min_plausibility=0.40, exclude_toxic=False)
    print(f"\n[Step 3] Physicochemically Filtered Subgraph (min_plausibility >= 0.40):")
    print(f"  * Nodes: {filtered_subgraph.number_of_nodes()}")
    print(f"  * Edges: {filtered_subgraph.number_of_edges()}")

    # 5. Non-Toxic Subgraph: Exclude active toxicity alerts
    non_toxic_subgraph = builder.get_filtered_subgraph(min_plausibility=0.40, exclude_toxic=True)
    print(f"\n[Step 4] Detoxified Subgraph (exclude_toxic=True):")
    print(f"  * Nodes: {non_toxic_subgraph.number_of_nodes()}")
    print(f"  * Edges: {non_toxic_subgraph.number_of_edges()}")

    # 6. Export to HTML
    html_path = "data/processed/decoction_knowledge_graph.html"
    builder.to_pyvis(html_path, subgraph=filtered_subgraph, min_plausibility=0.40)
    print(f"\n[Step 5] Exported Interactive PyVis Visualization to: {html_path}")

    # 7. Export to JSON dict
    json_data = builder.to_dict(filtered_subgraph)
    json_path = Path("data/processed/decoction_graph_nodelink.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  * Exported Node-Link JSON to: {json_path}")

    print("\n" + "=" * 105)
    print("DecoctionGraph Construction Completed Successfully.")
    print("=" * 105)


if __name__ == "__main__":
    main()

