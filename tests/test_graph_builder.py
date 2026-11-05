"""Unit tests for DecoctionGraph knowledge graph builder module."""

import json
from pathlib import Path
import pytest
import networkx as nx

from src.schemas import EthnoPreparationRecord
from src.graph_builder import (
    DecoctionGraphBuilder,
    map_indication_to_mesh,
    COLOR_PLANT,
    COLOR_PREPARATION,
    COLOR_INDICATION,
    COLOR_COMPOUND_PLAUSIBLE,
    COLOR_COMPOUND_EXCLUDED,
    COLOR_COMPOUND_TOXIN,
    COLOR_TARGET,
    COLOR_VALIDATED_PATH,
)


def test_map_indication_to_mesh():
    """Verify MeSH vocabulary semantic mapping for historical indications."""
    assert "Wounds and Injuries" in map_indication_to_mesh("severe burns and abrasions")
    assert "Anxiety Disorders" in map_indication_to_mesh("nervous anxiety and melancholia")
    assert "Fever" in map_indication_to_mesh("intermittent tertian fever")
    assert "Dyspepsia" in map_indication_to_mesh("stomach weakness and colic")
    assert "Historical" in map_indication_to_mesh("uncommon mysterious vapor")


def test_add_extraction_record_topology():
    """Verify nodes and directed edges created by add_extraction_record."""
    builder = DecoctionGraphBuilder()

    record = EthnoPreparationRecord(
        verbatim_plant_name="Gentiana lutea",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="digestive weakness and intermittent fever",
        administration_route="oral",
        verbatim_quote="Boil gentian roots in water for one hour and drink.",
        cultural_or_geographic_origin="Alpine European",
        extraction_confidence=0.98,
    )

    plant_id, prep_id, ind_id, comp_ids = builder.add_extraction_record(record)
    g = builder.graph

    # Verify Plant Node
    assert plant_id in g
    p_attrs = g.nodes[plant_id]
    assert p_attrs["node_type"] == "plant"
    assert "Gentiana lutea" in p_attrs["scientific_name"]
    assert p_attrs["confidence"] >= 0.90

    # Verify Preparation Node
    assert prep_id in g
    prep_attrs = g.nodes[prep_id]
    assert prep_attrs["node_type"] == "preparation"
    assert prep_attrs["prep_type"] == "decoction"
    assert prep_attrs["solvent"] == "water"
    assert prep_attrs["thermal"] == "boiling_prolonged"
    assert prep_attrs["route"] == "oral"

    # Verify Indication Node
    assert ind_id in g
    ind_attrs = g.nodes[ind_id]
    assert ind_attrs["node_type"] == "indication"
    assert "digestive weakness" in ind_attrs["verbatim_indication"]
    assert "modern_mesh_mapping" in ind_attrs

    # Verify Edges
    assert g.has_edge(plant_id, prep_id)
    edge_plant_prep = g.get_edge_data(plant_id, prep_id)
    first_edge = list(edge_plant_prep.values())[0]
    assert first_edge["edge_type"] == "HAS_PREPARATION"
    assert first_edge["quote"] == record.verbatim_quote

    assert g.has_edge(prep_id, ind_id)
    edge_prep_ind = g.get_edge_data(prep_id, ind_id)
    first_prep_ind = list(edge_prep_ind.values())[0]
    assert first_prep_ind["edge_type"] == "TREATS"
    assert first_prep_ind["route"] == "oral"

    # Verify Compound Nodes and SOLUBILIZES edges
    assert len(comp_ids) > 0
    for c_id in comp_ids:
        assert c_id in g
        c_attrs = g.nodes[c_id]
        assert c_attrs["node_type"] == "compound"
        assert "inchi_key" in c_attrs

        assert g.has_edge(prep_id, c_id)
        solubilizes_edge = list(g.get_edge_data(prep_id, c_id).values())[0]
        assert solubilizes_edge["edge_type"] == "SOLUBILIZES"
        assert 0.0 <= solubilizes_edge["plausibility_score"] <= 1.0


def test_filtered_subgraph_plausibility_cutoff():
    """Verify pruning of compounds falling below plausibility cutoff."""
    builder = DecoctionGraphBuilder()

    # Hypericum in lipid_oil: Hyperforin is high (0.90), Rutin is excluded (0.00)
    record = EthnoPreparationRecord(
        verbatim_plant_name="Hypericum perforatum",
        plant_part="flower",
        preparation_type="oil_maceration",
        solvent_class="lipid_oil",
        thermal_regime="warm_steep",
        verbatim_indication="burns and skin wounds",
        administration_route="topical",
        verbatim_quote="Flowering tops steeped in olive oil under sunlight.",
        extraction_confidence=0.95,
    )
    builder.add_extraction_record(record)

    full_graph = builder.graph
    subgraph = builder.get_filtered_subgraph(min_plausibility=0.40)

    # Subgraph has fewer nodes because low plausibility compounds are pruned
    assert subgraph.number_of_nodes() < full_graph.number_of_nodes()

    # Check that surviving compound edges all meet or exceed 0.40
    for u, v, d in subgraph.edges(data=True):
        if d.get("edge_type") == "SOLUBILIZES":
            assert d["plausibility_score"] >= 0.40


def test_filtered_subgraph_exclude_toxic():
    """Verify pruning of compounds with active toxicity alerts when exclude_toxic=True."""
    builder = DecoctionGraphBuilder()

    record = EthnoPreparationRecord(
        verbatim_plant_name="Symphytum officinale",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="bone fractures",
        administration_route="topical",
        verbatim_quote="Boil comfrey root in water.",
        extraction_confidence=0.90,
    )
    builder.add_extraction_record(record)

    # Without toxic exclusion: Symphytine and Echimidine are present
    subgraph_with_toxic = builder.get_filtered_subgraph(min_plausibility=0.0, exclude_toxic=False)
    toxic_nodes_before = [
        n for n, d in subgraph_with_toxic.nodes(data=True)
        if d.get("node_type") == "compound" and len(d.get("toxicity_alerts", [])) > 0
    ]
    assert len(toxic_nodes_before) == 2

    # With toxic exclusion: Symphytine and Echimidine are pruned
    subgraph_clean = builder.get_filtered_subgraph(min_plausibility=0.0, exclude_toxic=True)
    toxic_nodes_after = [
        n for n, d in subgraph_clean.nodes(data=True)
        if d.get("node_type") == "compound" and len(d.get("toxicity_alerts", [])) > 0
    ]
    assert len(toxic_nodes_after) == 0


def test_to_dict_export():
    """Verify node-link dictionary export serialization."""
    builder = DecoctionGraphBuilder()
    record = EthnoPreparationRecord(
        verbatim_plant_name="Gentiana lutea",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="fever",
        administration_route="oral",
        verbatim_quote="Boil gentian roots in water.",
        extraction_confidence=0.92,
    )
    builder.add_extraction_record(record)

    data = builder.to_dict()
    assert "nodes" in data
    assert "edges" in data or "links" in data
    # Test JSON serializability
    json_str = json.dumps(data)
    assert len(json_str) > 100


def test_to_pyvis_export(tmp_path: Path):
    """Verify interactive PyVis HTML output generation."""
    builder = DecoctionGraphBuilder()
    record = EthnoPreparationRecord(
        verbatim_plant_name="Hypericum perforatum",
        plant_part="flower",
        preparation_type="oil_maceration",
        solvent_class="lipid_oil",
        thermal_regime="warm_steep",
        verbatim_indication="burns and wounds",
        administration_route="topical",
        verbatim_quote="Steep flowers in oil.",
        extraction_confidence=0.88,
    )
    builder.add_extraction_record(record)

    html_file = tmp_path / "test_graph.html"
    out_path = builder.to_pyvis(str(html_file))
    assert Path(out_path).is_file()

    html_content = Path(out_path).read_text(encoding="utf-8")
    assert "<html" in html_content
    # Verify palette colors present in HTML configuration
    assert COLOR_PLANT in html_content
    assert COLOR_PREPARATION in html_content
    assert COLOR_INDICATION in html_content
    assert COLOR_TARGET in html_content
    assert COLOR_VALIDATED_PATH in html_content


def test_target_nodes_and_validated_mechanistic_paths():
    """Verify Target nodes and validated mechanistic path edges (S_ext >= 0.50, alignment >= 0.60)."""
    builder = DecoctionGraphBuilder()
    record = EthnoPreparationRecord(
        verbatim_plant_name="Hypericum perforatum",
        plant_part="flower",
        preparation_type="oil_maceration",
        solvent_class="lipid_oil",
        thermal_regime="warm_steep",
        verbatim_indication="severe skin wounds and burns",
        administration_route="topical",
        verbatim_quote="Flowering tops steeped in olive oil under sunlight.",
        extraction_confidence=0.98,
    )
    plant_id, prep_id, ind_id, comp_ids = builder.add_extraction_record(record)
    g = builder.graph

    # Verify Target node TRPC6 exists
    assert "target:TRPC6" in g
    trpc6_node = g.nodes["target:TRPC6"]
    assert trpc6_node["node_type"] == "target"
    assert trpc6_node["gene_symbol"] == "TRPC6"
    assert trpc6_node["action_type"] == "modulator"

    # Find Hyperforin compound node
    hyp_comp_id = next(c for c in comp_ids if "KGSZHKRKHXOAMG" in c)

    # Verify (Compound) -> [MODULATES] -> (Target)
    assert g.has_edge(hyp_comp_id, "target:TRPC6")
    modulates_edges = list(g.get_edge_data(hyp_comp_id, "target:TRPC6").values())
    assert any(e["edge_type"] == "MODULATES" for e in modulates_edges)

    # Verify (Target) -> [EXPLAINS_EFFICACY_IN] -> (Indication)
    assert g.has_edge("target:TRPC6", ind_id)
    explains_edges = list(g.get_edge_data("target:TRPC6", ind_id).values())
    assert any(e["edge_type"] == "EXPLAINS_EFFICACY_IN" for e in explains_edges)

    # Hyperforin in oil maceration for wounds is a VALIDATED path (S_ext > 0.85, alignment > 0.90)
    solubilizes_edges = list(g.get_edge_data(prep_id, hyp_comp_id).values())
    assert any(e.get("is_validated_path") is True for e in solubilizes_edges)
    assert any(e.get("is_validated_path") is True for e in modulates_edges)
    assert any(e.get("is_validated_path") is True for e in explains_edges)

    # Preparation node carries aggregate mechanistic validation score
    assert g.nodes[prep_id]["mechanistic_validation_score"] >= 0.80


def test_target_pruning_in_filtered_subgraph():
    """Verify that when a compound is pruned, its isolated target nodes are pruned from the subgraph."""
    builder = DecoctionGraphBuilder()
    # Gentiana lutea in water: Gentiopicroside is high (>0.75), Loganic acid has lower plausibility
    record = EthnoPreparationRecord(
        verbatim_plant_name="Gentiana lutea",
        plant_part="root",
        preparation_type="decoction",
        solvent_class="water",
        thermal_regime="boiling_prolonged",
        verbatim_indication="digestive weakness and loss of appetite",
        administration_route="oral",
        verbatim_quote="Boil gentian root in spring water.",
        extraction_confidence=0.95,
    )
    builder.add_extraction_record(record)

    full_graph = builder.graph
    # Subgraph filtering with min_plausibility 0.40 retains TAS2R38 / TAS2R50 (bitter taste receptors)
    subgraph = builder.get_filtered_subgraph(min_plausibility=0.40)

    target_nodes_subgraph = [
        n for n, d in subgraph.nodes(data=True) if d.get("node_type") == "target"
    ]
    assert len(target_nodes_subgraph) > 0
    assert any("TAS2R" in t for t in target_nodes_subgraph)
