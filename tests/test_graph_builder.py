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
        verbatim_indication="burns",
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
