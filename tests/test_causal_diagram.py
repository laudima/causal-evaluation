from causal_evaluation.viewer.causal_diagram import (
    layout_for_topology,
    parse_reasoning_graph,
    supported_topologies,
)


def test_parse_reasoning_graph_extracts_variables_and_edges():
    reasoning = {
        "step0": "Let X = having a sister; V2 = blood pressure; Y = heart condition.",
        "step1": "X->V2,X->Y,V2->Y",
    }
    variables, edges = parse_reasoning_graph(reasoning)
    assert variables == {"X": "having a sister", "V2": "blood pressure", "Y": "heart condition"}
    assert edges == [("X", "V2"), ("X", "Y"), ("V2", "Y")]


def test_parse_reasoning_graph_handles_missing_reasoning():
    assert parse_reasoning_graph(None) == ({}, [])


def test_layout_for_topology_mediation_places_treatment_and_outcome_apart():
    variables = {"X": "a", "V2": "b", "Y": "c"}
    edges = [("X", "V2"), ("X", "Y"), ("V2", "Y")]
    positions = layout_for_topology("mediation", variables, edges, "X", "Y")
    assert set(positions) == {"X", "V2", "Y"}
    assert positions["X"][0] != positions["Y"][0]
    assert positions["V2"][1] != positions["X"][1]


def test_layout_for_topology_confounding_places_confounder_apart_from_treatment_and_outcome():
    variables = {"V1": "a", "X": "b", "Y": "c"}
    edges = [("V1", "X"), ("V1", "Y"), ("X", "Y")]
    positions = layout_for_topology("confounding", variables, edges, "X", "Y")
    assert set(positions) == {"V1", "X", "Y"}
    assert positions["X"][0] != positions["Y"][0]
    assert positions["V1"][1] != positions["X"][1]


def test_layout_for_topology_fork_converges_causes_on_outcome():
    variables = {"X": "a", "V2": "b", "Y": "c"}
    edges = [("X", "Y"), ("V2", "Y")]
    positions = layout_for_topology("fork", variables, edges, "X", "Y")
    assert set(positions) == {"X", "V2", "Y"}
    assert positions["X"][0] != positions["Y"][0]
    assert positions["X"][1] != positions["V2"][1]


def test_layout_for_topology_diamond_spaces_parallel_mediators_between_treatment_and_outcome():
    variables = {"X": "a", "V2": "b", "V3": "c", "Y": "d"}
    edges = [("X", "V2"), ("X", "V3"), ("V2", "Y"), ("V3", "Y")]
    positions = layout_for_topology("diamond", variables, edges, "X", "Y")
    assert set(positions) == {"X", "V2", "V3", "Y"}
    assert positions["X"][0] != positions["Y"][0]
    assert positions["V2"][1] != positions["V3"][1]
    assert positions["V2"][0] == positions["V3"][0]


def test_layout_for_topology_diamondcut_roots_at_the_structural_source_not_the_treatment():
    # V1 is the true diamond source here; X (the treatment) is a parallel
    # middle node alongside V3, not the left point - this is the whole
    # reason diamondcut needs edges instead of just the treatment label.
    variables = {"V1": "ceo", "V3": "director", "X": "manager", "Y": "employee"}
    edges = [("V1", "V3"), ("V1", "X"), ("X", "Y"), ("V3", "Y")]
    positions = layout_for_topology("diamondcut", variables, edges, "X", "Y")
    assert set(positions) == {"V1", "V3", "X", "Y"}
    assert positions["V1"][0] == 0.0
    assert positions["Y"][0] == 640.0
    assert positions["X"][0] == positions["V3"][0] == 320.0
    assert positions["X"][1] != positions["V3"][1]


def test_layout_for_topology_arrowhead_places_both_roots_left_and_mediator_between():
    # X and V2 are symmetric roots that both feed V3 and Y directly; neither
    # root is structurally special, so this only works if roles come from
    # in-degree rather than the treatment label (same lesson as diamondcut).
    variables = {"X": "gender", "V2": "residency", "V3": "competitiveness", "Y": "admission"}
    edges = [("X", "V3"), ("V2", "V3"), ("X", "Y"), ("V2", "Y"), ("V3", "Y")]
    positions = layout_for_topology("arrowhead", variables, edges, "X", "Y")
    assert set(positions) == {"X", "V2", "V3", "Y"}
    assert positions["X"][0] == positions["V2"][0] == 0.0
    assert positions["X"][1] != positions["V2"][1]
    assert positions["V3"][0] not in (positions["X"][0], positions["Y"][0])


def test_layout_for_topology_collision_puts_treatment_and_outcome_on_a_separate_third_node():
    # X and Y have no edge between them at all here - both independently
    # cause V3, unlike fork where the causes converge on the outcome itself.
    variables = {"Y": "talent", "X": "appearance", "V3": "fame"}
    edges = [("X", "V3"), ("Y", "V3")]
    positions = layout_for_topology("collision", variables, edges, "X", "Y")
    assert set(positions) == {"X", "Y", "V3"}
    assert positions["X"][1] != positions["Y"][1]
    assert positions["V3"][0] not in (positions["X"][0], positions["Y"][0])


def test_layout_for_topology_chain_orders_nodes_along_the_actual_edge_path():
    variables = {"X": "education level", "V2": "skill", "Y": "salary"}
    edges = [("X", "V2"), ("V2", "Y")]
    positions = layout_for_topology("chain", variables, edges, "X", "Y")
    assert positions["X"][0] < positions["V2"][0] < positions["Y"][0]
    assert positions["X"][1] == positions["V2"][1] == positions["Y"][1]


def test_layout_for_topology_iv_separates_instrument_from_confounder():
    # V1 (confounder) feeds both X and Y; V2 (instrument) feeds only X.
    # Distinguishing them requires edges, not just the treatment label.
    variables = {"V2": "assignment", "V1": "unobserved", "X": "drug", "Y": "cholesterol"}
    edges = [("V1", "X"), ("V2", "X"), ("V1", "Y"), ("X", "Y")]
    positions = layout_for_topology("IV", variables, edges, "X", "Y")
    assert set(positions) == {"V1", "V2", "X", "Y"}
    assert positions["V2"][0] < positions["X"][0] < positions["Y"][0]
    assert positions["V1"][1] != positions["X"][1]


def test_layout_for_topology_frontdoor_has_no_direct_treatment_outcome_edge_in_its_shape():
    # V1 confounds X and Y directly; V3 sits on the only X->Y path.
    variables = {"V1": "gender", "X": "smoking", "V3": "tar", "Y": "cancer"}
    edges = [("V1", "X"), ("X", "V3"), ("V1", "Y"), ("V3", "Y")]
    positions = layout_for_topology("frontdoor", variables, edges, "X", "Y")
    assert set(positions) == {"V1", "V3", "X", "Y"}
    assert positions["X"][0] != positions["Y"][0]
    assert positions["V3"][1] != positions["V1"][1]


def test_layout_for_topology_unimplemented_returns_none():
    assert (
        layout_for_topology("made-up-topology", {"X": "a", "Y": "b"}, [("X", "Y")], "X", "Y")
        is None
    )
    assert {
        "mediation",
        "confounding",
        "fork",
        "diamond",
        "diamondcut",
        "arrowhead",
        "collision",
        "chain",
        "IV",
        "frontdoor",
    } <= set(supported_topologies())
