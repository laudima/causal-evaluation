"""Per-question causal diagram extraction and topology-specific layouts."""

from collections.abc import Callable

Variables = dict[str, str]
Edges = list[tuple[str, str]]
Positions = dict[str, tuple[float, float]]


def parse_reasoning_graph(reasoning: dict | None) -> tuple[Variables, Edges]:
    """Extract {var: label} and (source, target) edges from a CLadder reasoning block."""
    if not reasoning:
        return {}, []

    variables: Variables = {}
    body = reasoning.get("step0", "").strip()
    if body.lower().startswith("let "):
        body = body[4:]
    for clause in body.rstrip(".").split(";"):
        if "=" not in clause:
            continue
        var, _, name = clause.partition("=")
        variables[var.strip()] = name.strip()

    edges: Edges = []
    for pair in reasoning.get("step1", "").split(","):
        if "->" not in pair:
            continue
        source, _, target = pair.strip().partition("->")
        edges.append((source.strip(), target.strip()))

    return variables, edges


def _layout_apex_baseline(
    variables: Variables, edges: Edges, treatment: str, outcome: str
) -> Positions:
    """Triangle: treatment and outcome on a baseline, the remaining variable(s) above.

    Shared by two topologies that differ only in edge direction, not shape:
    - mediation:   X->V2, V2->Y, X->Y   (apex var is a mediator, mid-flow)
    - confounding: V1->X, V1->Y, X->Y   (apex var is a confounder, the common cause)
    """
    apex_vars = [v for v in variables if v not in (treatment, outcome)]
    positions: Positions = {treatment: (0.0, 120.0), outcome: (640.0, 120.0)}
    span, count = 640.0, max(len(apex_vars), 1)
    for i, var in enumerate(apex_vars):
        positions[var] = (span * (i + 1) / (count + 1), -80.0)
    return positions


def _layout_fork(variables: Variables, edges: Edges, treatment: str, outcome: str) -> Positions:
    """Two (or more) independent causes converging on one outcome: X->Y, V2->Y, ...

    Despite the name, this dataset's "fork" is what's classically called a
    collider-on-the-outcome: every non-outcome variable points straight at
    the outcome, with no edges among the causes themselves.
    """
    causes = [v for v in variables if v != outcome]
    positions: Positions = {outcome: (640.0, 0.0)}
    span, count = 300.0, max(len(causes), 1)
    for i, var in enumerate(causes):
        y = span * (i + 0.5) / count - span / 2
        positions[var] = (0.0, y)
    return positions


def _diamond_shape(source: str, middles: list[str], outcome: str) -> Positions:
    """Shared geometry: source at the left point, outcome at the right, middles stacked between."""
    positions: Positions = {source: (0.0, 0.0), outcome: (640.0, 0.0)}
    span, count = 300.0, max(len(middles), 1)
    for i, var in enumerate(middles):
        y = span * (i + 0.5) / count - span / 2
        positions[var] = (320.0, y)
    return positions


def _layout_diamond(variables: Variables, edges: Edges, treatment: str, outcome: str) -> Positions:
    """Diamond: treatment and outcome as left/right points, parallel mediators stacked between.

    X->V2, X->V3, V2->Y, V3->Y - two independent paths from treatment to
    outcome, unlike mediation's single mediator plus a direct X->Y edge.
    """
    mediators = [v for v in variables if v not in (treatment, outcome)]
    return _diamond_shape(treatment, mediators, outcome)


def _layout_diamondcut(
    variables: Variables, edges: Edges, treatment: str, outcome: str
) -> Positions:
    """Diamond rooted at a confounder, not the treatment: V1->V3, V1->X, X->Y, V3->Y.

    The treatment (X) sits as one of the two parallel middle nodes here
    rather than at the diamond's source point, so the source must be found
    structurally (highest out-degree among non-outcome nodes) instead of
    assumed from the treatment label - reusing _layout_diamond's positions
    would put X in the wrong spot.
    """
    non_outcome = [v for v in variables if v != outcome]
    out_degree = dict.fromkeys(non_outcome, 0)
    for src, _dst in edges:
        if src in out_degree:
            out_degree[src] += 1
    source = max(non_outcome, key=lambda v: out_degree[v], default=treatment)
    middles = [v for v in non_outcome if v != source]
    return _diamond_shape(source, middles, outcome)


def _layout_arrowhead(
    variables: Variables, edges: Edges, treatment: str, outcome: str
) -> Positions:
    """Two root causes both feed a shared mediator AND the outcome directly:
    X->V3, V2->V3, X->Y, V2->Y, V3->Y.

    X and V2 are symmetric roots (neither is structurally special), so roles
    are found by degree rather than trusting the treatment label - same
    reasoning as diamondcut. Roots (in-degree 0) go on the left, the
    remaining mediator(s) in the middle, outcome on the right; multiple
    converging edges into Y form the "arrowhead" shape.
    """
    non_outcome = [v for v in variables if v != outcome]
    in_degree = dict.fromkeys(non_outcome, 0)
    for _src, dst in edges:
        if dst in in_degree:
            in_degree[dst] += 1
    roots = [v for v in non_outcome if in_degree[v] == 0] or [treatment]
    middles = [v for v in non_outcome if v not in roots]

    positions: Positions = {outcome: (640.0, 0.0)}
    root_span, root_count = 300.0, max(len(roots), 1)
    for i, var in enumerate(roots):
        y = root_span * (i + 0.5) / root_count - root_span / 2
        positions[var] = (0.0, y)
    mid_span, mid_count = 200.0, max(len(middles), 1)
    for i, var in enumerate(middles):
        y = mid_span * (i + 0.5) / mid_count - mid_span / 2
        positions[var] = (320.0, y)
    return positions


def _layout_collision(
    variables: Variables, edges: Edges, treatment: str, outcome: str
) -> Positions:
    """Treatment and outcome each independently cause a third collider variable: X->V3, Y->V3.

    Unlike fork (which converges into the outcome itself), collision's sink
    is a separate third variable - there is no edge between X and Y at all.
    """
    collider = next((v for v in variables if v not in (treatment, outcome)), None)
    positions: Positions = {treatment: (0.0, -100.0), outcome: (0.0, 100.0)}
    if collider:
        positions[collider] = (500.0, 0.0)
    return positions


def _layout_chain(variables: Variables, edges: Edges, treatment: str, outcome: str) -> Positions:
    """Straight line, ordered by following edges from treatment to outcome: X->V2->...->Y.

    Order matters visually (X->V2->V3->Y differs from X->V3->V2->Y), so the
    path is walked via the edge list rather than just spacing dict keys.
    """
    adjacency = {src: dst for src, dst in edges}
    order = [treatment]
    seen = {treatment}
    current = treatment
    while current in adjacency and current != outcome and len(order) <= len(variables):
        nxt = adjacency[current]
        if nxt in seen:
            break
        order.append(nxt)
        seen.add(nxt)
        current = nxt
    for var in [outcome, *variables]:
        if var not in seen:
            order.append(var)
            seen.add(var)

    span, count = 640.0, max(len(order) - 1, 1)
    return {var: (span * i / count, 0.0) for i, var in enumerate(order)}


def _layout_iv(variables: Variables, edges: Edges, treatment: str, outcome: str) -> Positions:
    """Instrumental variable: V1->X, V2->X, V1->Y, X->Y.

    V1 is an unobserved confounder feeding both treatment and outcome (apex,
    as in confounding); V2 is the instrument feeding only the treatment, so
    it sits on the baseline to the left of treatment. Roles are found
    structurally (by which nodes each "other" variable targets) rather than
    assumed from naming, following the diamondcut/arrowhead precedent.
    """
    others = [v for v in variables if v not in (treatment, outcome)]
    targets = {v: {dst for src, dst in edges if src == v} for v in others}
    confounder = next((v for v in others if outcome in targets.get(v, set())), None)
    instrument = next((v for v in others if v != confounder), None)

    positions: Positions = {treatment: (320.0, 120.0), outcome: (640.0, 120.0)}
    if instrument:
        positions[instrument] = (0.0, 120.0)
    if confounder:
        positions[confounder] = (480.0, -80.0)
    return positions


def _layout_frontdoor(
    variables: Variables, edges: Edges, treatment: str, outcome: str
) -> Positions:
    """Front-door: V1->X, X->V3, V1->Y, V3->Y - no direct X->Y edge.

    V1 confounds treatment and outcome directly (apex, as in confounding);
    V3 sits on the front-door mediating path from treatment to outcome
    instead. Found structurally: the confounder is whichever "other"
    variable is a source for both treatment and outcome.
    """
    others = [v for v in variables if v not in (treatment, outcome)]
    sources_of = {v: {src for src, dst in edges if dst == v} for v in (treatment, outcome)}
    confounder = next(
        (v for v in others if v in sources_of[treatment] and v in sources_of[outcome]), None
    )
    mediator = next((v for v in others if v != confounder), None)

    positions: Positions = {treatment: (0.0, 200.0), outcome: (640.0, 200.0)}
    if mediator:
        positions[mediator] = (320.0, 60.0)
    if confounder:
        positions[confounder] = (320.0, -140.0)
    return positions


# Populated as each topology gets a hand-designed layout. Topologies not yet
# here fall back to a physics-based auto-layout in the viewer (see
# render_viewer_html) instead of a hardcoded diagram shape.
_TOPOLOGY_LAYOUTS: dict[str, Callable[[Variables, Edges, str, str], Positions]] = {
    "mediation": _layout_apex_baseline,
    "confounding": _layout_apex_baseline,
    "fork": _layout_fork,
    "diamond": _layout_diamond,
    "diamondcut": _layout_diamondcut,
    "arrowhead": _layout_arrowhead,
    "collision": _layout_collision,
    "chain": _layout_chain,
    "IV": _layout_iv,
    "frontdoor": _layout_frontdoor,
}


def layout_for_topology(
    topology: str, variables: Variables, edges: Edges, treatment: str, outcome: str
) -> Positions | None:
    """Return fixed node positions for a topology, or None if not yet implemented."""
    layout_fn = _TOPOLOGY_LAYOUTS.get(topology)
    return layout_fn(variables, edges, treatment, outcome) if layout_fn else None


def supported_topologies() -> list[str]:
    """Topologies with a hand-designed diagram layout implemented so far."""
    return list(_TOPOLOGY_LAYOUTS.keys())
