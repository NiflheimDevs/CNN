"""
nngraph/graph_utils.py

Shared graph-traversal utilities needed by more than one phase.
Deliberately small and focused -- currently just the topological sort,
which codegen.py (to decide forward()'s statement order) and
shape_inference.py (to propagate shapes in dependency order) both need
identically: same graph, same declaration-order tie-breaking, same
edge structure. Extracting it once avoids a second copy silently
drifting from the first, the same reasoning that produced
layer_catalogue.py.
"""

from nngraph.ast_nodes import Edge, Program


class GraphCycleError(Exception):
    """Raised if topological_order() can't order every node. Should be
    unreachable if SemanticAnalyzer._check_acyclic() already ran and
    found nothing -- this is a defensive backstop, not the primary
    cycle-detection path."""


def topological_order(program: Program, outgoing: dict[str, list[Edge]]) -> list[str]:
    """Kahn's algorithm, tie-broken by declaration order: two graphs
    whose independent branches are simply listed in a different source
    order should not produce differently-ordered results between
    compiler runs. The virtual input node is always ordered first
    (declaration index -1); every graph node follows in the order it
    was written in the .nng source.
    """
    decl_index = {program.model.input.name: -1}
    for i, n in enumerate(program.graph.nodes):
        decl_index[n.id] = i

    in_degree = {node_id: 0 for node_id in decl_index}
    for edges in outgoing.values():
        for edge in edges:
            in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1

    ready = [node_id for node_id, deg in in_degree.items() if deg == 0]
    order: list[str] = []
    while ready:
        ready.sort(key=lambda node_id: decl_index[node_id])
        current = ready.pop(0)
        order.append(current)
        for edge in outgoing.get(current, []):
            in_degree[edge.dst] -= 1
            if in_degree[edge.dst] == 0:
                ready.append(edge.dst)

    if len(order) != len(decl_index):
        raise GraphCycleError(
            "Topological sort could not order all nodes -- the graph "
            "still contains a cycle. This should already have been "
            "rejected by SemanticAnalyzer._check_acyclic()."
        )
    return order
