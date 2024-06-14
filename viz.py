from graphviz import Digraph

# visualize_tree: Displays the node tree using graphviz library
# Params: a list of Nodes to visualize
# Returns: the Digraph object to visualize the node tree
def visualize_tree(nodes):
    dot = Digraph()
    seen_edges = set()

    def add_nodes_edges(node, parent=None):
        dot.node(node.prompt)
        if parent:
            edge = (parent.prompt, node.prompt)
            if edge not in seen_edges:
                dot.edge(parent.prompt, node.prompt)
                seen_edges.add(edge)
        for child in node.children:
            add_nodes_edges(child, node)

    for node in nodes:
        add_nodes_edges(node)
    return dot