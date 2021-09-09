import networkx as nx
import matplotlib.pyplot as plt


def build_graph(pairs, nodes=None):
    start_list, order_list, end_list = [], [], []
    for p in pairs:
        split = p.split(' ')
        assert len(split) == 3
        start_list.append(split[0])
        order_list.append(split[1])
        end_list.append(split[2])
    
    nodes = set(start_list) | set(end_list) if nodes is None else nodes
    
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
    for start, order_symbol, end in zip(start_list, order_list, end_list):
        if order_symbol == '<':
            G.add_edge(start, end)
        elif order_symbol == '>':
            G.add_edge(end, start)
        else:
            raise ValueError(f'Unexpected order: {order_symbol}')
    return G


def plot_graph(graph):
    pos = nx.spring_layout(graph)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, node_color='w', edgecolors='black', ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=2, edge_color='black', ax=ax)
    return None


def get_possible_orders(possible_pairs, nodes=None):
    possible_orders = []
    for pairs in possible_pairs:
        G = build_graph(pairs, nodes)
        if nx.is_directed_acyclic_graph(G):
            print(f'Order {pairs} is a DAG')
            for order in nx.algorithms.all_topological_sorts(G):
                possible_orders.append(order)
        else:
            print(f'Order {pairs} is NOT a DAG')
    return possible_orders


# pairs = ['Er > Et', 'Fr < Fe', 'Fr > Ga', 'Fe < Et', 'Ga > Et']  # not a dag
# pairs = ['Er > Et', 'Fr > Fe', 'Fr < Ga', 'Fe < Et', 'Ga > Et']
possible_pairs = [
    ['Er > Et', 'Fr < Fe', 'Fr > Ga', 'Fe < Et', 'Ga > Et'],
    ['Er > Et', 'Fr > Fe', 'Fr < Ga', 'Fe < Et', 'Ga > Et']
]
possible_orders = get_possible_orders(possible_pairs, nodes=None)
for order in possible_orders:
    print(' -> '.join(order))


possible_pairs = [
    ['Mat < Lau']
]
nodes = ['Mat', 'Lau', 'Mar', 'Nat']
possible_orders = get_possible_orders(possible_pairs, nodes=nodes)
for order in possible_orders:
    if order[2] != 'Mar' and order[3] != 'Mar':
        print(' -> '.join(order))