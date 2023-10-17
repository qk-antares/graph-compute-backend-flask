import networkx as nx
import time


def compute_ged(graph1, graph2):
    g1 = data_converter(graph1)
    g2 = data_converter(graph2)
    # 计算两个图之间的编辑距离
    start_time = time.time()
    paths, cost = nx.optimal_edit_paths(g1, g2, node_match=node_match, edge_match=edge_match)
    end_time = time.time()
    total_time = end_time - start_time
    return paths, cost, total_time


def data_converter(graph):
    # 创建一个空的图
    g = nx.Graph()

    # 将节点添加到图
    for node_data in graph["nodes"]:
        node_id = node_data['id']
        node_attrs = {'label': node_data['label']}
        g.add_node(node_id, **node_attrs)

    # 将边添加到图
    for edge_data in graph["edges"]:
        source_node_id = edge_data['source']
        target_node_id = edge_data['target']
        g.add_edge(source_node_id, target_node_id)

    return g


def node_match(node1, node2):
    return node1['label'] == node2['label']


def edge_match(edge1, edge2):
    return True
