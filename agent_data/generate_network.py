import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import random
import pickle
import copy


def compute_target_edges(n, target_density=0.1):
    return int(target_density * n * (n - 1) / 2)


def adjust_edge_count(G, E_target):
    current_E = G.number_of_edges()
    if current_E > E_target:
        edges_to_remove = list(G.edges())[: current_E - E_target]
        G.remove_edges_from(edges_to_remove)
    elif current_E < E_target:
        nodes = list(G.nodes())
        existing_edges = set(G.edges())
        added = 0
        while added < E_target - current_E:
            u, v = random.sample(nodes, 2)
            if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                G.add_edge(u, v)
                existing_edges.add((u, v))
                added += 1
    return G


def relabel_nodes(G):
    mapping = {old: old + 1 for old in G.nodes()}
    return nx.relabel_nodes(G, mapping)


def save_network_info(G, output_filename):
    network_info = {
        "Number_of_nodes": G.number_of_nodes(),
        "Number_of_edges": G.number_of_edges(),
        "Density": nx.density(G),
        "Average_clustering_coefficient": nx.average_clustering(G),
        "Average_shortest_path_length": None,
        "Connections": {node: list(G.neighbors(node)) for node in G.nodes()},
    }
    if nx.is_connected(G):
        network_info["Average_shortest_path_length"] = nx.average_shortest_path_length(
            G
        )
    else:
        network_info["Average_shortest_path_length"] = "Graph is not connected."

    with open(output_filename, "w") as f:
        json.dump(network_info, f, indent=4)


def draw_and_save(G, image_path1, image_path2, graph_ml, graph_pkl, model_name):
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.savefig(image_path1, dpi=300, bbox_inches="tight")
    plt.clf()

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        edge_color="gray",
        width=2,
        alpha=0.7,
        font_size=12,
        font_weight="bold",
    )
    plt.savefig(image_path2, dpi=300, bbox_inches="tight")
    plt.clf()

    with open(graph_pkl, "wb") as f:
        pickle.dump(G, f)
    if model_name == "stochastic_block":
        G2 = copy.deepcopy(G)
        G2.graph.clear()
        nx.write_graphml(G2, graph_ml)
    else:
        nx.write_graphml(G, graph_ml)


def save_graph(G, model_name, profile_folder):
    G = relabel_nodes(G)
    img1 = f"{model_name}_network.png"
    img2 = f"{model_name}_network_improved.png"
    graph_ml = f"{model_name}_graph.praphml"
    graph_pkl = f"{model_name}_graph.pkl"
    json_file = "network_info.json"
    os.makedirs(profile_folder, exist_ok=True)
    if not os.path.exists(os.path.join(profile_folder, json_file)):
        save_network_info(G, json_file)
        draw_and_save(G, img1, img2, graph_ml, graph_pkl, model_name)
        for f in [img1, img2, json_file, graph_ml, graph_pkl]:
            os.replace(f, os.path.join(profile_folder, f))


def generate_erdos_renyi_network(n, target_density=0.1, profile_folder="../profile"):
    E_target = compute_target_edges(n, target_density)
    G = nx.gnm_random_graph(n, E_target)
    save_graph(G, "erdos_renyi", profile_folder)


def generate_watts_strogatz_network(n, target_density=0.1, profile_folder="../profile"):
    E_target = compute_target_edges(n, target_density)
    k = int(round(2 * E_target / n))
    if k % 2 != 0:
        k += 1
    if k >= n:
        k = n - 1 if (n - 1) % 2 == 0 else n - 2
    G = nx.watts_strogatz_graph(n, k, p=0.1)
    G = adjust_edge_count(G, E_target)
    save_graph(G, "watts_strogatz", profile_folder)


def generate_barabasi_albert_network(
    n, target_density=0.1, profile_folder="../profile"
):
    E_target = compute_target_edges(n, target_density)
    m = max(1, min(int(E_target / n), n - 1))
    G = nx.barabasi_albert_graph(n, m)
    G = adjust_edge_count(G, E_target)
    save_graph(G, "barabasi_albert", profile_folder)


def estimate_expected_edges(sizes, p):
    k = len(sizes)
    total = 0
    for i in range(k):
        total += (sizes[i] * (sizes[i] - 1) // 2) * p[i][i]
        for j in range(i + 1, k):
            total += sizes[i] * sizes[j] * p[i][j]
    return total


def generate_stochastic_block_network(
    n, target_density=0.1, profile_folder="../profile"
):
    E_target = compute_target_edges(n, target_density)
    if n < 100:
        group_size = 16
    elif n > 400:
        group_size = 30
    else:
        group_size = 22
    k = round(n / group_size)
    if k == 0:
        k = 1
    total_nodes = k * group_size
    sizes = [group_size] * k
    if total_nodes < n:
        remainder = n - total_nodes
        for i in range(remainder):
            sizes[i % k] += 1
    elif total_nodes > n:
        surplus = total_nodes - n
        for i in range(surplus):
            sizes[i % k] -= 1
    best_diff = float("inf")
    best_p = None
    for p_in in [x / 100 for x in range(50, 70, 5)]:
        for p_out in [x / 100 for x in range(1, 11, 2)]:
            p_matrix = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
            E_est = estimate_expected_edges(sizes, p_matrix)
            diff = abs(E_est - E_target)
            if diff < best_diff:
                best_diff = diff
                best_p = p_matrix
    G = nx.stochastic_block_model(sizes, best_p, seed=42)
    G = adjust_edge_count(G, E_target)
    save_graph(G, "stochastic_block", profile_folder)

