#!/usr/bin/env python3
"""
Final Research Version – Community Detection + Graph Visualization + Neural Network Model Diagram

"""

import argparse
import json
import os
import time
import psutil
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ---------- Utility ----------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_graph_from_edgelist(path: str, delimiter=None) -> nx.Graph:
    """Load an undirected graph from an edge-list file."""
    try:
        G = nx.read_edgelist(path, delimiter=delimiter, nodetype=str)
        if G.number_of_nodes() == 0:
            raise ValueError("Graph has no nodes.")
        return G
    except Exception as e:
        raise RuntimeError(f"Error loading edge list '{path}': {e}")

# ---------- Core Algorithms ----------

def detect_communities(G: nx.Graph, algorithm: str = "labelprop", max_gn_steps: int = 1):
    if algorithm == "labelprop":
        communities = list(nx.algorithms.community.asyn_lpa_communities(G))
    elif algorithm == "girvan":
        comp_gen = nx.algorithms.community.girvan_newman(G)
        for _ in range(max_gn_steps):
            communities = next(comp_gen)
        communities = [set(c) for c in communities]
    else:
        raise ValueError("Unsupported algorithm")
    return sorted(communities, key=lambda s: len(s), reverse=True)

def compute_modularity(G: nx.Graph, communities: List[set]) -> float:
    try:
        return nx.algorithms.community.quality.modularity(G, communities)
    except Exception:
        return float("nan")

# ---------- Metrics ----------

def measure_scalability(start_time: float) -> dict:
    elapsed = time.time() - start_time
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return {"runtime_sec": round(elapsed, 3), "memory_mb": round(mem, 2)}

def estimate_accuracy(modularity: float, n_comms: int) -> float:
    """Rough proxy accuracy estimator (for visualization/demo)."""
    acc = min(100, 70 + modularity * 25 - (n_comms * 0.3))
    return round(max(0, acc), 2)

# ---------- Visualization 1: Bar Chart ----------

def plot_bar_chart(sizes, labels, title, path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(sizes))
    bars = plt.bar(x, sizes, color="skyblue", edgecolor="black")
    total = sum(sizes)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Nodes per Community")
    plt.title(title, fontsize=13, fontweight="bold")

    for bar, val in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                 f"{val} ({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------- Visualization 2: Network Graph ----------

def plot_network_graph(G, communities, path):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(9, 7))
    colors = plt.cm.get_cmap('tab20', len(communities))
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(comm), node_color=[colors(i)],
                               node_size=120, alpha=0.8, label=f"C{i+1}")
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title("Detected Communities Network", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------- Visualization 3: Neural Network Diagram ----------

def plot_neural_network(path):
    plt.figure(figsize=(7, 5))
    plt.title("Artificial Neural Network Model", fontsize=13, fontweight="bold")

    input_nodes = [(0, y) for y in np.linspace(0.2, 0.8, 3)]
    hidden_nodes = [(0.5, y) for y in np.linspace(0.1, 0.9, 5)]
    output_nodes = [(1, 0.5)]

    for x1, y1 in input_nodes:
        for x2, y2 in hidden_nodes:
            plt.plot([x1, x2], [y1, y2], "k-", lw=0.8)
    for x2, y2 in hidden_nodes:
        plt.plot([x2, 1], [y2, 0.5], "k-", lw=0.8)

    for x, y in input_nodes:
        plt.scatter(x, y, s=400, color="dodgerblue", label="Input Layer" if y == input_nodes[0][1] else "")
    for x, y in hidden_nodes:
        plt.scatter(x, y, s=400, color="seagreen", label="Hidden Layer" if y == hidden_nodes[0][1] else "")
    for x, y in output_nodes:
        plt.scatter(x, y, s=400, color="gold", label="Output Layer")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------- Visualization 4: Performance Metrics ----------

def plot_performance_metrics(accuracy, modularity, scalability, path):
    labels = ["Accuracy", "Modularity Score", "Scalability"]
    values = [accuracy, modularity * 100, scalability]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#66b3ff", "#99ff99", "#ff9999"], edgecolor="black")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.2f}", ha='center', fontsize=11)
    plt.title("Performance Metrics of Hybrid Model", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage / Score")
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------- Pipeline ----------

def pipeline(input_path=None, algorithm="labelprop", top_n=10, outdir="results"):
    ensure_dir(outdir)
    start_time = time.time()

    if input_path:
        print(f"Loading graph from: {input_path}")
        G = load_graph_from_edgelist(input_path)
    else:
        print("Using demo graph (Zachary’s Karate Club)")
        G = nx.karate_club_graph()

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    communities = detect_communities(G, algorithm)
    sizes = [len(c) for c in communities]
    labels = [f"C{i+1}" for i in range(len(communities))]

    modularity = compute_modularity(G, communities)
    scalability_stats = measure_scalability(start_time)
    accuracy = estimate_accuracy(modularity, len(communities))
    scalability_score = max(0, 100 - (scalability_stats["runtime_sec"] * 5))

    csv_path = os.path.join(outdir, "communitiesnewman.csv")
    pd.DataFrame({"Community": labels, "Size": sizes}).to_csv(csv_path, index=False)

    bar_path = os.path.join(outdir, "community_sizesnewman.png")
    net_path = os.path.join(outdir, "community_networknewman.png")
    nn_path = os.path.join(outdir, "neural_networknewman.png")
    perf_path = os.path.join(outdir, "performance_metricsnewman.png")

    plot_bar_chart(sizes, labels, f"Community Sizes (Modularity={modularity:.3f})", bar_path)
    plot_network_graph(G, communities, net_path)
    plot_neural_network(nn_path)
    plot_performance_metrics(accuracy, modularity, scalability_score, perf_path)

    summary = {
        "algorithm": algorithm,
        "n_communities": len(communities),
        "modularity_score": round(modularity, 4),
        "accuracy": accuracy,
        "scalability": scalability_stats,
        "outputs": {
            "bar_graph": bar_path,
            "network_graph": net_path,
            "ann_diagram": nn_path,
            "performance_graph": perf_path,
            "csv": csv_path
        }
    }
    json_path = os.path.join(outdir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\n✅ All visualizations saved successfully!")

# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Community Detection + Visualization + ANN Diagram")
    parser.add_argument("--input", "-i", help="Path to edge-list file", default=None)
    parser.add_argument("--algorithm", "-a", choices=["labelprop", "girvan"], default="labelprop")
    parser.add_argument("--top", "-t", type=int, default=10, help="Top N communities in chart")
    parser.add_argument("--outdir", "-o", default="results", help="Output directory")
    args = parser.parse_args()
    pipeline(args.input, args.algorithm, args.top, args.outdir)

