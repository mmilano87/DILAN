#!/usr/bin/env python3
"""
DILAN: Differential Local Alignment of Networks
================================================

DILAN is a rewiring-aware pairwise local alignment algorithm for differential
biological networks. Given two weighted differential networks and node-level
rewiring scores, the algorithm identifies compact conserved modules showing
similar rewiring behavior across the two networks.

Input files
-----------
1. Edge file for network 1: whitespace-separated file with at least three columns
   Node1 Node2 Weight
2. Rewiring-score file for network 1: whitespace-separated file with columns
   Node Rewiring_score, or a headerless two-column file
3. Edge file for network 2
4. Rewiring-score file for network 2

Main outputs
------------
- dilan_seed_list.txt
- rewiring_alignment_graph_edges.txt
- dilan_modules.txt
- module_statistics.txt
- summary_alignment.txt
- final_alignment_metrics_report.txt

Author: Marianna Milano
Repository: https://github.com/mmilano87/DILAN
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

Node = str
EdgeKey = Tuple[Node, Node]
ScoreDict = Dict[Node, float]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class DILANConfig:
    """Container for all DILAN hyperparameters."""

    top_seeds: int = 100

    min_node_rewiring_similarity: float = 0.30
    min_edge_similarity: float = 0.30
    min_alignment_edge_weight: float = 0.30

    min_module_size: int = 5
    max_module_size: int = 80

    min_module_density: float = 0.02
    min_module_score: float = 0.30

    max_overlap_allowed: float = 0.85

    min_gain: float = 0.01
    patience: int = 3
    allow_density_decrease: bool = False

    alpha_edge: float = 0.40
    beta_node: float = 0.30
    gamma_rmc: float = 0.30

    candidate_pool_size: int = 50


@dataclass
class DILANInputs:
    """Input paths and names for one pairwise DILAN run."""

    network_1_name: str
    network_2_name: str
    network_1_edges: Path
    network_1_scores: Path
    network_2_edges: Path
    network_2_scores: Path
    output_dir: Path


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def load_edges(edge_file: Path) -> nx.Graph:
    """
    Load a weighted undirected differential network.

    The expected format is a whitespace-separated file with at least three
    columns: Node1, Node2, and Weight. Additional columns are ignored.

    Parameters
    ----------
    edge_file : Path
        Path to the differential edge list.

    Returns
    -------
    networkx.Graph
        Weighted undirected graph.
    """

    df = pd.read_csv(edge_file, sep=r"\s+", header=None)

    if df.shape[1] < 3:
        raise ValueError(
            f"Edge file {edge_file} must contain at least three columns: "
            "Node1 Node2 Weight."
        )

    df = df.iloc[:, :3]
    df.columns = ["Node1", "Node2", "Weight"]

    df = df[df["Node1"] != df["Node2"]]
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Weight"])

    graph = nx.Graph()

    for _, row in df.iterrows():
        graph.add_edge(str(row["Node1"]), str(row["Node2"]), weight=float(row["Weight"]))

    return graph



def load_rewiring_scores(score_file: Path) -> ScoreDict:
    """
    Load node-level rewiring scores.

    The function supports both a headered file with columns `Node` and
    `Rewiring_score`, and a headerless two-column file.

    Parameters
    ----------
    score_file : Path
        Path to the rewiring-score file.

    Returns
    -------
    dict
        Mapping node identifier -> rewiring score.
    """

    df = pd.read_csv(score_file, sep=r"\s+")

    if "Node" not in df.columns or "Rewiring_score" not in df.columns:
        df = pd.read_csv(score_file, sep=r"\s+", header=None)
        if df.shape[1] < 2:
            raise ValueError(
                f"Score file {score_file} must contain at least two columns: "
                "Node Rewiring_score."
            )
        df = df.iloc[:, :2]
        df.columns = ["Node", "Rewiring_score"]

    df["Rewiring_score"] = pd.to_numeric(df["Rewiring_score"], errors="coerce")
    df = df.dropna(subset=["Rewiring_score"])

    return dict(zip(df["Node"].astype(str), df["Rewiring_score"].astype(float)))



def normalize_scores(scores: ScoreDict) -> ScoreDict:
    """
    Normalize rewiring scores by the maximum observed score.

    Parameters
    ----------
    scores : dict
        Raw node-level rewiring scores.

    Returns
    -------
    dict
        Normalized rewiring scores in the interval [0, 1], when the maximum
        score is positive.
    """

    if not scores:
        return {}

    max_value = max(scores.values())

    if max_value == 0:
        return {node: 0.0 for node in scores}

    return {node: value / max_value for node, value in scores.items()}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def edge_key(u: Node, v: Node) -> EdgeKey:
    """Return a canonical key for an undirected edge."""

    return tuple(sorted((u, v)))



def get_edge_dict(graph: nx.Graph) -> Dict[EdgeKey, float]:
    """Convert a weighted graph into a dictionary indexed by canonical edges."""

    return {
        edge_key(str(u), str(v)): float(data.get("weight", 1.0))
        for u, v, data in graph.edges(data=True)
    }



def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, using 0 when the denominator is zero."""

    return 0.0 if denominator == 0 else numerator / denominator



def exp_similarity(a: float, b: float) -> float:
    """
    Compute an exponential similarity between two scalar values.

    The value is equal to 1 for identical values and decreases exponentially as
    the absolute difference increases.
    """

    return math.exp(-abs(a - b))



def f1_score(precision: float, recall: float) -> float:
    """Compute the F1 score from precision and recall."""

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -----------------------------------------------------------------------------
# Step 1: seed selection
# -----------------------------------------------------------------------------


def build_seed_list(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    scores_1: ScoreDict,
    scores_2: ScoreDict,
    config: DILANConfig,
) -> Tuple[List[Node], pd.DataFrame]:
    """
    Identify and rank candidate seed nodes shared by the two networks.

    A seed is selected when its rewiring similarity across the two networks is
    above the configured threshold. Seeds are ranked by the product between
    average rewiring activity and rewiring similarity.
    """

    shared_nodes = sorted(set(graph_1.nodes()).intersection(set(graph_2.nodes())))
    seed_rows = []

    for node in shared_nodes:
        r1 = scores_1.get(node, 0.0)
        r2 = scores_2.get(node, 0.0)

        rewiring_similarity = exp_similarity(r1, r2)
        average_rewiring = (r1 + r2) / 2.0
        seed_score = average_rewiring * rewiring_similarity

        if rewiring_similarity >= config.min_node_rewiring_similarity:
            seed_rows.append(
                {
                    "Seed": node,
                    "Rewiring_score_G1": r1,
                    "Rewiring_score_G2": r2,
                    "Rewiring_similarity": rewiring_similarity,
                    "Average_rewiring": average_rewiring,
                    "Seed_score": seed_score,
                }
            )

    seed_df = pd.DataFrame(seed_rows)

    if seed_df.empty:
        return [], seed_df

    seed_df = seed_df.sort_values(by="Seed_score", ascending=False)
    seeds = seed_df["Seed"].head(config.top_seeds).tolist()

    return seeds, seed_df


# -----------------------------------------------------------------------------
# Step 2: local alignment graph construction
# -----------------------------------------------------------------------------


def build_local_alignment_graph(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    scores_1: ScoreDict,
    scores_2: ScoreDict,
    seed: Node,
    config: DILANConfig,
) -> nx.Graph:
    """
    Construct a seed-centered local alignment graph.

    Candidate nodes are obtained from the union of the seed neighborhoods in the
    two differential networks and restricted to shared nodes. Candidate edges are
    weighted by integrating edge similarity, node rewiring similarity, and edge
    conservation.
    """

    edges_1 = get_edge_dict(graph_1)
    edges_2 = get_edge_dict(graph_2)

    shared_nodes = set(graph_1.nodes()).intersection(set(graph_2.nodes()))
    candidate_nodes = {seed}

    if seed in graph_1:
        candidate_nodes.update(graph_1.neighbors(seed))

    if seed in graph_2:
        candidate_nodes.update(graph_2.neighbors(seed))

    candidate_nodes = candidate_nodes.intersection(shared_nodes)
    alignment_graph = nx.Graph()

    for node in candidate_nodes:
        r1 = scores_1.get(node, 0.0)
        r2 = scores_2.get(node, 0.0)
        node_similarity = exp_similarity(r1, r2)

        if node_similarity >= config.min_node_rewiring_similarity:
            alignment_graph.add_node(
                node,
                node_g1=node,
                node_g2=node,
                rewiring_score_g1=r1,
                rewiring_score_g2=r2,
                node_rewiring_similarity=node_similarity,
            )

    candidate_edges: Set[EdgeKey] = set()

    for u, v in set(edges_1.keys()).union(set(edges_2.keys())):
        if u in alignment_graph.nodes() and v in alignment_graph.nodes():
            candidate_edges.add((u, v))

    for u, v in candidate_edges:
        w1 = edges_1.get((u, v), 0.0)
        w2 = edges_2.get((u, v), 0.0)

        edge_present_g1 = (u, v) in edges_1
        edge_present_g2 = (u, v) in edges_2

        edge_similarity = exp_similarity(w1, w2)

        node_similarity = (
            alignment_graph.nodes[u]["node_rewiring_similarity"]
            + alignment_graph.nodes[v]["node_rewiring_similarity"]
        ) / 2.0

        edge_conservation = 1.0 if edge_present_g1 and edge_present_g2 else 0.5

        alignment_weight = (
            0.50 * edge_similarity
            + 0.30 * node_similarity
            + 0.20 * edge_conservation
        )

        if (
            edge_similarity >= config.min_edge_similarity
            and alignment_weight >= config.min_alignment_edge_weight
        ):
            alignment_graph.add_edge(
                u,
                v,
                weight=alignment_weight,
                edge_similarity=edge_similarity,
                node_similarity=node_similarity,
                edge_conservation=edge_conservation,
                weight_g1=w1,
                weight_g2=w2,
                edge_present_g1=edge_present_g1,
                edge_present_g2=edge_present_g2,
            )

    return alignment_graph


# -----------------------------------------------------------------------------
# Step 3: greedy module extraction
# -----------------------------------------------------------------------------


def compute_alignment_rmc(alignment_graph: nx.Graph, nodes: Iterable[Node]) -> float:
    """Compute the average alignment weight inside a node-induced subgraph."""

    subgraph = alignment_graph.subgraph(nodes)

    if subgraph.number_of_edges() == 0:
        return 0.0

    weights = [data.get("weight", 0.0) for _, _, data in subgraph.edges(data=True)]
    return sum(weights) / len(weights)



def compute_module_density(alignment_graph: nx.Graph, nodes: Iterable[Node]) -> float:
    """Compute the density of a node-induced subgraph."""

    subgraph = alignment_graph.subgraph(nodes)

    if subgraph.number_of_nodes() <= 1:
        return 0.0

    return nx.density(subgraph)



def candidate_connection_score(
    alignment_graph: nx.Graph,
    candidate: Node,
    module_nodes: Set[Node],
) -> float:
    """Compute the average connection strength between a candidate and a module."""

    weights = []

    for node in module_nodes:
        if alignment_graph.has_edge(candidate, node):
            weights.append(alignment_graph[candidate][node].get("weight", 0.0))

    if not weights:
        return 0.0

    return sum(weights) / len(weights)



def get_candidate_neighbors(alignment_graph: nx.Graph, module_nodes: Set[Node]) -> Set[Node]:
    """Return nodes adjacent to the current module and not already included."""

    candidates = set()

    for node in module_nodes:
        if node not in alignment_graph:
            continue

        for neighbor in alignment_graph.neighbors(node):
            if neighbor not in module_nodes:
                candidates.add(neighbor)

    return candidates



def extract_dilan_module(
    alignment_graph: nx.Graph,
    seed: Node,
    config: DILANConfig,
) -> Optional[nx.Graph]:
    """
    Extract one DILAN module from a seed-centered local alignment graph.

    The module is initialized with the seed and its strongest neighbors. It is
    then expanded greedily by adding the candidate node that maximizes the gain
    in the combined module quality score.
    """

    if seed not in alignment_graph:
        return None

    module_nodes: Set[Node] = {seed}
    no_gain_steps = 0

    initial_neighbors = sorted(
        [
            (neighbor, alignment_graph[seed][neighbor].get("weight", 0.0))
            for neighbor in alignment_graph.neighbors(seed)
            if alignment_graph[seed][neighbor].get("weight", 0.0)
            >= config.min_alignment_edge_weight
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    for neighbor, _ in initial_neighbors:
        if len(module_nodes) >= config.min_module_size:
            break
        module_nodes.add(neighbor)

    current_rmc = compute_alignment_rmc(alignment_graph, module_nodes)
    current_density = compute_module_density(alignment_graph, module_nodes)
    current_combined = 0.70 * current_rmc + 0.30 * current_density

    while len(module_nodes) < config.max_module_size:
        candidates = get_candidate_neighbors(alignment_graph, module_nodes)
        candidate_scores = []

        for candidate in candidates:
            connection_score = candidate_connection_score(
                alignment_graph,
                candidate,
                module_nodes,
            )
            if connection_score >= config.min_alignment_edge_weight:
                candidate_scores.append((candidate, connection_score))

        if not candidate_scores:
            break

        candidate_scores = sorted(candidate_scores, key=lambda item: item[1], reverse=True)

        best_candidate = None
        best_gain = 0.0
        best_rmc = current_rmc
        best_density = current_density
        best_combined = current_combined

        for candidate, _ in candidate_scores[: config.candidate_pool_size]:
            test_nodes = set(module_nodes)
            test_nodes.add(candidate)

            new_rmc = compute_alignment_rmc(alignment_graph, test_nodes)
            new_density = compute_module_density(alignment_graph, test_nodes)
            new_combined = 0.70 * new_rmc + 0.30 * new_density
            gain = new_combined - current_combined

            if not config.allow_density_decrease and new_density < current_density:
                continue

            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate
                best_rmc = new_rmc
                best_density = new_density
                best_combined = new_combined

        if best_candidate is not None and best_gain >= config.min_gain:
            module_nodes.add(best_candidate)
            current_rmc = best_rmc
            current_density = best_density
            current_combined = best_combined
            no_gain_steps = 0
        else:
            no_gain_steps += 1
            if no_gain_steps >= config.patience:
                break
            break

    module = alignment_graph.subgraph(module_nodes).copy()

    if module.number_of_nodes() < config.min_module_size:
        return None

    if module.number_of_edges() == 0:
        return None

    density = nx.density(module) if module.number_of_nodes() > 1 else 0.0

    if density < config.min_module_density:
        return None

    return module


# -----------------------------------------------------------------------------
# Step 4: module metrics
# -----------------------------------------------------------------------------


def compute_classical_alignment_metrics(
    module: nx.Graph,
    graph_1: nx.Graph,
    graph_2: nx.Graph,
) -> Tuple[float, float, float, float]:
    """
    Compute classical node-alignment metrics in the shared-node scenario.

    These values are included for diagnostic purposes. Since DILAN operates on
    shared node identifiers in this implementation, each aligned pair is of the
    form (node, node).
    """

    nodes = set(module.nodes())

    aligned_pairs = {(node, node) for node in nodes}
    true_pairs = {(node, node) for node in set(graph_1.nodes()).intersection(set(graph_2.nodes()))}

    correct_pairs = aligned_pairs.intersection(true_pairs)

    tp = len(correct_pairs)
    fp = len(aligned_pairs) - tp
    fn = len(true_pairs) - tp

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = f1_score(precision, recall)
    nc = safe_divide(tp, len(aligned_pairs))

    return precision, recall, f1, nc



def compute_module_metrics(
    module: nx.Graph,
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    config: DILANConfig,
) -> Optional[Dict[str, float]]:
    """Compute classical and rewiring-aware metrics for one candidate module."""

    nodes = set(module.nodes())

    edges_1 = get_edge_dict(graph_1)
    edges_2 = get_edge_dict(graph_2)

    possible_edges = {
        edge
        for edge in set(edges_1.keys()).union(set(edges_2.keys()))
        if edge[0] in nodes and edge[1] in nodes
    }

    conserved_edges = {edge for edge in possible_edges if edge in edges_1 and edge in edges_2}

    precision, recall, f1, nc = compute_classical_alignment_metrics(module, graph_1, graph_2)

    node_rewiring_values = [
        module.nodes[node].get("node_rewiring_similarity", 0.0)
        for node in nodes
    ]
    rnc = sum(node_rewiring_values) / len(node_rewiring_values) if node_rewiring_values else 0.0

    edge_sim_values = [
        data.get("edge_similarity", 0.0)
        for _, _, data in module.edges(data=True)
    ]
    rcs = sum(edge_sim_values) / len(edge_sim_values) if edge_sim_values else 0.0

    dep = safe_divide(len(conserved_edges), len(possible_edges))
    rmc = compute_alignment_rmc(module, module.nodes())
    density = nx.density(module) if module.number_of_nodes() > 1 else 0.0

    score = config.alpha_edge * rcs + config.beta_node * rnc + config.gamma_rmc * rmc

    if score < config.min_module_score:
        return None

    return {
        "size": module.number_of_nodes(),
        "alignment_edges": module.number_of_edges(),
        "possible_edges": len(possible_edges),
        "conserved_edges": len(conserved_edges),
        "density": density,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nc": nc,
        "edge_conservation": dep,
        "gs3": dep,
        "dep": dep,
        "rcs": rcs,
        "rnc": rnc,
        "rmc": rmc,
        "score": score,
    }


# -----------------------------------------------------------------------------
# Step 5: redundancy filtering
# -----------------------------------------------------------------------------


def module_overlap_nodes(module_1: nx.Graph, module_2: nx.Graph) -> float:
    """Compute node overlap relative to the smaller module."""

    nodes_1 = set(module_1.nodes())
    nodes_2 = set(module_2.nodes())

    if len(nodes_1) == 0 or len(nodes_2) == 0:
        return 0.0

    return len(nodes_1.intersection(nodes_2)) / min(len(nodes_1), len(nodes_2))



def filter_redundant_modules(
    module_records: List[Dict],
    config: DILANConfig,
) -> List[Dict]:
    """
    Remove highly overlapping modules while keeping the best-scoring solutions."""

    module_records = sorted(
        module_records,
        key=lambda record: (
            record["metrics"]["score"],
            record["metrics"]["rmc"],
            record["metrics"]["size"],
        ),
        reverse=True,
    )

    selected = []

    for record in module_records:
        redundant = False

        for kept in selected:
            overlap = module_overlap_nodes(record["module"], kept["module"])

            if overlap > config.max_overlap_allowed:
                redundant = True
                break

        if not redundant:
            selected.append(record)

    return selected


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def compute_global_averages(module_records: List[Dict]) -> Dict[str, float]:
    """Compute average metric values across all final modules."""

    if not module_records:
        return {}

    metric_names = [
        "size",
        "alignment_edges",
        "possible_edges",
        "conserved_edges",
        "density",
        "precision",
        "recall",
        "f1",
        "nc",
        "edge_conservation",
        "gs3",
        "dep",
        "rcs",
        "rnc",
        "rmc",
        "score",
    ]

    averages = {}

    for metric in metric_names:
        averages[f"Average_{metric}"] = (
            sum(record["metrics"][metric] for record in module_records) / len(module_records)
        )

    return averages



def write_seed_list(seed_df: pd.DataFrame, output_path: Path) -> None:
    """Write the ranked seed list."""

    seed_df.to_csv(output_path, sep="\t", index=False)



def write_alignment_graph(all_alignment_edges: List[Dict], output_path: Path) -> None:
    """Write all generated local alignment graph edges."""

    pd.DataFrame(all_alignment_edges).to_csv(output_path, sep="\t", index=False)



def write_modules(module_records: List[Dict], output_path: Path) -> None:
    """Write a human-readable description of the final DILAN modules."""

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("DILAN LOCAL ALIGNMENT MODULES\n")
        handle.write("=============================\n\n")

        if not module_records:
            handle.write("No modules found.\n")
            return

        for i, record in enumerate(module_records, start=1):
            module = record["module"]
            metrics = record["metrics"]

            handle.write(f"MODULE_{i}\n")
            handle.write("-----------------------\n")
            handle.write(f"Seed: {record['seed']}\n")
            handle.write(f"UniRef90_count: {metrics['size']}\n")
            handle.write(f"Alignment_edges: {metrics['alignment_edges']}\n")
            handle.write(f"Possible_edges: {metrics['possible_edges']}\n")
            handle.write(f"Conserved_edges: {metrics['conserved_edges']}\n")
            handle.write(f"Density: {metrics['density']:.6f}\n\n")

            handle.write("Classical_alignment_metrics:\n")
            handle.write(f"Precision: {metrics['precision']:.6f}\n")
            handle.write(f"Recall: {metrics['recall']:.6f}\n")
            handle.write(f"F1: {metrics['f1']:.6f}\n")
            handle.write(f"NC: {metrics['nc']:.6f}\n")
            handle.write(f"GS3: {metrics['gs3']:.6f}\n\n")

            handle.write("Rewiring_aware_metrics:\n")
            handle.write(f"DEP: {metrics['dep']:.6f}\n")
            handle.write(f"RCS: {metrics['rcs']:.6f}\n")
            handle.write(f"RNC: {metrics['rnc']:.6f}\n")
            handle.write(f"RMC: {metrics['rmc']:.6f}\n")
            handle.write(f"Score: {metrics['score']:.6f}\n\n")

            handle.write("Shared_UniRef90:\n")
            for node in sorted(module.nodes()):
                handle.write(f"{node}\n")

            handle.write("\nAlignment_edges:\n")
            for u, v, data in sorted(module.edges(data=True)):
                handle.write(
                    f"{u}\t{v}\t"
                    f"{data.get('weight', 0.0):.6f}\t"
                    f"{data.get('edge_similarity', 0.0):.6f}\n"
                )

            handle.write("\n\n")



def write_statistics(module_records: List[Dict], output_path: Path) -> None:
    """Write one row of metrics for each final module."""

    rows = []

    for i, record in enumerate(module_records, start=1):
        metrics = record["metrics"]

        rows.append(
            {
                "Module_ID": f"MODULE_{i}",
                "Seed": record["seed"],
                "UniRef90_count": metrics["size"],
                "Alignment_edges": metrics["alignment_edges"],
                "Possible_edges": metrics["possible_edges"],
                "Conserved_edges": metrics["conserved_edges"],
                "Density": metrics["density"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "NC": metrics["nc"],
                "GS3": metrics["gs3"],
                "DEP": metrics["dep"],
                "RCS": metrics["rcs"],
                "RNC": metrics["rnc"],
                "RMC": metrics["rmc"],
                "Score": metrics["score"],
            }
        )

    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)



def write_final_report(module_records: List[Dict], output_path: Path) -> None:
    """Write a text report containing all final module metrics and global averages."""

    averages = compute_global_averages(module_records)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("DILAN FINAL ALIGNMENT METRICS REPORT\n")
        handle.write("====================================\n\n")

        if not module_records:
            handle.write("No modules found.\n")
            return

        for i, record in enumerate(module_records, start=1):
            metrics = record["metrics"]

            handle.write(f"MODULE_{i}\n")
            handle.write("-----------------------\n")
            handle.write(f"Seed: {record['seed']}\n")
            handle.write(f"UniRef90_count: {metrics['size']}\n")
            handle.write(f"Alignment_edges: {metrics['alignment_edges']}\n")
            handle.write(f"Density: {metrics['density']:.6f}\n")
            handle.write(f"Precision: {metrics['precision']:.6f}\n")
            handle.write(f"Recall: {metrics['recall']:.6f}\n")
            handle.write(f"F1: {metrics['f1']:.6f}\n")
            handle.write(f"NC: {metrics['nc']:.6f}\n")
            handle.write(f"GS3: {metrics['gs3']:.6f}\n")
            handle.write(f"DEP: {metrics['dep']:.6f}\n")
            handle.write(f"RCS: {metrics['rcs']:.6f}\n")
            handle.write(f"RNC: {metrics['rnc']:.6f}\n")
            handle.write(f"RMC: {metrics['rmc']:.6f}\n")
            handle.write(f"Score: {metrics['score']:.6f}\n\n")

        handle.write("\nGLOBAL ALIGNMENT PERFORMANCE\n")
        handle.write("============================\n")
        handle.write(f"Modules_found: {len(module_records)}\n")

        for key, value in averages.items():
            handle.write(f"{key}: {value:.6f}\n")



def write_summary(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    seeds: Sequence[Node],
    raw_records: List[Dict],
    final_records: List[Dict],
    all_alignment_edges: List[Dict],
    inputs: DILANInputs,
    config: DILANConfig,
    output_path: Path,
) -> None:
    """Write a one-row summary of the DILAN run."""

    averages = compute_global_averages(final_records)

    summary = {
        "Network_1": inputs.network_1_name,
        "Network_2": inputs.network_2_name,
        "G1_nodes": graph_1.number_of_nodes(),
        "G1_edges": graph_1.number_of_edges(),
        "G2_nodes": graph_2.number_of_nodes(),
        "G2_edges": graph_2.number_of_edges(),
        "Shared_nodes": len(set(graph_1.nodes()).intersection(set(graph_2.nodes()))),
        "Selected_seeds": len(seeds),
        "Raw_modules": len(raw_records),
        "Final_modules": len(final_records),
        "Total_alignment_edges_generated": len(all_alignment_edges),
        "TOP_SEEDS": config.top_seeds,
        "MIN_NODE_REWIRING_SIMILARITY": config.min_node_rewiring_similarity,
        "MIN_EDGE_SIMILARITY": config.min_edge_similarity,
        "MIN_ALIGNMENT_EDGE_WEIGHT": config.min_alignment_edge_weight,
        "MIN_MODULE_SIZE": config.min_module_size,
        "MAX_MODULE_SIZE": config.max_module_size,
        "MIN_MODULE_DENSITY": config.min_module_density,
        "MIN_MODULE_SCORE": config.min_module_score,
        "MIN_GAIN": config.min_gain,
        "PATIENCE": config.patience,
        "ALLOW_DENSITY_DECREASE": config.allow_density_decrease,
        "MAX_OVERLAP_ALLOWED": config.max_overlap_allowed,
    }

    summary.update(averages)

    pd.DataFrame([summary]).to_csv(output_path, sep="\t", index=False)


# -----------------------------------------------------------------------------
# Main DILAN execution
# -----------------------------------------------------------------------------


def run_dilan(inputs: DILANInputs, config: DILANConfig) -> List[Dict]:
    """
    Run DILAN on a pair of differential networks.

    Parameters
    ----------
    inputs : DILANInputs
        Input paths, network names, and output directory.
    config : DILANConfig
        DILAN hyperparameters.

    Returns
    -------
    list of dict
        Final non-redundant module records.
    """

    inputs.output_dir.mkdir(parents=True, exist_ok=True)

    output_seed_list = inputs.output_dir / "dilan_seed_list.txt"
    output_alignment_graph = inputs.output_dir / "rewiring_alignment_graph_edges.txt"
    output_modules = inputs.output_dir / "dilan_modules.txt"
    output_statistics = inputs.output_dir / "module_statistics.txt"
    output_summary = inputs.output_dir / "summary_alignment.txt"
    output_final_report = inputs.output_dir / "final_alignment_metrics_report.txt"

    print("Loading differential networks...")

    graph_1 = load_edges(inputs.network_1_edges)
    graph_2 = load_edges(inputs.network_2_edges)

    scores_1 = normalize_scores(load_rewiring_scores(inputs.network_1_scores))
    scores_2 = normalize_scores(load_rewiring_scores(inputs.network_2_scores))

    print(f"{inputs.network_1_name}: {graph_1.number_of_nodes()} nodes, {graph_1.number_of_edges()} edges")
    print(f"{inputs.network_2_name}: {graph_2.number_of_nodes()} nodes, {graph_2.number_of_edges()} edges")

    shared_nodes = set(graph_1.nodes()).intersection(set(graph_2.nodes()))
    print(f"Shared nodes: {len(shared_nodes)}")

    print("Building seed list...")
    seeds, seed_df = build_seed_list(graph_1, graph_2, scores_1, scores_2, config)
    write_seed_list(seed_df, output_seed_list)
    print(f"Selected seeds: {len(seeds)}")

    raw_records: List[Dict] = []
    all_alignment_edges: List[Dict] = []

    print("Running DILAN local alignment...")

    for i, seed in enumerate(seeds, start=1):
        if i % 10 == 0:
            print(f"Processing seed {i}/{len(seeds)}")

        local_alignment_graph = build_local_alignment_graph(
            graph_1,
            graph_2,
            scores_1,
            scores_2,
            seed,
            config,
        )

        for u, v, data in local_alignment_graph.edges(data=True):
            all_alignment_edges.append(
                {
                    "Seed": seed,
                    "Node1": u,
                    "Node2": v,
                    "Alignment_weight": data.get("weight", 0.0),
                    "Edge_similarity": data.get("edge_similarity", 0.0),
                    "Node_similarity": data.get("node_similarity", 0.0),
                    "Edge_conservation": data.get("edge_conservation", 0.0),
                    "Weight_G1": data.get("weight_g1", 0.0),
                    "Weight_G2": data.get("weight_g2", 0.0),
                    "Edge_present_G1": data.get("edge_present_g1", False),
                    "Edge_present_G2": data.get("edge_present_g2", False),
                }
            )

        module = extract_dilan_module(local_alignment_graph, seed, config)

        if module is None:
            continue

        metrics = compute_module_metrics(module, graph_1, graph_2, config)

        if metrics is None:
            continue

        raw_records.append(
            {
                "seed": seed,
                "module": module,
                "metrics": metrics,
            }
        )

    final_records = filter_redundant_modules(raw_records, config)

    write_alignment_graph(all_alignment_edges, output_alignment_graph)
    write_modules(final_records, output_modules)
    write_statistics(final_records, output_statistics)
    write_final_report(final_records, output_final_report)
    write_summary(
        graph_1,
        graph_2,
        seeds,
        raw_records,
        final_records,
        all_alignment_edges,
        inputs,
        config,
        output_summary,
    )

    print("DILAN completed.")
    print(f"Raw modules: {len(raw_records)}")
    print(f"Final non-redundant modules: {len(final_records)}")
    print(f"Results saved in: {inputs.output_dir}")

    return final_records


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="DILAN: Differential Local Alignment of Networks"
    )

    parser.add_argument("--network-1-name", required=True, help="Name of the first differential network.")
    parser.add_argument("--network-2-name", required=True, help="Name of the second differential network.")

    parser.add_argument("--network-1-edges", required=True, type=Path, help="Edge file for network 1.")
    parser.add_argument("--network-1-scores", required=True, type=Path, help="Rewiring-score file for network 1.")

    parser.add_argument("--network-2-edges", required=True, type=Path, help="Edge file for network 2.")
    parser.add_argument("--network-2-scores", required=True, type=Path, help="Rewiring-score file for network 2.")

    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")

    parser.add_argument("--top-seeds", type=int, default=100, help="Maximum number of seed nodes.")
    parser.add_argument("--min-module-size", type=int, default=5, help="Minimum module size.")
    parser.add_argument("--max-module-size", type=int, default=80, help="Maximum module size.")
    parser.add_argument("--min-module-score", type=float, default=0.30, help="Minimum DILAN module score.")
    parser.add_argument("--max-overlap", type=float, default=0.85, help="Maximum allowed module overlap.")

    return parser.parse_args()



def main() -> None:
    """Entry point for command-line execution."""

    args = parse_arguments()

    inputs = DILANInputs(
        network_1_name=args.network_1_name,
        network_2_name=args.network_2_name,
        network_1_edges=args.network_1_edges,
        network_1_scores=args.network_1_scores,
        network_2_edges=args.network_2_edges,
        network_2_scores=args.network_2_scores,
        output_dir=args.output_dir,
    )

    config = DILANConfig(
        top_seeds=args.top_seeds,
        min_module_size=args.min_module_size,
        max_module_size=args.max_module_size,
        min_module_score=args.min_module_score,
        max_overlap_allowed=args.max_overlap,
    )

    run_dilan(inputs, config)


if __name__ == "__main__":
    main()
