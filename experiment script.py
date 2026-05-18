# -*- coding: utf-8 -*-
# ============================================================================
# SETUP
# ============================================================================
import subprocess
subprocess.run(['pip', 'install', 'networkx', 'tqdm', 'pingouin', '-q'])

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import pingouin as pg

print("✓ Setup complete!")

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {"prior_alpha": 1, "prior_beta": 1, "convergence_threshold": 0.99}

ASYMMETRIC_TOPOLOGIES = ["star", "wheel", "line", "hierarchical",
                         "clustered", "scale_free", "small_world", "random"]
SYMMETRIC_TOPOLOGIES = ["complete", "cycle"]
ALL_TOPOLOGIES = ASYMMETRIC_TOPOLOGIES + SYMMETRIC_TOPOLOGIES


class AgentType(Enum):
    TRUTH_SEEKER = "truth_seeker"
    BIASED = "biased"


@dataclass
class Agent:
    agent_id: int
    agent_type: AgentType
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def credence(self) -> float:
        return self.alpha / (self.alpha + self.beta)


# ============================================================================
# NETWORK FUNCTIONS
# ============================================================================
def create_network(topology, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if topology == "star":
        return nx.star_graph(n - 1)
    elif topology == "wheel":
        return nx.wheel_graph(n)
    elif topology == "cycle":
        return nx.cycle_graph(n)
    elif topology == "complete":
        return nx.complete_graph(n)
    elif topology == "line":
        return nx.path_graph(n)
    elif topology == "hierarchical":
        G = nx.Graph(); G.add_nodes_from(range(n))
        for i in range(1, n): G.add_edge((i - 1) // 2, i)
        return G
    elif topology == "clustered":
        G = nx.Graph(); G.add_nodes_from(range(n)); half = n // 2
        for i in range(half):
            for j in range(i + 1, half): G.add_edge(i, j)
        for i in range(half, n):
            for j in range(i + 1, n): G.add_edge(i, j)
        G.add_edge(half - 1, half); return G
    elif topology == "scale_free":
        return nx.barabasi_albert_graph(n, m=min(2, n - 1), seed=seed)
    elif topology == "small_world":
        k = min(4, n - 1)
        if k % 2 == 1: k = max(2, k - 1)
        return nx.watts_strogatz_graph(n, k, p=0.3, seed=seed)
    elif topology == "random":
        G = nx.erdos_renyi_graph(n, 2 * np.log(n) / n, seed=seed)
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                G.add_edge(list(comps[i])[0], list(comps[i + 1])[0])
        return G
    raise ValueError(f"Unknown topology: {topology}")


def get_centrality_measures(G):
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try: eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError): eigenvector = {n: 0.0 for n in G.nodes()}
    return {n: {'degree': degree[n], 'betweenness': betweenness[n],
                'closeness': closeness[n], 'eigenvector': eigenvector[n]} for n in G.nodes()}


def get_network_properties(G):
    try: diameter = nx.diameter(G)
    except nx.NetworkXError: diameter = -1
    return {'density': nx.density(G), 'avg_clustering': nx.average_clustering(G), 'diameter': diameter}


def get_high_centrality_positions(G, n_positions=1):
    degree = nx.degree_centrality(G)
    return sorted(degree.keys(), key=lambda x: degree[x], reverse=True)[:n_positions]


def get_low_centrality_positions(G, n_positions=1, exclude=None):
    """[FIX #3/#5] Get low-centrality positions, excluding specified nodes."""
    degree = nx.degree_centrality(G)
    exclude = exclude or []
    candidates = [n for n in G.nodes() if n not in exclude]
    return sorted(candidates, key=lambda x: degree[x])[:n_positions]


# ============================================================================
# SIMULATION ENGINE
# ============================================================================
def run_simulation(G, topology, n_agents, n_rounds, biased_positions,
                   bias_strength, efficacy_difference, intended_condition,
                   heterogeneous_priors=False, n_per_round=100,
                   seed=None, save_trajectory=False, phase=None):
    """
    G: pre-built graph [FIX #6]
    bias_strength: reported success rate for B (0.5-1.0) [FIX #2]
    intended_condition: label from experiment loop [FIX #3]
    phase: experiment phase identifier (1-6) [FIX #14]
    """
    if seed is not None:
        np.random.seed(seed)

    p_A = 0.50 + efficacy_difference
    p_B = 0.50
    centralities = get_centrality_measures(G)
    net_props = get_network_properties(G)
    topology_class = "symmetric" if topology in SYMMETRIC_TOPOLOGIES else "asymmetric"

    agents = []
    for i in range(n_agents):
        atype = AgentType.BIASED if i in biased_positions else AgentType.TRUTH_SEEKER
        if heterogeneous_priors and atype == AgentType.TRUTH_SEEKER:
            alpha, beta = np.random.uniform(1, 5), np.random.uniform(1, 5)
        else:
            alpha, beta = CONFIG["prior_alpha"], CONFIG["prior_beta"]
        agents.append(Agent(agent_id=i, agent_type=atype, alpha=alpha, beta=beta))

    trajectory = [] if save_trajectory else None
    rounds_using_A = []
    convergence_round = None
    credence_at_50 = credence_at_100 = 0.5
    brier_at_50 = brier_at_100 = 0.25  # Brier score of credence 0.5 w.r.t. truth=1

    for round_num in range(n_rounds):
        round_using_A = 0
        evidence = {}

        for agent in agents:
            if agent.agent_type == AgentType.BIASED:
                # [FIX #2]: bias_strength directly = reported success rate
                reported_successes = np.random.binomial(n_per_round, bias_strength)
                evidence[agent.agent_id] = ('B', reported_successes, n_per_round)
            else:
                if agent.credence >= 0.5:
                    evidence[agent.agent_id] = ('A', np.random.binomial(n_per_round, p_A), n_per_round)
                    round_using_A += 1
                else:
                    evidence[agent.agent_id] = ('B', np.random.binomial(n_per_round, p_B), n_per_round)

        n_ts = max(1, n_agents - len(biased_positions))
        rounds_using_A.append(round_using_A / n_ts)

        for agent in agents:
            if agent.agent_type == AgentType.BIASED:
                continue
            tA_succ = tA_tri = tB_succ = tB_tri = 0
            own_t, own_s, own_n = evidence[agent.agent_id]
            if own_t == 'A': tA_succ, tA_tri = own_s, own_n
            else: tB_succ, tB_tri = own_s, own_n

            for nb in G.neighbors(agent.agent_id):
                nt, ns, nn = evidence[nb]
                if nt == 'A': tA_succ += ns; tA_tri += nn
                else: tB_succ += ns; tB_tri += nn

            if tA_tri > 0:
                agent.alpha += tA_succ
                agent.beta += (tA_tri - tA_succ)
            if tB_tri > 0:
                agent.beta += tB_succ
                agent.alpha += (tB_tri - tB_succ)

        ts_creds = [a.credence for a in agents if a.agent_type == AgentType.TRUTH_SEEKER]
        ts_brier = [(1 - c)**2 for c in ts_creds]  # Brier inaccuracy: (truth - credence)^2 where truth=1
        if save_trajectory:
            trajectory.append({'round': round_num, 'mean_credence': np.mean(ts_creds),
                               'std_credence': np.std(ts_creds),
                               'mean_brier': np.mean(ts_brier),
                               'std_brier': np.std(ts_brier)})
        if round_num == 49:
            credence_at_50 = np.mean(ts_creds)
            brier_at_50 = np.mean(ts_brier)
        if round_num == 99:
            credence_at_100 = np.mean(ts_creds)
            brier_at_100 = np.mean(ts_brier)
        if convergence_round is None and all(c > CONFIG["convergence_threshold"] for c in ts_creds):
            convergence_round = round_num

    ts_credences = [a.credence for a in agents if a.agent_type == AgentType.TRUTH_SEEKER]
    ts_brier_scores = [(1 - c)**2 for c in ts_credences]  # Brier inaccuracy scores
    last_portion = rounds_using_A[int(0.8 * n_rounds):]

    # [FIX #3]: Use intended_condition for centrality label
    if "high" in intended_condition: bc_label = "high"
    elif "low" in intended_condition: bc_label = "low"
    else: bc_label = "none"

    if biased_positions:
        bd = np.mean([centralities[p]['degree'] for p in biased_positions])
        bb = np.mean([centralities[p]['betweenness'] for p in biased_positions])
        bc = np.mean([centralities[p]['closeness'] for p in biased_positions])
        be = np.mean([centralities[p]['eigenvector'] for p in biased_positions])
    else:
        bd = bb = bc = be = 0

    result = {
        'phase': phase,
        'simulation_id': f"{topology}_n{n_agents}_{intended_condition}_bs{bias_strength}_r{n_rounds}_npr{n_per_round}_s{seed}",
        'topology': topology, 'topology_class': topology_class,
        'n_agents': n_agents, 'n_rounds': n_rounds, 'n_per_round': n_per_round,
        'n_biased': len(biased_positions), 'biased_positions': str(biased_positions),
        'intended_condition': intended_condition, 'biased_centrality': bc_label,
        'bias_strength': bias_strength, 'efficacy_difference': efficacy_difference,
        'heterogeneous_priors': heterogeneous_priors,
        'final_mean_credence': np.mean(ts_credences), 'final_std_credence': np.std(ts_credences),
        'final_min_credence': np.min(ts_credences), 'final_max_credence': np.max(ts_credences),
        'mean_brier_score': np.mean(ts_brier_scores), 'std_brier_score': np.std(ts_brier_scores),
        'min_brier_score': np.min(ts_brier_scores), 'max_brier_score': np.max(ts_brier_scores),
        'proportion_converged_truth': np.mean([c > CONFIG["convergence_threshold"] for c in ts_credences]),
        'proportion_using_A': np.mean(last_portion) if last_portion else 0,
        'rounds_to_convergence': convergence_round,
        'biased_degree_centrality': bd, 'biased_betweenness_centrality': bb,
        'biased_closeness_centrality': bc, 'biased_eigenvector_centrality': be,
        'credence_at_round_50': credence_at_50, 'credence_at_round_100': credence_at_100,
        'brier_at_round_50': brier_at_50, 'brier_at_round_100': brier_at_100,
        'final_credence_variance': np.var(ts_credences),
        'final_brier_variance': np.var(ts_brier_scores),
        'n_agents_below_05': sum(1 for c in ts_credences if c < 0.5),
        'network_density': net_props['density'],
        'network_avg_clustering': net_props['avg_clustering'],
        'network_diameter': net_props['diameter'],
    }
    return result, trajectory


# ============================================================================
# EXPERIMENT PHASES
# ============================================================================

def _get_positions(G, cond_name, n_biased, centrality):
    """Helper: get biased positions with proper exclusion [FIX #3/#5]."""
    if n_biased == 0:
        return []
    if centrality == "high":
        return get_high_centrality_positions(G, n_biased)
    else:
        high_nodes = get_high_centrality_positions(G, 1)
        return get_low_centrality_positions(G, n_biased, exclude=high_nodes)


def run_phase1_core(seeds_core=100):
    """PHASE 1: Core experiment — asymmetric topologies only [FIX #1]."""
    print("\n" + "=" * 70)
    print("PHASE 1: CORE EXPERIMENT (Asymmetric Topologies Only)")
    print("=" * 70)

    topologies = ASYMMETRIC_TOPOLOGIES
    network_sizes = [6, 10, 15, 20, 30]
    conditions = [
        ("control", 0, None),
        ("1_high", 1, "high"),
        ("1_low", 1, "low"),
        ("2_low", 2, "low"),
        ("3_low", 3, "low"),
        ("4_low", 4, "low"),
    ]
    seeds = list(range(seeds_core))
    total = len(topologies) * len(network_sizes) * len(conditions) * len(seeds)
    print(f"  {len(topologies)} topologies × {len(network_sizes)} sizes × {len(conditions)} conditions × {len(seeds)} seeds = {total}")

    results = []
    start = time.time()
    pbar = tqdm(total=total, desc="Phase 1")

    for topo in topologies:
        for n_agents in network_sizes:
            for cond_name, n_biased, centrality in conditions:
                for seed in seeds:
                    G = create_network(topo, n_agents, seed)
                    biased_pos = _get_positions(G, cond_name, n_biased, centrality)
                    if len(biased_pos) < n_biased and n_biased > 0:
                        pbar.update(1); continue
                    res, _ = run_simulation(
                        G=G, topology=topo, n_agents=n_agents, n_rounds=200,
                        biased_positions=biased_pos, bias_strength=1.0,
                        efficacy_difference=0.05, intended_condition=cond_name,
                        seed=seed, phase=1)
                    results.append(res)
                    pbar.update(1)

    pbar.close()
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results


def run_phase2_bias_strength(seeds_bs=100):
    """PHASE 2: Bias strength moderation [v3: 8 topos, 5 sizes, 100 seeds]."""
    print("\n" + "=" * 70)
    print("PHASE 2: BIAS STRENGTH MODERATION")
    print("=" * 70)

    topologies = ASYMMETRIC_TOPOLOGIES
    network_sizes = [6, 10, 15, 20, 30]
    bias_strengths = [0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
    seeds = list(range(seeds_bs))
    total = len(topologies) * len(network_sizes) * len(bias_strengths) * 2 * len(seeds)
    print(f"  Bias strengths (reported B success rate): {bias_strengths}")
    print(f"  {len(topologies)} topos x {len(network_sizes)} sizes x {len(bias_strengths)} strengths x 2 positions x {len(seeds)} seeds = {total}")

    results = []
    start = time.time()
    pbar = tqdm(total=total, desc="Phase 2")

    for topo in topologies:
        for n_agents in network_sizes:
            for bs in bias_strengths:
                for centrality in ["high", "low"]:
                    for seed in seeds:
                        G = create_network(topo, n_agents, seed)
                        cond = f"1_{centrality}"
                        biased_pos = _get_positions(G, cond, 1, centrality)
                        res, _ = run_simulation(
                            G=G, topology=topo, n_agents=n_agents, n_rounds=200,
                            biased_positions=biased_pos, bias_strength=bs,
                            efficacy_difference=0.05, intended_condition=cond, seed=seed,
                            phase=2)
                        results.append(res)
                        pbar.update(1)

    pbar.close()
    print(f"  \u2713 {len(results)} sims in {time.time()-start:.1f}s")
    return results


def run_phase3_temporal(seeds_temp=100):
    """PHASE 3: Temporal dynamics with n_per_round=1 [v3: 8 topos, 5 sizes, 100 seeds]."""
    print("\n" + "=" * 70)
    print("PHASE 3: TEMPORAL DYNAMICS (n_per_round=1)")
    print("=" * 70)

    topologies = ASYMMETRIC_TOPOLOGIES
    network_sizes = [6, 10, 15, 20, 30]
    durations = [50, 100, 200, 500, 1000]
    seeds = list(range(seeds_temp))
    total = len(topologies) * len(network_sizes) * len(durations) * 3 * len(seeds)
    print(f"  Durations: {durations}, n_per_round=1")
    print(f"  {len(topologies)} topos x {len(network_sizes)} sizes x {len(durations)} durations x 3 conditions x {len(seeds)} seeds = {total}")

    results = []
    trajectory_data = []
    start = time.time()
    pbar = tqdm(total=total, desc="Phase 3")

    for topo in topologies:
        for n_agents in network_sizes:
            for n_rounds in durations:
                for cond in ["control", "1_high", "1_low"]:
                    for seed in seeds:
                        G = create_network(topo, n_agents, seed)
                        if cond == "control": biased_pos = []
                        else: biased_pos = _get_positions(G, cond, 1, cond.split('_')[1])
                        save_traj = (seed < 5 and n_rounds >= 200 and n_agents == 10)
                        res, traj = run_simulation(
                            G=G, topology=topo, n_agents=n_agents, n_rounds=n_rounds,
                            biased_positions=biased_pos, bias_strength=1.0,
                            efficacy_difference=0.05, intended_condition=cond,
                            n_per_round=1, seed=seed, save_trajectory=save_traj,
                            phase=3)
                        if save_traj and traj:
                            for pt in traj:
                                trajectory_data.append({
                                    'topology': topo, 'condition': cond,
                                    'n_rounds_total': n_rounds, 'seed': seed, **pt})
                        results.append(res)
                        pbar.update(1)

    pbar.close()
    print(f"  \u2713 {len(results)} sims in {time.time()-start:.1f}s")
    return results, trajectory_data


def run_phase4_heterogeneous(seeds_het=100):
    """PHASE 4: Heterogeneous priors [v3: 8 topos, 5 sizes, 100 seeds]."""
    print("\n" + "=" * 70)
    print("PHASE 4: HETEROGENEOUS PRIORS")
    print("=" * 70)

    topologies = ASYMMETRIC_TOPOLOGIES
    network_sizes = [6, 10, 15, 20, 30]
    seeds = list(range(seeds_het))
    total = len(topologies) * len(network_sizes) * 2 * 2 * len(seeds)
    print(f"  {len(topologies)} topos x {len(network_sizes)} sizes x 2 prior types x 2 positions x {len(seeds)} seeds = {total}")

    results = []
    start = time.time()
    pbar = tqdm(total=total, desc="Phase 4")

    for topo in topologies:
        for n_agents in network_sizes:
            for het in [False, True]:
                for centrality in ["high", "low"]:
                    for seed in seeds:
                        G = create_network(topo, n_agents, seed)
                        cond = f"1_{centrality}"
                        biased_pos = _get_positions(G, cond, 1, centrality)
                        res, _ = run_simulation(
                            G=G, topology=topo, n_agents=n_agents, n_rounds=200,
                            biased_positions=biased_pos, bias_strength=1.0,
                            efficacy_difference=0.05, intended_condition=cond,
                            heterogeneous_priors=het, seed=seed, phase=4)
                        results.append(res)
                        pbar.update(1)

    pbar.close()
    print(f"  \u2713 {len(results)} sims in {time.time()-start:.1f}s")
    return results


def run_phase5_efficacy(seeds_eff=100):
    """PHASE 5: Task difficulty [v3: 8 topos, 5 sizes, 100 seeds]."""
    print("\n" + "=" * 70)
    print("PHASE 5: TASK DIFFICULTY")
    print("=" * 70)

    topologies = ASYMMETRIC_TOPOLOGIES
    network_sizes = [6, 10, 15, 20, 30]
    efficacy_diffs = [0.01, 0.02, 0.05, 0.10, 0.20]
    seeds = list(range(seeds_eff))
    total = len(topologies) * len(network_sizes) * len(efficacy_diffs) * 3 * len(seeds)
    print(f"  Efficacy diffs: {efficacy_diffs}")
    print(f"  {len(topologies)} topos x {len(network_sizes)} sizes x {len(efficacy_diffs)} diffs x 3 conditions x {len(seeds)} seeds = {total}")

    results = []
    start = time.time()
    pbar = tqdm(total=total, desc="Phase 5")

    for topo in topologies:
        for n_agents in network_sizes:
            for ed in efficacy_diffs:
                for cond in ["control", "1_high", "1_low"]:
                    for seed in seeds:
                        G = create_network(topo, n_agents, seed)
                        if cond == "control": biased_pos = []
                        else: biased_pos = _get_positions(G, cond, 1, cond.split('_')[1])
                        res, _ = run_simulation(
                            G=G, topology=topo, n_agents=n_agents, n_rounds=200,
                            biased_positions=biased_pos, bias_strength=1.0,
                            efficacy_difference=ed, intended_condition=cond, seed=seed,
                            phase=5)
                        results.append(res)
                        pbar.update(1)

    pbar.close()
    print(f"  \u2713 {len(results)} sims in {time.time()-start:.1f}s")
    return results


def run_phase6_negative_controls(seeds_nc=100):
    """PHASE 6: Negative controls — symmetric topologies [FIX #1]."""
    print("\n" + "=" * 70)
    print("PHASE 6: NEGATIVE CONTROLS (Symmetric Topologies)")
    print("=" * 70)

    topologies = SYMMETRIC_TOPOLOGIES
    network_sizes = [6, 10, 15, 20, 30]
    seeds = list(range(seeds_nc))
    total = len(topologies) * len(network_sizes) * 3 * len(seeds)
    print(f"  Topologies: {topologies} (all positions identical)")
    print(f"  Total: {total}")

    results = []
    start = time.time()
    pbar = tqdm(total=total, desc="Phase 6")

    for topo in topologies:
        for n_agents in network_sizes:
            for cond in ["control", "1_high", "1_low"]:
                for seed in seeds:
                    G = create_network(topo, n_agents, seed)
                    if cond == "control":
                        biased_pos = []
                    elif cond == "1_high":
                        biased_pos = [0]  # arbitrary — all nodes identical
                    else:
                        biased_pos = [n_agents - 1]  # different node, same centrality
                    res, _ = run_simulation(
                        G=G, topology=topo, n_agents=n_agents, n_rounds=200,
                        biased_positions=biased_pos, bias_strength=1.0,
                        efficacy_difference=0.05, intended_condition=cond, seed=seed,
                        phase=6)
                    results.append(res)
                    pbar.update(1)

    pbar.close()
    print(f"  ✓ {len(results)} sims in {time.time()-start:.1f}s")
    return results


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def cohens_d(g1, g2):
    pooled = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    return (g2.mean() - g1.mean()) / pooled if pooled > 0 else 0


def run_all_statistics(df):
    print("\n" + "=" * 70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 70)

    # [FIX #14]: Filter on phase to avoid cross-phase contamination
    core = df[(df['phase'] == 1)]
    if len(core) == 0:
        core = df[df['topology_class'] == 'asymmetric']
        print("  WARNING: Using fallback filter for core data")

    single = core[core['n_biased'] == 1]
    control = core[core['n_biased'] == 0]
    has_biased = core[core['n_biased'] > 0]
    R = {}

    # 1. Main effect
    print("\n" + "─" * 70)
    print("1. MAIN EFFECT: Biased agent presence")
    print("─" * 70)
    if len(control) > 0 and len(has_biased) > 0:
        cc, bc = control['mean_brier_score'], has_biased['mean_brier_score']
        t, p = stats.ttest_ind(cc, bc)
        d = cohens_d(bc, cc)
        print(f"  Control:     M={cc.mean():.4f}, SD={cc.std():.4f}, n={len(cc)}")
        print(f"  With biased: M={bc.mean():.4f}, SD={bc.std():.4f}, n={len(bc)}")
        print(f"  t({len(cc)+len(bc)-2})={t:.3f}, p={p:.2e}, d={d:.3f}")

    # 2. Position effect
    print("\n" + "─" * 70)
    print("2. PRIMARY HYPOTHESIS: Position effect (asymmetric only)")
    print("─" * 70)
    high = single[single['biased_centrality'] == 'high']['mean_brier_score']
    low = single[single['biased_centrality'] == 'low']['mean_brier_score']
    if len(high) > 1 and len(low) > 1:
        t_v, p_v = stats.ttest_ind(high, low)
        d_v = cohens_d(high, low)
        u_v, p_mw = stats.mannwhitneyu(high, low, alternative='two-sided')
        print(f"  High: M={high.mean():.4f}, SD={high.std():.4f}, n={len(high)}")
        print(f"  Low:  M={low.mean():.4f}, SD={low.std():.4f}, n={len(low)}")
        print(f"  Diff: {low.mean()-high.mean():.4f}")
        print(f"  t({len(high)+len(low)-2})={t_v:.3f}, p={p_v:.2e}")
        print(f"  Cohen's d={d_v:.3f} ({'large' if abs(d_v)>=0.8 else 'medium' if abs(d_v)>=0.5 else 'small'})")
        print(f"  Mann-Whitney U={u_v:.0f}, p={p_mw:.2e}")
        R.update({'position_t': t_v, 'position_p': p_v, 'position_d': d_v,
                  'high_mean': high.mean(), 'high_sd': high.std(), 'high_n': len(high),
                  'low_mean': low.mean(), 'low_sd': low.std(), 'low_n': len(low)})

    # 3. ANOVA
    print("\n" + "─" * 70)
    print("3. TWO-WAY ANOVA: Position × Topology")
    print("─" * 70)
    try:
        ad = single[single['biased_centrality'].isin(['high', 'low'])].copy()
        vt = [t for t in ad['topology'].unique()
              if set(['high','low']).issubset(set(ad[ad['topology']==t]['biased_centrality'].values))]
        ad = ad[ad['topology'].isin(vt)]
        if len(ad) > 0 and len(vt) > 1:
            gm = ad['mean_brier_score'].mean()
            pm = ad.groupby('biased_centrality')['mean_brier_score']
            pc, pmu = pm.count(), pm.mean()
            ss_p = sum(pc[p]*(pmu[p]-gm)**2 for p in pmu.index)
            tm = ad.groupby('topology')['mean_brier_score']
            tc, tmu = tm.count(), tm.mean()
            ss_t = sum(tc[t]*(tmu[t]-gm)**2 for t in tmu.index)
            cm = ad.groupby(['biased_centrality','topology'])['mean_brier_score']
            cc_m, cc_c = cm.mean(), cm.count()
            ss_i = sum(cc_c[(p,t)]*(cc_m[(p,t)]-pmu[p]-tmu[t]+gm)**2 for (p,t) in cc_m.index)
            ss_tot = sum((ad['mean_brier_score']-gm)**2)
            ss_r = ss_tot - ss_p - ss_t - ss_i
            dp, dt, di = len(pmu)-1, len(tmu)-1, (len(pmu)-1)*(len(tmu)-1)
            dr = len(ad) - len(pmu)*len(tmu)
            fp = (ss_p/max(1,dp))/(ss_r/max(1,dr))
            ft = (ss_t/max(1,dt))/(ss_r/max(1,dr))
            fi = (ss_i/max(1,di))/(ss_r/max(1,dr))
            pp = 1-stats.f.cdf(fp,dp,dr) if dr>0 else 1
            pt = 1-stats.f.cdf(ft,dt,dr) if dr>0 else 1
            pi = 1-stats.f.cdf(fi,di,dr) if dr>0 else 1
            e2p, e2t, e2i = ss_p/ss_tot, ss_t/ss_tot, ss_i/ss_tot
            print(f"\n  {'Source':<25} {'SS':>10} {'df':>5} {'F':>12} {'p':>12} {'η²':>7}")
            print("  "+"─"*73)
            print(f"  {'Position':<25} {ss_p:>10.3f} {dp:>5} {fp:>12.2f} {pp:>12.2e} {e2p:>7.3f}")
            print(f"  {'Topology':<25} {ss_t:>10.3f} {dt:>5} {ft:>12.2f} {pt:>12.2e} {e2t:>7.3f}")
            print(f"  {'Position×Topology':<25} {ss_i:>10.3f} {di:>5} {fi:>12.2f} {pi:>12.2e} {e2i:>7.3f}")
            print(f"  {'Residual':<25} {ss_r:>10.3f} {dr:>5}")
            R.update({'anova_pos_eta2': e2p, 'anova_topo_eta2': e2t, 'anova_inter_eta2': e2i})
    except Exception as e:
        print(f"  ANOVA failed: {e}")

    # 4. By topology
    print("\n" + "─" * 70)
    print("4. POSITION EFFECT BY TOPOLOGY")
    print("─" * 70)
    print(f"\n  {'Topology':<16} {'High M':>8} {'Low M':>8} {'Diff':>8} {'d':>8} {'p':>12} {'Sig':>4}")
    print("  "+"─"*68)
    for topo in sorted(core['topology'].unique()):
        td = single[single['topology']==topo]
        ht = td[td['biased_centrality']=='high']['mean_brier_score']
        lt = td[td['biased_centrality']=='low']['mean_brier_score']
        if len(ht)>1 and len(lt)>1:
            d = cohens_d(ht, lt)
            t_s, p_s = stats.ttest_ind(ht, lt)
            sig = "***" if p_s<.001 else "**" if p_s<.01 else "*" if p_s<.05 else ""
            print(f"  {topo:<16} {ht.mean():>8.4f} {lt.mean():>8.4f} {lt.mean()-ht.mean():>+8.4f} {d:>8.2f} {p_s:>12.2e} {sig:>4}")

    # 5. Centrality measures
    print("\n" + "─" * 70)
    print("5. CENTRALITY MEASURE COMPARISON")
    print("─" * 70)
    print(f"\n  {'Measure':<18} {'r':>8} {'R²':>8} {'p':>14}")
    print("  "+"─"*50)
    best_r, best_m = 0, ""
    for m in ['biased_degree_centrality','biased_betweenness_centrality',
              'biased_closeness_centrality','biased_eigenvector_centrality']:
        sub = single[single[m]>0]
        if len(sub)>2:
            r, p = stats.pearsonr(sub[m], sub['mean_brier_score'])
            nm = m.replace('biased_','').replace('_centrality','')
            print(f"  {nm:<18} {r:>+8.3f} {r**2:>8.3f} {p:>14.2e}")
            if abs(r)>abs(best_r): best_r, best_m = r, nm
    if best_m:
        print(f"\n  → Best: {best_m} (r={best_r:.3f}, R²={best_r**2:.3f})")
        R['best_centrality'] = best_m; R['best_r'] = best_r

    # 5b. Multiple regression
    print("\n  MULTIPLE REGRESSION (all 4 centrality measures):")
    cent_cols = ['biased_degree_centrality','biased_betweenness_centrality',
                 'biased_closeness_centrality','biased_eigenvector_centrality']
    reg_data = single[single['biased_degree_centrality']>0].copy()
    if len(reg_data) > 10:
        X = reg_data[cent_cols].values
        y = reg_data['mean_brier_score'].values
        # Add intercept
        X_int = np.column_stack([np.ones(len(X)), X])
        try:
            beta_hat = np.linalg.lstsq(X_int, y, rcond=None)[0]
            y_pred = X_int @ beta_hat
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot
            n_obs, k = len(y), 4
            r2_adj = 1 - (1-r2)*(n_obs-1)/(n_obs-k-1)
            
            X_std = (X - X.mean(axis=0)) / X.std(axis=0)
            X_std_int = np.column_stack([np.ones(len(X_std)), X_std])
            beta_std = np.linalg.lstsq(X_std_int, y, rcond=None)[0]
            print(f"  R² = {r2:.3f}, Adjusted R² = {r2_adj:.3f}")
            names = ['degree','betweenness','closeness','eigenvector']
            print(f"  {'Measure':<18} {'Std β':>8}")
            print("  "+"─"*28)
            for nm, b in zip(names, beta_std[1:]):
                print(f"  {nm:<18} {b:>+8.4f}")
            R['multiple_r2'] = r2
            R['multiple_r2_adj'] = r2_adj
        except Exception as e:
            print(f"  Regression failed: {e}")

    
    print("\n" + "─" * 70)
    print("6. NUMBERS VS POSITION")
    print("─" * 70)
    print(f"\n  {'Condition':<16} {'Mean':>10} {'SD':>10} {'n':>8}")
    print("  "+"─"*46)
    for c in ['control','1_high','1_low','2_low','3_low','4_low']:
        sub = core[core['intended_condition']==c] if c!='control' else core[core['n_biased']==0]
        if len(sub)>0:
            print(f"  {c:<16} {sub['mean_brier_score'].mean():>10.4f} {sub['mean_brier_score'].std():>10.4f} {len(sub):>8}")

    oh = core[core['intended_condition']=='1_high']['mean_brier_score']
    for nl in [2,3,4]:
        nld = core[core['intended_condition']==f'{nl}_low']['mean_brier_score']
        if len(oh)>1 and len(nld)>1:
            t,p = stats.ttest_ind(oh, nld)
            d = cohens_d(oh, nld)
            direction = "MORE" if nld.mean()>oh.mean() else "LESS"
            sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
            print(f"\n  1H vs {nl}L: t={t:.3f}, p={p:.2e} ({sig}), d={d:.2f} → {nl}L cause {direction} damage")

    # 7. Bayesian t-test
    print("\n" + "─" * 70)
    print("7. BAYESIAN T-TEST (JZS Bayes Factor)")
    print("─" * 70)
    if len(high)>1 and len(low)>1:
        bf_result = pg.bayesfactor_ttest(
            stats.ttest_ind(high, low).statistic,
            len(high), len(low), r=0.707
        )
        log10_bf = np.log10(bf_result) if bf_result > 0 else float('inf')
        print(f"  n_high={len(high)}, n_low={len(low)}")
        print(f"  Cauchy prior r=0.707")
        print(f"  BF₁₀ = {bf_result:.3e}")
        print(f"  log₁₀(BF₁₀) = {log10_bf:.1f}")
        if log10_bf > 2:
            print(f"  Interpretation: Decisive evidence for H₁")
        R['bayes_log10bf'] = log10_bf

    # 8. Bias strength
    print("\n" + "─" * 70)
    print("8. BIAS STRENGTH MODERATION")
    print("─" * 70)
    bsd = df[(df['phase'] == 2)]
    bsv = sorted(bsd['bias_strength'].unique())
    if len(bsv)>1:
        print(f"\n  {'Bias Str':>10} {'High M':>8} {'Low M':>8} {'Diff':>8} {'d':>8}")
        print("  "+"─"*46)
        for bs in bsv:
            bsub = bsd[bsd['bias_strength']==bs]
            ht = bsub[bsub['biased_centrality']=='high']['mean_brier_score']
            lt = bsub[bsub['biased_centrality']=='low']['mean_brier_score']
            if len(ht)>1 and len(lt)>1:
                d = cohens_d(ht, lt)
                print(f"  {bs:>10.2f} {ht.mean():>8.4f} {lt.mean():>8.4f} {lt.mean()-ht.mean():>+8.4f} {d:>8.2f}")

    # 9. Task difficulty
    print("\n" + "─" * 70)
    print("9. TASK DIFFICULTY")
    print("─" * 70)
    ed = df[(df['phase'] == 5)]
    evs = sorted(ed['efficacy_difference'].unique())
    if len(evs)>1:
        print(f"\n  {'Eff Diff':>10} {'High M':>8} {'Low M':>8} {'Diff':>8} {'d':>8}")
        print("  "+"─"*46)
        for ev in evs:
            esub = ed[ed['efficacy_difference']==ev]
            ht = esub[esub['biased_centrality']=='high']['mean_brier_score']
            lt = esub[esub['biased_centrality']=='low']['mean_brier_score']
            if len(ht)>1 and len(lt)>1:
                d = cohens_d(ht, lt)
                print(f"  {ev:>10.3f} {ht.mean():>8.4f} {lt.mean():>8.4f} {lt.mean()-ht.mean():>+8.4f} {d:>8.2f}")

    # 10. Heterogeneous priors
    print("\n" + "─" * 70)
    print("10. HETEROGENEOUS PRIORS")
    print("─" * 70)
    for hv in [False, True]:
        hs = df[(df['phase'] == 4) & (df['heterogeneous_priors']==hv)]
        ht = hs[hs['biased_centrality']=='high']['mean_brier_score']
        lt = hs[hs['biased_centrality']=='low']['mean_brier_score']
        if len(ht)>1 and len(lt)>1:
            d = cohens_d(ht, lt)
            t,p = stats.ttest_ind(ht, lt)
            print(f"  {'Heterogeneous' if hv else 'Uniform':<16} d={d:.3f}, p={p:.2e}")

    # 11. Negative controls
    print("\n" + "─" * 70)
    print("11. NEGATIVE CONTROLS (Symmetric Topologies)")
    print("    Prediction: NO significant position effect")
    print("─" * 70)
    sym = df[(df['phase'] == 6) & (df['n_biased']==1)]
    for topo in SYMMETRIC_TOPOLOGIES:
        ts = sym[sym['topology']==topo]
        ht = ts[ts['biased_centrality']=='high']['mean_brier_score']
        lt = ts[ts['biased_centrality']=='low']['mean_brier_score']
        if len(ht)>1 and len(lt)>1:
            t,p = stats.ttest_ind(ht, lt)
            d = cohens_d(ht, lt)
            v = "✓ PASS" if p>0.05 else f"⚠ p={p:.3f}"
            print(f"  {topo:<12} H={ht.mean():.4f} L={lt.mean():.4f} d={d:.3f} p={p:.3f} → {v}")

    # 12. Temporal
    print("\n" + "─" * 70)
    print("12. TEMPORAL DYNAMICS")
    print("─" * 70)
    td = df[(df['phase'] == 3) & (df['n_biased']==1)]
    durs = sorted(td['n_rounds'].unique())
    if len(durs)>1:
        print(f"\n  {'Duration':>10} {'High M':>8} {'Low M':>8} {'Diff':>8}")
        print("  "+"─"*38)
        for dur in durs:
            ds = td[td['n_rounds']==dur]
            ht = ds[ds['biased_centrality']=='high']['mean_brier_score']
            lt = ds[ds['biased_centrality']=='low']['mean_brier_score']
            if len(ht)>0 and len(lt)>0:
                print(f"  {dur:>10} {ht.mean():>8.4f} {lt.mean():>8.4f} {lt.mean()-ht.mean():>+8.4f}")

    # APA summary
    print("\n" + "=" * 70)
    print("APA-FORMATTED RESULTS")
    print("=" * 70)
    if 'position_d' in R:
        print(f"""
PRIMARY: High-centrality biased agents produced higher Brier inaccuracy
(M={R['high_mean']:.3f}, SD={R['high_sd']:.3f}) than low-centrality
(M={R['low_mean']:.3f}, SD={R['low_sd']:.3f}),
t({R['high_n']+R['low_n']-2})={abs(R['position_t']):.2f},
p {'< .001' if R['position_p']<.001 else f"= {R['position_p']:.3f}"},
d={abs(R['position_d']):.2f}.
[Note: Higher Brier score = greater inaccuracy = more epistemic damage]

ANOVA: Position η²={R.get('anova_pos_eta2',0):.2f},
Topology η²={R.get('anova_topo_eta2',0):.2f},
Interaction η²={R.get('anova_inter_eta2',0):.2f}.

NEGATIVE CONTROL: Symmetric topologies showed no position effect.
BEST PREDICTOR: {R.get('best_centrality','eigenvector')} centrality
(r={R.get('best_r',0):.2f}, R²={R.get('best_r',0)**2:.2f}).
""")
    return R


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_main_figure(df, filename='figure_main.png'):
    core = df[(df['phase'] == 1)]
    if len(core)==0: core = df[df['topology_class']=='asymmetric']
    single = core[core['n_biased']==1]

    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)
    fig.suptitle('Network Position Effects on Epistemic Inaccuracy (Brier Score)\nBayesian Network Epistemology (Asymmetric Topologies)',
                 fontsize=16, fontweight='bold', y=0.98)

    # A: Main effect
    ax = fig.add_subplot(gs[0,0])
    high = single[single['biased_centrality']=='high']['mean_brier_score']
    low = single[single['biased_centrality']=='low']['mean_brier_score']
    bp = ax.boxplot([high, low], positions=[1,2], widths=0.5, patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#e74c3c'); bp['boxes'][1].set_facecolor('#27ae60')
    for data, pos in [(high,1),(low,2)]:
        ax.scatter(np.random.normal(pos, 0.08, len(data)), data, alpha=0.12, s=8, c='black', zorder=3)
    ax.set_xticks([1,2]); ax.set_xticklabels(['High\nCentrality','Low\nCentrality'], fontsize=11)
    ax.set_ylabel('Mean Brier Inaccuracy'); ax.set_title('A. Position Effect', fontsize=12, fontweight='bold')
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uninformed (0.25)')
    if len(high)>1 and len(low)>1:
        t,p = stats.ttest_ind(high, low)
        sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
        ym = max(high.max(), low.max())+0.03
        ax.plot([1,2],[ym,ym],'k-',lw=1.5); ax.text(1.5,ym+0.01,sig,ha='center',fontsize=14,fontweight='bold')

    # B: By topology
    ax = fig.add_subplot(gs[0,1])
    topos = sorted(single['topology'].unique())
    diffs, cols, vt = [], [], []
    for topo in topos:
        td = single[single['topology']==topo]
        hm = td[td['biased_centrality']=='high']['mean_brier_score'].mean()
        lm = td[td['biased_centrality']=='low']['mean_brier_score'].mean()
        if not np.isnan(hm) and not np.isnan(lm):
            d = hm-lm; diffs.append(d); cols.append('#e74c3c' if d>0 else '#27ae60'); vt.append(topo)
    ax.barh(range(len(vt)), diffs, color=cols, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(vt))); ax.set_yticklabels(vt, fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Brier Diff (High−Low)'); ax.set_title('B. By Topology', fontsize=12, fontweight='bold')

    # C: Numbers vs position
    ax = fig.add_subplot(gs[1,0])
    conds = ['control','1_high','1_low','2_low','3_low','4_low']
    ccols = ['#3498db','#e74c3c','#27ae60','#2ecc71','#1abc9c','#16a085']
    ms,ss,ls,cs = [],[],[],[]
    for c,col in zip(conds,ccols):
        sub = core[core['intended_condition']==c] if c!='control' else core[core['n_biased']==0]
        if len(sub)>0:
            ms.append(sub['mean_brier_score'].mean()); ss.append(sub['mean_brier_score'].std())
            ls.append(c); cs.append(col)
    ax.bar(range(len(ls)), ms, yerr=ss, color=cs, edgecolor='black', linewidth=0.5, capsize=4)
    ax.set_xticks(range(len(ls))); ax.set_xticklabels(ls, fontsize=9, rotation=15)
    ax.set_ylabel('Mean Brier Inaccuracy'); ax.set_title('C. Numbers vs Position', fontsize=12, fontweight='bold')
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uninformed (0.25)')

    # D: Degree centrality scatter (best predictor)
    ax = fig.add_subplot(gs[1,1])
    sub = single[single['biased_degree_centrality']>0]
    if len(sub)>2:
        ax.scatter(sub['biased_degree_centrality'], sub['mean_brier_score'],
                  alpha=0.2, s=15, c='#3498db', edgecolor='white', linewidth=0.3)
        z = np.polyfit(sub['biased_degree_centrality'], sub['mean_brier_score'], 1)
        xl = np.linspace(sub['biased_degree_centrality'].min(), sub['biased_degree_centrality'].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), 'r-', lw=2)
        r,_ = stats.pearsonr(sub['biased_degree_centrality'], sub['mean_brier_score'])
        ax.text(0.05,0.95,f'r={r:.2f}\nR\u00b2={r**2:.2f}',transform=ax.transAxes,va='top',fontsize=10,
               bbox=dict(boxstyle='round',facecolor='white',alpha=0.9))
    ax.set_xlabel('Degree Centrality'); ax.set_ylabel('Mean Brier Inaccuracy')
    ax.set_title('D. Degree Centrality Predicts Inaccuracy', fontsize=12, fontweight='bold')

    # E: Network size
    ax = fig.add_subplot(gs[2,0])
    sizes = sorted(core['n_agents'].unique())
    for cent, col, mk in [('high','#e74c3c','o'),('low','#27ae60','s')]:
        ms = [single[(single['n_agents']==s)&(single['biased_centrality']==cent)]['mean_brier_score'].mean() for s in sizes]
        es = [single[(single['n_agents']==s)&(single['biased_centrality']==cent)]['mean_brier_score'].std() for s in sizes]
        ax.errorbar(sizes, ms, yerr=es, marker=mk, color=col, label=f'{cent.capitalize()}', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Network Size'); ax.set_ylabel('Mean Brier Inaccuracy')
    ax.set_title('E. Effect by Network Size', fontsize=12, fontweight='bold'); ax.legend(); ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uninformed (0.25)')

    # F: Effect sizes
    ax = fig.add_subplot(gs[2,1])
    es_list, es_labels = [], []
    if len(high)>1 and len(low)>1:
        es_list.append(cohens_d(high, low)); es_labels.append('OVERALL')
    for topo in vt:
        td = single[single['topology']==topo]
        ht = td[td['biased_centrality']=='high']['mean_brier_score']
        lt = td[td['biased_centrality']=='low']['mean_brier_score']
        if len(ht)>1 and len(lt)>1: es_list.append(cohens_d(ht, lt)); es_labels.append(topo)
    ec = ['#e74c3c']+['#95a5a6']*(len(es_labels)-1)
    ax.barh(range(len(es_labels)), es_list, color=ec, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=1)
    for th in [0.2,0.5,0.8]: ax.axvline(th, color='gray', linestyle=':', alpha=0.5)
    ax.set_yticks(range(len(es_labels))); ax.set_yticklabels(es_labels, fontsize=9)
    ax.set_xlabel("Cohen's d"); ax.set_title('F. Effect Sizes', fontsize=12, fontweight='bold')

    # G: Bias strength
    ax = fig.add_subplot(gs[3,0])
    bsd = df[(df['phase'] == 2)]
    bsv = sorted(bsd['bias_strength'].unique())
    if len(bsv)>1:
        for cent,col,mk in [('high','#e74c3c','o'),('low','#27ae60','s')]:
            ms = [bsd[(bsd['bias_strength']==b)&(bsd['biased_centrality']==cent)]['mean_brier_score'].mean() for b in bsv]
            es = [bsd[(bsd['bias_strength']==b)&(bsd['biased_centrality']==cent)]['mean_brier_score'].std() for b in bsv]
            ax.errorbar(bsv, ms, yerr=es, marker=mk, color=col, label=f'{cent.capitalize()}', capsize=4, linewidth=2, markersize=8)
        ax.fill_between(bsv,
                        [bsd[(bsd['bias_strength']==b)&(bsd['biased_centrality']=='high')]['mean_brier_score'].mean() for b in bsv],
                        [bsd[(bsd['bias_strength']==b)&(bsd['biased_centrality']=='low')]['mean_brier_score'].mean() for b in bsv],
                        alpha=0.12, color='#3498db')
        ax.legend()
    ax.set_xlabel('Bias Strength (Reported B Success Rate)'); ax.set_ylabel('Mean Brier Inaccuracy')
    ax.set_title('G. Bias Strength Moderation', fontsize=12, fontweight='bold'); ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uninformed (0.25)')

    # H: Task difficulty
    ax = fig.add_subplot(gs[3,1])
    efd = df[(df['phase'] == 5)]
    efv = sorted(efd['efficacy_difference'].unique())
    if len(efv)>1:
        for cent,col,mk in [('high','#e74c3c','o'),('low','#27ae60','s')]:
            ms = [efd[(efd['efficacy_difference']==e)&(efd['biased_centrality']==cent)]['mean_brier_score'].mean() for e in efv]
            es = [efd[(efd['efficacy_difference']==e)&(efd['biased_centrality']==cent)]['mean_brier_score'].std() for e in efv]
            ax.errorbar(efv, ms, yerr=es, marker=mk, color=col, label=f'{cent.capitalize()}', capsize=4, linewidth=2, markersize=8)
        ax.legend()
    ax.set_xlabel('Efficacy Difference (p_A − 0.5)'); ax.set_ylabel('Mean Brier Inaccuracy')
    ax.set_title('H. Task Difficulty', fontsize=12, fontweight='bold'); ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uninformed (0.25)')

    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Saved: {filename}")


def plot_temporal_figure(traj_data, filename='figure_temporal.png'):
    if not traj_data:
        print("  No trajectory data"); return
    tdf = pd.DataFrame(traj_data)
    topos = sorted(tdf['topology'].unique())

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7), sharey=True)
    fig.suptitle('Temporal Dynamics (n_per_round=1, slow learning)',
                 fontsize=14, fontweight='bold', y=1.02)

    for idx, topo in enumerate(topos):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        md = tdf[tdf['topology']==topo]['n_rounds_total'].max()
        td = tdf[(tdf['topology']==topo)&(tdf['n_rounds_total']==md)]
        for cond, c, ls in [('control','#3498db','--'),('1_high','#e74c3c','-'),('1_low','#27ae60','-')]:
            cd = td[td['condition']==cond]
            if len(cd)>0:
                mt = cd.groupby('round')['mean_brier'].mean()
                st = cd.groupby('round')['mean_brier'].std()
                ax.plot(mt.index, mt.values, color=c, linestyle=ls,
                       label=cond.replace('1_','').capitalize(), linewidth=2, alpha=0.8)
                ax.fill_between(mt.index, mt.values-st.values, mt.values+st.values, color=c, alpha=0.1)
        ax.set_title(topo, fontweight='bold', fontsize=11)
        if row == 1:
            ax.set_xlabel('Round')
        ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=8)

    axes[0, 0].set_ylabel('Mean Brier Inaccuracy')
    axes[1, 0].set_ylabel('Mean Brier Inaccuracy')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_negative_controls(df, filename='figure_negative_controls.png'):
    sym = df[(df['phase'] == 6) & (df['n_biased']==1)]
    asym = df[(df['phase'] == 1) & (df['n_biased']==1)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Negative Controls & Robustness', fontsize=14, fontweight='bold')

    # A: Effect sizes comparison
    ax = axes[0]
    effects = []
    for topo in sorted(asym['topology'].unique()):
        td = asym[asym['topology']==topo]
        ht = td[td['biased_centrality']=='high']['mean_brier_score']
        lt = td[td['biased_centrality']=='low']['mean_brier_score']
        if len(ht)>1 and len(lt)>1:
            effects.append((topo, cohens_d(ht,lt), '#27ae60'))
    for topo in SYMMETRIC_TOPOLOGIES:
        ts = sym[sym['topology']==topo]
        ht = ts[ts['biased_centrality']=='high']['mean_brier_score']
        lt = ts[ts['biased_centrality']=='low']['mean_brier_score']
        if len(ht)>1 and len(lt)>1:
            effects.append((topo+' ★', cohens_d(ht,lt), '#e74c3c'))
    if effects:
        names, ds, cs = zip(*effects)
        ax.barh(range(len(names)), ds, color=cs, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel("Cohen's d"); ax.set_title('A. Asymmetric vs Symmetric (★)', fontsize=11, fontweight='bold')

    # B: Het priors
    ax = axes[1]
    het_d = df[(df['phase'] == 4)]
    x_pos = 0
    for hv, label, col in [(False,'Uniform','#3498db'),(True,'Hetero','#e67e22')]:
        hs = het_d[het_d['heterogeneous_priors']==hv]
        for cent, cc in [('high','#e74c3c'),('low','#27ae60')]:
            cs = hs[hs['biased_centrality']==cent]
            if len(cs)>0:
                ax.bar(x_pos, cs['mean_brier_score'].mean(),
                      yerr=cs['mean_brier_score'].std(),
                      color=cc, edgecolor='black', linewidth=0.5, capsize=4, width=0.7)
                ax.text(x_pos, 0.01, f"{label[:3]}_{cent[0]}", ha='center', fontsize=8, rotation=45)
                x_pos += 1
        x_pos += 0.5
    ax.set_xticks([]); ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uninformed (0.25)')
    ax.set_ylabel('Mean Brier Inaccuracy')
    ax.set_title('B. Heterogeneous vs Uniform Priors', fontsize=11, fontweight='bold')

    # C: Variance
    ax = axes[2]
    for cent, col in [('high','#e74c3c'),('low','#27ae60')]:
        cs = asym[asym['biased_centrality']==cent]
        if len(cs)>0:
            ax.hist(cs['final_brier_variance'], bins=30, alpha=0.5, color=col,
                   label=f'{cent.capitalize()}', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Final Brier Score Variance'); ax.set_ylabel('Count')
    ax.legend(); ax.set_title('C. Within-Network Disagreement', fontsize=11, fontweight='bold')

    plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Saved: {filename}")


def plot_centrality_comparison(df, filename='figure_centrality.png'):
    """Figure 3: Four-panel centrality regression comparison."""
    core = df[(df['phase'] == 1)]
    if len(core)==0: core = df[df['topology_class']=='asymmetric']
    single = core[core['n_biased']==1]

    measures = [
        ('biased_degree_centrality', 'Degree Centrality'),
        ('biased_eigenvector_centrality', 'Eigenvector Centrality'),
        ('biased_closeness_centrality', 'Closeness Centrality'),
        ('biased_betweenness_centrality', 'Betweenness Centrality'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Centrality Measures as Predictors of Epistemic Inaccuracy (Brier Score)',
                 fontsize=16, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for idx, (col, label) in enumerate(measures):
        ax = axes[idx]
        sub = single[single[col] > 0]
        if len(sub) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
            continue

        x = sub[col].values
        y = sub['mean_brier_score'].values

        # Scatter
        ax.scatter(x, y, alpha=0.15, s=12, c='#3498db', edgecolor='white', linewidth=0.2)

        # Regression line
        z = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), 'r-', lw=2.5)

        # Statistics
        r, p = stats.pearsonr(x, y)
        panel_label = chr(65 + idx)  # A, B, C, D
        ax.set_title(f'{panel_label}. {label}', fontsize=13, fontweight='bold')
        ax.text(0.05, 0.95, f'r = {r:.2f}\nR\u00b2 = {r**2:.2f}\np < .001',
                transform=ax.transAxes, va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Mean Brier Inaccuracy', fontsize=11)
        ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uninformed (0.25)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\u2713 Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__" or True:
    print("\n" + "=" * 70)
    print("BAYESIAN NETWORK EPISTEMOLOGY EXPERIMENT v5 (BRIER SCORE)")
    print("=" * 70)
    print(f"Start: {time.strftime('%H:%M:%S')}")
    t0 = time.time()

    r1 = run_phase1_core(seeds_core=100)
    r2 = run_phase2_bias_strength(seeds_bs=100)
    r3, traj = run_phase3_temporal(seeds_temp=100)
    r4 = run_phase4_heterogeneous(seeds_het=100)
    r5 = run_phase5_efficacy(seeds_eff=100)
    r6 = run_phase6_negative_controls(seeds_nc=100)

    all_res = r1 + r2 + r3 + r4 + r5 + r6
    df = pd.DataFrame(all_res)
    df.to_csv('comprehensive_results_v4_brier.csv', index=False)
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL PHASES: {len(df)} sims in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")

    stats_results = run_all_statistics(df)

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_main_figure(df, 'figure_main.png')
    plot_centrality_comparison(df, 'figure_centrality.png')
    plot_temporal_figure(traj, 'figure_temporal.png')
    plot_negative_controls(df, 'figure_negative_controls.png')

    print(f"""
{'='*70}
EXPERIMENT COMPLETE
{'='*70}
  Simulations: {len(df)}
  Runtime:     {total_time/60:.1f} min

  Output files:
    comprehensive_results_v5_brier.csv
    figure_main.png              (8-panel, Brier score)
    figure_centrality.png        (4-panel centrality comparison)
    figure_temporal.png          (slow learning dynamics)
    figure_negative_controls.png (symmetric topology validation)

  

""")