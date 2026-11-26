import os
import sys
import time
import random
import numpy as np
from scipy.stats import binom

import cassiopeia as cass
from cassiopeia.simulator import BirthDeathFitnessSimulator, Cas9LineageTracingDataSimulator, UniformLeafSubsampler
from cassiopeia.solver import MaxCutGreedySolver, MaxCutSolver, NeighborJoiningSolver, \
    VanillaGreedySolver, SharedMutationJoiningSolver, SpectralGreedySolver, SpectralSolver

# the cPHS functions are those of the KPTracer analysis, in ../cphs
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cphs import calc_phs_and_lca_times, get_lam_and_q, prepare_tree


def apply_FDR(pvalues):
    # BH-adjusted p-values (not capped at 1)
    N = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    adjusted_pvalues = pvalues_sorted * N / np.arange(1, N+1)
    return adjusted_pvalues


class sim_functions:

    solver_names = ["nj", "greedy", "smj", "maxcut", "maxcut_greedy", "spectral", "spectral_greedy"]

    def _get_tree(n_cells, leaf_subsampling_ratio, k, m, rho):
        # simulate tree (birth-death process with fitness, Appendix J). with subsampling, the tree
        # is grown to n_cells / leaf_subsampling_ratio cells, and n_cells of them are sampled
        n_cells_total = int(round(n_cells / leaf_subsampling_ratio))
        tree_simulator = BirthDeathFitnessSimulator(
            num_extant=n_cells_total,
            birth_waiting_distribution=lambda scale: np.random.exponential(1/scale),
            initial_birth_scale=2,
            death_waiting_distribution=lambda: np.random.exponential(1.5),
            mutation_distribution=lambda: 1 if np.random.uniform() < 0.5 else 0,
            fitness_distribution=lambda: np.random.normal(0.5, 0.25),
            fitness_base=1.1)
        tree = tree_simulator.simulate_tree()

        # subsample leaves
        if n_cells_total > n_cells:
            tree = UniformLeafSubsampler(number_of_leaves=n_cells).subsample_leaves(tree)

        # scale to unit length
        tree.scale_to_unit_length()

        # simulate lineage tracing: non-modifiable model with k sites and m states,
        # the state probabilities are drawn from an exponential distribution, and there is no missing data
        state_priors = np.random.exponential(size=m)
        state_priors /= np.sum(state_priors)
        state_priors = {j + 1: state_priors[j] for j in range(m)}
        lt_simulator = Cas9LineageTracingDataSimulator(
            number_of_cassettes=k,
            size_of_cassette=1,
            mutation_rate=-np.log(1.0 - rho),
            number_of_states=m,
            state_priors=state_priors,
            heritable_silencing_rate=0.0,
            stochastic_silencing_rate=0.0)
        lt_simulator.overlay_data(tree)
        tree.priors = {i: state_priors for i in range(k)}

        return tree

    def _solve_tree(tree, solver_name):
        # get solver by name
        if solver_name == "nj":
            solver = NeighborJoiningSolver(add_root=True)
        elif solver_name == "greedy":
            solver = VanillaGreedySolver()
        elif solver_name == "smj":
            solver = SharedMutationJoiningSolver()
        elif solver_name == "maxcut":
            solver = MaxCutSolver()
        elif solver_name == "maxcut_greedy":
            solver = MaxCutGreedySolver()
        elif solver_name == "spectral":
            solver = SpectralSolver()
        elif solver_name == "spectral_greedy":
            solver = SpectralGreedySolver()

        # solve
        solver.solve(tree)

    def _preprocess_tree(tree):
        # contract mutationless edges, and then the same preprocessing as for the KPTracer trees
        # (cphs.prepare_tree): small-MP ancestral states, unmutated root and ML branch lengths
        tree.collapse_mutationless_edges(infer_ancestral_characters=True)
        return prepare_tree(tree)


    def simulate_and_run_solvers(seed, n_cells, leaf_subsampling_ratio, k, m, rho, solver_names):
        np.random.seed(seed)
        random.seed(seed)

        # simulate the GT tree
        tree_gt = sim_functions._get_tree(n_cells, leaf_subsampling_ratio, k, m, rho)
        character_matrix = tree_gt.character_matrix.copy()
        priors = tree_gt.priors

        # the GT tree is preprocessed like the reconstructed trees, so that their scores are comparable
        # (otherwise, the GT tree gets an unfair advantage)
        trees = {"gt": sim_functions._preprocess_tree(tree_gt)}

        # solve tree using given solvers
        for solver_name in solver_names:
            tree = cass.data.CassiopeiaTree(character_matrix=character_matrix.copy(), priors=priors)
            try:
                sim_functions._solve_tree(tree, solver_name)
                trees[solver_name] = sim_functions._preprocess_tree(tree)
            except Exception as e:
                # some solvers fail on some inputs, in which case we skip them
                print("solver {} failed: {}".format(solver_name, e))

        return trees


    def calculate_metrics(tree, tree_gt, mutation_rate, q):
        n, k = tree.character_matrix.shape

        # RF score
        rf, rf_max = cass.critique.robinson_foulds(tree_gt, tree)
        rf = rf / rf_max if rf_max > 0 else 0.0

        # triplets
        triplets = cass.critique.triplets_correct(tree_gt, tree)
        triplets = np.mean(list(triplets[0].values()))

        # parsimony
        parsimony = cass.tools.calculate_parsimony(tree)

        # p-values of all pairs of leaves (as in cphs.calc_cPHS, which returns only the cPHS)
        phs, heights = calc_phs_and_lca_times(tree)
        heights = heights.to_numpy()
        alpha = np.exp(-mutation_rate * heights)
        beta = 1 - np.exp(-mutation_rate * (1 - heights))
        prob = alpha * beta**2 * q
        prob[heights == 1] = 1 # fixing a pathology due to internal nodes with height 1 (with mutationless edges to their children leaves)
        pvalues_matrix = 1 - binom.cdf(phs.to_numpy() - 1, k, prob)
        off_diagonal_entries = ~np.eye(n, dtype=bool)
        pvalues = pvalues_matrix[off_diagonal_entries]
        pvalues[pvalues == 0] = np.finfo(float).eps # zeros cannot be real zeros, as prob > 0

        # cPHS (Eq. 13), and the variant of Appendix A.3 without the pairs that have no homoplasy (p-value 1)
        cPHS = np.min(apply_FDR(pvalues))
        pvalues_A3 = pvalues[pvalues < 1]
        if len(pvalues_A3) > 0:
            cPHS_A3 = np.min(apply_FDR(pvalues_A3))
        else:
            cPHS_A3 = 1.0

        # parsimony pvalue P[M >= parsimony | T] (Appendix D). the characters are iid, so the
        # parsimony distribution is the k-fold convolution of the distribution of a single character
        dist = sim_functions._parsimony_dist(tree, mutation_rate)
        convolved = np.ones(1)
        for _ in range(k):
            convolved = sim_functions._trim_tail(np.convolve(convolved, dist))
        parsimony_pvalue = min(1.0, np.sum(convolved[parsimony:]))

        # likelihood. the mutation rate is estimated by cassiopeia (also for the GT tree), and there is no missing data
        tree.parameters.pop("mutation_rate", None)
        tree.parameters.pop("stochastic_missing_probability", None)
        tree.parameters["heritable_missing_rate"] = 0.0
        likelihood = cass.tools.calculate_likelihood_continuous(tree)

        return {"rf": rf, "triplets": triplets, "parsimony": parsimony, "parsimony_pvalue": parsimony_pvalue,
                "likelihood": likelihood, "cPHS": cPHS, "cPHS_A3": cPHS_A3}

    def _parsimony_dist(tree, mutation_rate):
        # distribution of the number of mutations of a single character, given the tree with its branch lengths.
        # computed from the leaves up: dist[node] is the distribution in the subtree of node
        dist = {}
        for node in tree.depth_first_traverse_nodes(postorder=True):
            node_dist = np.ones(1)
            for ch in tree.children(node):
                no_mutation_prob = np.exp(-mutation_rate * tree.get_branch_length(node, ch))
                ch_dist = dist.pop(ch)
                edge_dist = np.zeros(max(len(ch_dist), 2))
                edge_dist[:len(ch_dist)] = no_mutation_prob * ch_dist # no mutation on the edge to ch
                edge_dist[1] += 1 - no_mutation_prob # a mutation on the edge to ch, so no more mutations below it
                node_dist = np.convolve(node_dist, edge_dist)
            dist[node] = sim_functions._trim_tail(node_dist)
        return dist[tree.root]

    def _trim_tail(dist):
        # drop negligible probabilities at the end of the distribution (for speed)
        return np.trim_zeros(np.where(dist < 1e-300, 0, dist), "b")


    def simulate_and_calculate_metrics(seed, n_cells, leaf_subsampling_ratio, k, m, rho, solver_names):
        trees = sim_functions.simulate_and_run_solvers(seed, n_cells, leaf_subsampling_ratio, k, m, rho, solver_names)
        tree_gt = trees["gt"]

        # estimate the mutation rate from the data. the collision probability q is known (Appendix A.1)
        mutation_rate_estimate, collision_prob = get_lam_and_q(tree_gt)

        # calculate metrics
        scores = []
        for tree_name in trees:
            print("calculating metrics for {}...".format(tree_name))
            start_time = time.time()
            metrics = sim_functions.calculate_metrics(trees[tree_name], tree_gt, mutation_rate_estimate, collision_prob)
            scores.append({"tree": tree_name, "lam_hat": mutation_rate_estimate, "q": collision_prob, **metrics,
                           "time_s": round(time.time() - start_time, 2)})
        return scores
