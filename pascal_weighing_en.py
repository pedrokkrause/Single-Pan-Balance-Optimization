# =======================================================================
# Script: Pascal Combinatorial Weighing for Arbitrary N and m (v2.0)
# =======================================================================
# Theoretical Foundations:
# 1. Bilateral Truncation of Pascal's Triangle and Ridge k_A = k_B + 1
# 2. Cyclic Orbit Decomposition with 1st and 2nd Order Isotropy
#    (Automatic selection of Difference Sets / Fano Plane)
# 3. Spectral Decomposition of the Schur Complement and Closed-Form Formula
# 4. Benchtop Routing via Minimum Hamming Distance + 2-Opt
# =======================================================================

import itertools
import math
import os
import numpy as np

# Enable ANSI colors in Windows terminal
if os.name == 'nt':
    os.system('color')

# ANSI color codes
COLOR_TITLE = '\033[96m'      # Cyan
COLOR_ADD = '\033[92m'        # Green
COLOR_REMOVE = '\033[91m'     # Red
COLOR_PROMPT = '\033[93m'     # Yellow
COLOR_RESULT = '\033[92m'     # Green
COLOR_ERROR = '\033[91m'      # Red
COLOR_WARNING = '\033[95m'    # Magenta (Undo / Alert)
COLOR_HIGHLIGHT = '\033[94m'  # Light Blue
RESET = '\033[0m'

# =======================================================================
# 1. EXPERIMENT CONFIGURATION PARAMETERS
# =======================================================================
# The first item in the list MUST be the primary object of interest (m1).
# The remaining objects act as auxiliary masses (m2, ..., mN).
ITEMS = ['flash_drive', 'square_key', 'toy_llama', 'padlock', 'eraser', 'toy_frog']

# Desired total weighing budget (m >= len(ITEMS))
# Examples for N=6 (p=5): Star Clean Point = 7; Intermediate = 22; Full Factorial = 63
M_TARGET = 10


# =======================================================================
# 2. PASCAL DESIGN CONSTRUCTION AND SPECTRAL SHIELDING
# =======================================================================
def _matrix_isotropy_metrics(X_aux):
    """Computes the 1st-order dispersion Var(k_i) and 2nd-order dispersion
    Var(lambda_ij) for the auxiliary submatrix X_aux.
    """
    p = X_aux.shape[1]
    if len(X_aux) == 0 or p <= 1:
        return 0.0, 0.0
    M = X_aux.T @ X_aux
    k_diag = np.diag(M)
    var_k = float(np.var(k_diag))
    mask_off = ~np.eye(p, dtype=bool)
    var_lambda = float(np.var(M[mask_off]))
    return var_k, var_lambda


def decompose_layer_into_orbits(p, r):
    """Decomposes layer C(p, r) into orbits of the cyclic group C_p.
    Sorts orbits prioritizing:
      1. Maximum length (1st-order isotropy: Var(k_i) = 0)
      2. Minimum pair co-occurrence variance Var(lambda_ij) (2nd-order
         isotropy: automatically selects perfect difference sets, such
         as the Fano seed {0, 1, 3} mod 7).
    """
    all_combs = list(itertools.combinations(range(p), r))
    seen = set()
    orbits_info = []

    for comb in all_combs:
        if comb in seen:
            continue
        orb = []
        for shift in range(p):
            rot = tuple(sorted((x + shift) % p for x in comb))
            if rot not in orb:
                orb.append(rot)
                seen.add(rot)

        # Binary matrix of the orbit to evaluate 2nd-order isotropy
        X_orb = np.zeros((len(orb), p), dtype=float)
        for idx, c in enumerate(orb):
            X_orb[idx, list(c)] = 1.0
        var_k, var_lambda = _matrix_isotropy_metrics(X_orb)

        orbits_info.append({
            'orb': orb,
            'X_orb': X_orb,
            'len': len(orb),
            'var_k': var_k,
            'var_lambda': var_lambda,
        })

    # Sort: largest size -> lowest Var(k_i) -> lowest Var(lambda_ij)
    orbits_info.sort(
        key=lambda item: (-item['len'], round(item['var_k'], 9), round(item['var_lambda'], 9))
    )
    return orbits_info


def select_balanced_rows(X_base, X_orb_cand, k_remainder, N, include_m1):
    """Selects k_remainder rows from a single candidate orbit minimizing
    jointly the 1st-order variance Var(k_i) and 2nd-order variance Var(lambda_ij).
    """
    if k_remainder <= 0 or len(X_orb_cand) == 0:
        return X_base

    p = N - 1
    X_base_aux = X_base[:, 1:] if len(X_base) > 0 else np.empty((0, p), dtype=float)
    n_cand = len(X_orb_cand)
    indices = list(range(n_cand))

    best_combo = None
    best_key = (np.inf, np.inf)

    # Since X_orb_cand comes from a single orbit (n_cand <= p), C(p, k_remainder) is small
    if math.comb(n_cand, k_remainder) <= 5000:
        for combo in itertools.combinations(indices, k_remainder):
            X_trial_aux = np.vstack([X_base_aux, X_orb_cand[list(combo), :]])
            var_k, var_l = _matrix_isotropy_metrics(X_trial_aux)
            key = (round(var_k, 9), round(var_l, 9))
            if key < best_key:
                best_key = key
                best_combo = list(combo)
    else:
        current = []
        available = indices.copy()
        X_running = X_base_aux.copy()
        for _ in range(k_remainder):
            best_idx = None
            best_step_key = (np.inf, np.inf)
            for idx in available:
                X_trial = np.vstack([X_running, X_orb_cand[[idx], :]])
                var_k, var_l = _matrix_isotropy_metrics(X_trial)
                key = (round(var_k, 9), round(var_l, 9))
                if key < best_step_key:
                    best_step_key = key
                    best_idx = idx
            current.append(best_idx)
            available.remove(best_idx)
            X_running = np.vstack([X_running, X_orb_cand[[best_idx], :]])
        best_combo = current

    sub_aux = X_orb_cand[best_combo, :]
    col_m1 = np.ones((k_remainder, 1)) if include_m1 else np.zeros((k_remainder, 1))
    chosen_rows = np.hstack([col_m1, sub_aux])

    return np.vstack([X_base, chosen_rows]) if len(X_base) > 0 else chosen_rows


def build_hierarchical_block(p, m_block, include_m1, N):
    """Constructs Block A or B following the strict hierarchy:
      1. Full binomial layers of Pascal's Triangle
      2. Full cyclic orbits selected by cumulative 2nd-order isotropy
      3. Sub-orbital subset (k < p) extracted from the best residual orbit
    """
    weight_order = range(p + 1) if include_m1 else range(p, 0, -1)
    X_block = np.empty((0, N), dtype=float)
    remaining_budget = m_block

    for r in weight_order:
        if remaining_budget <= 0:
            break

        layer_total = math.comb(p, r)

        # 1. Full layer fits in the remaining budget
        if remaining_budget >= layer_total:
            combs = list(itertools.combinations(range(p), r))
            rows = np.zeros((layer_total, N), dtype=float)
            if include_m1:
                rows[:, 0] = 1.0
            for idx, c in enumerate(combs):
                for bit in c:
                    rows[idx, bit + 1] = 1.0
            X_block = np.vstack([X_block, rows])
            remaining_budget -= layer_total
            continue

        # 2. Partial layer: select complete orbits minimizing cumulative Var(lambda_ij)
        available_orbits = decompose_layer_into_orbits(p, r)

        while remaining_budget > 0 and available_orbits:
            # Filter orbits that fit completely in the remaining budget
            fitting = [
                (i, info) for i, info in enumerate(available_orbits)
                if info['len'] <= remaining_budget
            ]
            if not fitting:
                break

            # Choose the orbit that maximizes size and minimizes cumulative Var(k_i) and Var(lambda_ij)
            X_base_aux = X_block[:, 1:] if len(X_block) > 0 else np.empty((0, p), dtype=float)
            best_pos = None
            best_crit = (np.inf, np.inf, np.inf)

            for pos_list, info in fitting:
                X_trial_aux = np.vstack([X_base_aux, info['X_orb']])
                var_k, var_l = _matrix_isotropy_metrics(X_trial_aux)
                crit = (-info['len'], round(var_k, 9), round(var_l, 9))
                if crit < best_crit:
                    best_crit = crit
                    best_pos = pos_list

            chosen = available_orbits.pop(best_pos)
            size = chosen['len']
            col_m1 = np.ones((size, 1)) if include_m1 else np.zeros((size, 1))
            orb_rows = np.hstack([col_m1, chosen['X_orb']])
            X_block = np.vstack([X_block, orb_rows])
            remaining_budget -= size

        # 3. Sub-orbital remainder (1 <= remaining_budget < p): extract from best residual orbit
        if remaining_budget > 0 and available_orbits:
            best_residual_orb = available_orbits[0]['X_orb']
            X_block = select_balanced_rows(
                X_block, best_residual_orb, remaining_budget, N, include_m1
            )
            remaining_budget = 0

    return X_block


def compute_uncertainty(X):
    """Computes sigma_1 / sigma with exact Gauss-Markov estimability test."""
    m, n = X.shape
    if m < n:
        return np.inf

    XtX = X.T @ X
    c = np.zeros(n, dtype=float)
    c[0] = 1.0

    try:
        pinv_XtX = np.linalg.pinv(XtX)
        if not np.allclose(c @ pinv_XtX @ XtX, c, atol=1e-5):
            return np.inf
        var_m1 = pinv_XtX[0, 0]
        return float(np.sqrt(max(0.0, var_m1)))
    except np.linalg.LinAlgError:
        return np.inf


def decompose_schur_spectrum(X):
    """Performs the spectral decomposition of the Schur penalty:
    Phi = m_A - (P_collective + P_leakage)
    """
    p = X.shape[1] - 1
    mask_A = X[:, 0] == 1.0
    mA = int(np.sum(mask_A))
    mB = int(len(X) - mA)

    X_aux = X[:, 1:]
    X_A_aux = X[mask_A, 1:]

    s_vec = np.sum(X_A_aux, axis=0)
    s_bar = float(np.mean(s_vec))
    std_s = float(np.std(s_vec))

    M_aux = X_aux.T @ X_aux
    total_penalty = float(s_vec.T @ np.linalg.pinv(M_aux) @ s_vec)
    phi = float(mA - total_penalty)

    # Collective mode v1 = 1_p / sqrt(p)
    v1 = np.ones(p) / np.sqrt(p)
    mu1 = float(v1.T @ M_aux @ v1)  # Mean of row sums of M_aux
    p_collective = float(((v1.T @ s_vec) ** 2) / mu1) if mu1 > 0 else np.inf
    p_leakage = max(0.0, total_penalty - p_collective)

    # Verify whether v1 is an exact eigenvector of M_aux and s_vec is uniform
    eigenvector_residual = float(np.linalg.norm(M_aux @ v1 - mu1 * v1))
    isotropic_1st_order = (std_s < 1e-7) and (eigenvector_residual < 1e-7)

    return {
        'mA': mA,
        'mB': mB,
        's_bar': s_bar,
        'std_s': std_s,
        'mu1': mu1,
        'ratio_s_mu1': s_bar / mu1 if mu1 > 0 else 0.0,
        'total_penalty': total_penalty,
        'p_collective': p_collective,
        'p_leakage': p_leakage,
        'phi': phi,
        'isotropic_1st_order': isotropic_1st_order,
    }


def generate_pascal_design(N, m_target):
    """Scans all partitions (mA, mB) and selects the one maximizing net information Phi
    (breaking ties by minimum parasitic leakage).
    """
    p = N - 1
    best_key = (np.inf, np.inf)
    best_X = None
    best_info = None

    max_possible_mB = (2 ** p) - 1
    min_mA = 1
    max_mA = min(2 ** p, m_target - 1)

    for mA in range(min_mA, max_mA + 1):
        mB = m_target - mA
        if mB < 1 or mB > max_possible_mB:
            continue

        X_A = build_hierarchical_block(p, mA, include_m1=True, N=N)
        X_B = build_hierarchical_block(p, mB, include_m1=False, N=N)
        X_cand = np.vstack([X_A, X_B])

        sig = compute_uncertainty(X_cand)
        if np.isinf(sig):
            continue

        info = decompose_schur_spectrum(X_cand)
        # Prioritize lowest uncertainty (6 decimal places) and tie-break by lowest spectral leakage
        cand_key = (round(sig, 6), round(info['p_leakage'], 6))
        if cand_key < best_key:
            best_key = cand_key
            best_X = X_cand
            best_info = info
            best_info['sig1_factor'] = sig

    if best_X is None:
        return None, None

    return best_X, best_info


def optimize_weighing_route(X):
    """Reorders weighings to minimize cumulative Hamming distance
    (Nearest Neighbor + 2-Opt local refinement starting from an empty pan).
    """
    m, N = X.shape
    row_masks = [int(sum(int(X[i, j]) << j for j in range(N))) for i in range(m)]

    def hamming_distance(mask_a, mask_b):
        return (mask_a ^ mask_b).bit_count()

    # 1. Greedy construction (Nearest Neighbor starting from state 0 = empty pan)
    route = []
    unvisited = list(range(m))
    current_state = 0

    while unvisited:
        best_idx = min(
            unvisited,
            key=lambda idx: (hamming_distance(row_masks[idx], current_state), row_masks[idx].bit_count()),
        )
        route.append(best_idx)
        current_state = row_masks[best_idx]
        unvisited.remove(best_idx)

    # 2. 2-Opt local refinement to reduce physical manipulations on the bench
    def route_cost(seq):
        cost = hamming_distance(0, row_masks[seq[0]])
        for k in range(len(seq) - 1):
            cost += hamming_distance(row_masks[seq[k]], row_masks[seq[k + 1]])
        return cost

    improved = True
    while improved:
        improved = False
        current_cost = route_cost(route)
        for i in range(m - 1):
            for j in range(i + 1, m):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                if route_cost(new_route) < current_cost:
                    route = new_route
                    current_cost = route_cost(route)
                    improved = True

    return X[route, :], [row_masks[r] for r in route], route_cost(route)


# =======================================================================
# 3. INTERFACE AND BENCHTOP OPERATIONAL ROUTINE
# =======================================================================
def main():
    N = len(ITEMS)
    p = N - 1
    m = M_TARGET

    if N < 2:
        print(f'{COLOR_ERROR}Error: Define at least 2 objects in ITEMS.{RESET}')
        return

    if m < N or m > (2 ** N) - 1:
        print(
            f'{COLOR_ERROR}Error: Budget m ({m}) must be between N ({N})'
            f' and 2^N - 1 ({(2**N)-1}).{RESET}'
        )
        return

    X_opt, info = generate_pascal_design(N, m)
    if X_opt is None:
        print(
            f'{COLOR_ERROR}Error: Could not construct an estimable design'
            f' for N={N} and m={m}.{RESET}'
        )
        return

    X_exec, seq_masks, total_movements = optimize_weighing_route(X_opt)

    # Theoretical classification of the operating point
    if info['isotropic_1st_order'] and abs(info['ratio_s_mu1'] - 1.0 / N) < 1e-6:
        regime_str = f'Clean Point on the Lagrange Ridge (s_bar/mu1 = 1/{N})'
    elif info['isotropic_1st_order']:
        regime_str = 'Strict Orbital Isotropy (Parasitic Leakage = 0)'
    else:
        regime_str = 'Sub-Orbital Fractional Budget (k < p)'

    print(f'{COLOR_TITLE}====================================================================={RESET}')
    print(f'{COLOR_TITLE} PASCAL COMBINATORIAL WEIGHING PROCEDURE{RESET}')
    print(f'{COLOR_TITLE}====================================================================={RESET}')
    print(f" Object of Interest (m1) : {COLOR_PROMPT}{ITEMS[0]}{RESET}")
    print(f" Auxiliary Masses (p={p}) : {', '.join(ITEMS[1:])}")
    print(f" Total Budget (m)        : {m} weighings (Block A: {info['mA']} | Block B: {info['mB']})")
    print(f" Spectral Regime         : {COLOR_HIGHLIGHT}{regime_str}{RESET}")
    print(f" Net Information (Phi)   : {info['phi']:.5f} (Collective Mode: -{info['p_collective']:.5f} | Leakage: -{info['p_leakage']:.5f})")
    print(f" Factor u(m1)/sigma      : {COLOR_RESULT}{info['sig1_factor']:.5f} x sigma{RESET}")
    print(f" Benchtop Movements      : {total_movements} optimized add/remove actions (Gray + 2-Opt)")
    print(f'{COLOR_TITLE}====================================================================={RESET}\n')
    print(
        f'{COLOR_ERROR}WARNING: Ensure the balance is EMPTY and TARED'
        f' before starting.{RESET}\n'
    )

    measurements = [0.0] * m
    i = 0
    balance_state = 0

    while i < m:
        next_state = seq_masks[i]

        items_to_add = [
            ITEMS[b]
            for b in range(N)
            if (next_state & (1 << b)) and not (balance_state & (1 << b))
        ]
        items_to_remove = [
            ITEMS[b]
            for b in range(N)
            if (balance_state & (1 << b)) and not (next_state & (1 << b))
        ]
        items_on_pan = [
            ITEMS[b] for b in range(N) if (next_state & (1 << b))
        ]

        actions = []
        if items_to_add:
            actions.append(f"{COLOR_ADD}[+] PLACE : {', '.join(items_to_add)}{RESET}")
        if items_to_remove:
            actions.append(f"{COLOR_REMOVE}[-] REMOVE: {', '.join(items_to_remove)}{RESET}")

        action_msg = ' | '.join(actions) if actions else 'Keep the same objects on the pan'
        print(f'\n{COLOR_TITLE}--- Weighing {i + 1} of {m} ---{RESET}')
        print(f'-> {action_msg}')
        print(f"   {COLOR_HIGHLIGHT}Pan must contain: [ {', '.join(items_on_pan)} ]{RESET}")

        while True:
            user_input = input(
                f"{COLOR_PROMPT}Enter balance reading (or 'z' to undo previous step): {RESET}"
            ).strip().lower()

            if user_input == 'z':
                if i == 0:
                    print(f'{COLOR_ERROR}You are already at the first weighing.{RESET}')
                    continue
                i -= 1
                balance_state = seq_masks[i - 1] if i > 0 else 0
                reverted_items = [
                    ITEMS[b] for b in range(N) if (balance_state & (1 << b))
                ]
                rev_text = ', '.join(reverted_items) if reverted_items else 'EMPTY'
                print(f'\n{COLOR_WARNING}<<< STEPPING BACK <<< {RESET}')
                print(f'{COLOR_WARNING}Adjust the balance to contain ONLY: [ {rev_text} ]{RESET}')
                break

            try:
                weight = float(user_input.replace(',', '.'))
                measurements[i] = weight
                balance_state = next_state
                i += 1
                break
            except ValueError:
                print(f"{COLOR_ERROR}Invalid input! Enter a real number or 'z'.{RESET}")

    # =======================================================================
    # 4. METROLOGICAL RESOLUTION (GAUSS-MARKOV + BENCHTOP CLOSED-FORM)
    # =======================================================================
    print(f'\n{COLOR_TITLE}====================================================================={RESET}')
    print(f'{COLOR_TITLE} METROLOGICAL RESULTS AND SPECTRAL DIAGNOSTICS{RESET}')
    print(f'{COLOR_TITLE}====================================================================={RESET}\n')

    y = np.array(measurements, dtype=float)
    X = X_exec

    # 1. General Gauss-Markov Matrix Estimator (Least Squares)
    beta_hat, _, rank_X, _ = np.linalg.lstsq(X, y, rcond=None)
    XtX = X.T @ X
    pinv_XtX = np.linalg.pinv(XtX)

    residuals = y - X @ beta_hat
    residual_sum_sq = float(np.sum(residuals ** 2))
    degrees_of_freedom = int(m - rank_X)

    if degrees_of_freedom > 0:
        balance_sigma = float(np.sqrt(residual_sum_sq / degrees_of_freedom))
        cov_beta = (balance_sigma ** 2) * pinv_XtX
        u_beta = np.sqrt(np.maximum(0.0, np.diag(cov_beta)))
        uncertainty_available = True
    else:
        balance_sigma = 0.0
        u_beta = np.sqrt(np.maximum(0.0, np.diag(pinv_XtX)))
        uncertainty_available = False

    # 2. Analytical Estimator via Benchtop Closed-Form Formula
    mask_A_exec = X[:, 0] == 1.0
    w_aux_exec = np.sum(X[:, 1:], axis=1)
    sum_y_A = float(np.sum(y[mask_A_exec]))
    sum_weighted_aux = float(np.sum(w_aux_exec * y))
    m1_closed_form = (
        sum_y_A - info['ratio_s_mu1'] * sum_weighted_aux
    ) / info['phi']

    # Presentation of Primary Mass (m1)
    uncertainty_str = (
        f" ± {COLOR_WARNING}{u_beta[0]:.5f}{RESET}"
        if uncertainty_available
        else f" (factor: {u_beta[0]:.5f} x sigma)"
    )
    print(
        f"{COLOR_RESULT}>> OBJECT OF INTEREST (m1 - Gauss-Markov):{RESET} "
        f"{COLOR_PROMPT}{ITEMS[0]}{RESET} = "
        f"{COLOR_RESULT}{beta_hat[0]:.5f}{RESET}"
        f"{uncertainty_str}"
    )

    if info['isotropic_1st_order']:
        print(
            f"{COLOR_HIGHLIGHT}>> BENCHTOP CLOSED-FORM FORMULA (Matrix-Free):{RESET} "
            f"{ITEMS[0]} = {m1_closed_form:.5f} "
            f"(s_bar/mu1 = {info['ratio_s_mu1']:.5f})"
        )

    print(f'\n{COLOR_TITLE}>> SIMULTANEOUSLY CALIBRATED AUXILIARY MASSES:{RESET}')
    for k in range(1, N):
        txt_unc = (
            f' ± {u_beta[k]:.5f}'
            if uncertainty_available
            else f' (factor: {u_beta[k]:.5f} x sigma)'
        )
        print(f'   {ITEMS[k].ljust(18)}: {beta_hat[k]:.5f}{txt_unc}')

    print(f'\n{COLOR_TITLE}>> EXPERIMENT METROLOGICAL DIAGNOSTICS:{RESET}')
    theoretical_reduction = (1.0 - info['sig1_factor']) * 100.0
    print(f"   - Uncertainty factor u(m1)/sigma : {info['sig1_factor']:.5f} ({theoretical_reduction:.1f}% reduction vs. 1 direct weighing)")
    if uncertainty_available:
        print(f'   - Degrees of freedom (m - N)     : {degrees_of_freedom}')
        print(f'   - Balance standard deviation (s) : {balance_sigma:.5f} units')
    else:
        print('   - m = N: Saturated system (degrees of freedom = 0).')
        print('   - To estimate the balance standard deviation s experimentally, use m > N.')


if __name__ == '__main__':
    main()