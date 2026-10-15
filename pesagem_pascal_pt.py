# =======================================================================
# Script: Pesagem Combinatória de Pascal para N e m Arbitrários (v2.0)
# =======================================================================
# Fundamentação Teórica:
# 1. Truncagem Bilateral do Triângulo de Pascal e Crista k_A = k_B + 1
# 2. Decomposição em Órbitas Cíclicas com Isotropia de 1ª e 2ª Ordens
#    (Seleção automática de Conjuntos de Diferenças / Plano de Fano)
# 3. Decomposição Espectral do Complemento de Schur e Fórmula Fechada
# 4. Roteirização de Bancada por Distância de Hamming Mínima + 2-Opt
# =======================================================================

import itertools
import math
import os
import numpy as np

# Habilita cores ANSI no terminal do Windows
if os.name == 'nt':
    os.system('color')

# Códigos de cor ANSI
COR_TITULO = '\033[96m'    # Ciano
COR_COLOCAR = '\033[92m'   # Verde
COR_RETIRAR = '\033[91m'   # Vermelho
COR_PROMPT = '\033[93m'    # Amarelo
COR_RESULT = '\033[92m'    # Verde
COR_ERRO = '\033[91m'      # Vermelho
COR_AVISO = '\033[95m'     # Magenta (Undo / Alerta)
COR_DESTAQUE = '\033[94m'  # Azul Claro
RESET = '\033[0m'

# =======================================================================
# 1. PARÂMETROS DE CONFIGURAÇÃO DO EXPERIMENTO
# =======================================================================
# O primeiro item da lista É OBRIGATORIAMENTE o corpo de interesse (m1).
# Os demais corpos atuam como massas auxiliares (m2, ..., mN).
ITENS = ['pendrive', 'chav_quad', 'lhama', 'cadeado', 'borracha', 'sapo']

# Orçamento total de pesagens desejado (m >= len(ITENS))
# Exemplos para N=6 (p=5): Ponto Limpo Estrela = 7; Intermediário = 22; Fatorial = 63
M_TARGET = 10


# =======================================================================
# 2. CONSTRUÇÃO DO DELINEAMENTO DE PASCAL E BLINDAGEM ESPECTRAL
# =======================================================================
def _metricas_isotropia_matriz(X_aux):
    """Calcula a dispersão de 1ª ordem Var(k_i) e de 2ª ordem Var(lambda_ij)
    para a submatriz de corpos auxiliares X_aux.
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


def decompor_camada_em_orbitas(p, r):
    """Decompõe a camada C(p, r) em órbitas do grupo cíclico C_p.
    Ordena as órbitas priorizando:
      1. Comprimento máximo (isotropia de 1ª ordem: Var(k_i) = 0)
      2. Menor variância de coocorrência de pares Var(lambda_ij) (isotropia
         de 2ª ordem: seleciona automaticamente conjuntos de diferenças
         perfeitos, como a semente de Fano {0, 1, 3} mod 7).
    """
    all_combs = list(itertools.combinations(range(p), r))
    vistos = set()
    orbitas_info = []

    for comb in all_combs:
        if comb in vistos:
            continue
        orb = []
        for shift in range(p):
            rot = tuple(sorted((x + shift) % p for x in comb))
            if rot not in orb:
                orb.append(rot)
                vistos.add(rot)

        # Matriz binária da órbita para avaliar isotropia de 2ª ordem
        X_orb = np.zeros((len(orb), p), dtype=float)
        for idx, c in enumerate(orb):
            X_orb[idx, list(c)] = 1.0
        var_k, var_lambda = _metricas_isotropia_matriz(X_orb)

        orbitas_info.append({
            'orb': orb,
            'X_orb': X_orb,
            'len': len(orb),
            'var_k': var_k,
            'var_lambda': var_lambda,
        })

    # Ordena: maior tamanho -> menor Var(k_i) -> menor Var(lambda_ij)
    orbitas_info.sort(
        key=lambda item: (-item['len'], round(item['var_k'], 9), round(item['var_lambda'], 9))
    )
    return orbitas_info


def selecionar_linhas_balanceadas(X_base, X_orb_cand, k_resto, N, inclui_m1):
    """Seleciona k_resto linhas de uma única órbita candidata minimizando
    conjuntamente a variância de 1ª ordem Var(k_i) e de 2ª ordem Var(lambda_ij).
    """
    if k_resto <= 0 or len(X_orb_cand) == 0:
        return X_base

    p = N - 1
    X_base_aux = X_base[:, 1:] if len(X_base) > 0 else np.empty((0, p), dtype=float)
    n_cand = len(X_orb_cand)
    indices = list(range(n_cand))

    melhor_combo = None
    melhor_chave = (np.inf, np.inf)

    # Como X_orb_cand vem de uma única órbita (n_cand <= p), C(p, k_resto) é pequeno
    if math.comb(n_cand, k_resto) <= 5000:
        for combo in itertools.combinations(indices, k_resto):
            X_trial_aux = np.vstack([X_base_aux, X_orb_cand[list(combo), :]])
            var_k, var_l = _metricas_isotropia_matriz(X_trial_aux)
            chave = (round(var_k, 9), round(var_l, 9))
            if chave < melhor_chave:
                melhor_chave = chave
                melhor_combo = list(combo)
    else:
        atuais = []
        disponiveis = indices.copy()
        X_running = X_base_aux.copy()
        for _ in range(k_resto):
            best_idx = None
            best_step_key = (np.inf, np.inf)
            for idx in disponiveis:
                X_trial = np.vstack([X_running, X_orb_cand[[idx], :]])
                var_k, var_l = _metricas_isotropia_matriz(X_trial)
                chave = (round(var_k, 9), round(var_l, 9))
                if chave < best_step_key:
                    best_step_key = chave
                    best_idx = idx
            atuais.append(best_idx)
            disponiveis.remove(best_idx)
            X_running = np.vstack([X_running, X_orb_cand[[best_idx], :]])
        melhor_combo = atuais

    sub_aux = X_orb_cand[melhor_combo, :]
    col_m1 = np.ones((k_resto, 1)) if inclui_m1 else np.zeros((k_resto, 1))
    linhas_escolhidas = np.hstack([col_m1, sub_aux])

    return np.vstack([X_base, linhas_escolhidas]) if len(X_base) > 0 else linhas_escolhidas


def construir_bloco_hierarquico(p, m_bloco, inclui_m1, N):
    """Constrói o Bloco A ou B seguindo a hierarquia estrita:
    1. Camadas binomiais completas do Triângulo de Pascal
    2. Órbitas cíclicas completas selecionadas por isotropia de 2ª ordem acumulada
    3. Subconjunto sub-orbital (k < p) extraído da melhor órbita residual
    """
    ordem_pesos = range(p + 1) if inclui_m1 else range(p, 0, -1)
    X_bloco = np.empty((0, N), dtype=float)
    orcamento_restante = m_bloco

    for r in ordem_pesos:
        if orcamento_restante <= 0:
            break

        total_camada = math.comb(p, r)

        # 1. Camada inteira cabe no orçamento
        if orcamento_restante >= total_camada:
            combs = list(itertools.combinations(range(p), r))
            linhas = np.zeros((total_camada, N), dtype=float)
            if inclui_m1:
                linhas[:, 0] = 1.0
            for idx, c in enumerate(combs):
                for bit in c:
                    linhas[idx, bit + 1] = 1.0
            X_bloco = np.vstack([X_bloco, linhas])
            orcamento_restante -= total_camada
            continue

        # 2. Camada parcial: selecionar órbitas completas minimizando Var(lambda_ij) acumulada
        orbitas_disponiveis = decompor_camada_em_orbitas(p, r)

        while orcamento_restante > 0 and orbitas_disponiveis:
            # Filtra órbitas que cabem inteiras no orçamento restante
            que_cabem = [
                (i, info) for i, info in enumerate(orbitas_disponiveis)
                if info['len'] <= orcamento_restante
            ]
            if not que_cabem:
                break

            # Escolhe a órbita que maximiza o tamanho e minimiza Var(k_i) e Var(lambda_ij) acumulados
            X_base_aux = X_bloco[:, 1:] if len(X_bloco) > 0 else np.empty((0, p), dtype=float)
            melhor_pos = None
            melhor_crit = (np.inf, np.inf, np.inf)

            for pos_lista, info in que_cabem:
                X_trial_aux = np.vstack([X_base_aux, info['X_orb']])
                var_k, var_l = _metricas_isotropia_matriz(X_trial_aux)
                crit = (-info['len'], round(var_k, 9), round(var_l, 9))
                if crit < melhor_crit:
                    melhor_crit = crit
                    melhor_pos = pos_lista

            escolhida = orbitas_disponiveis.pop(melhor_pos)
            tam = escolhida['len']
            col_m1 = np.ones((tam, 1)) if inclui_m1 else np.zeros((tam, 1))
            linhas_orb = np.hstack([col_m1, escolhida['X_orb']])
            X_bloco = np.vstack([X_bloco, linhas_orb])
            orcamento_restante -= tam

        # 3. Resto sub-orbital (1 <= orcamento_restante < p): extrai da melhor órbita restante
        if orcamento_restante > 0 and orbitas_disponiveis:
            melhor_orb_residual = orbitas_disponiveis[0]['X_orb']
            X_bloco = selecionar_linhas_balanceadas(
                X_bloco, melhor_orb_residual, orcamento_restante, N, inclui_m1
            )
            orcamento_restante = 0

    return X_bloco


def compute_uncertainty(X):
    """Calcula sigma_1 / sigma com teste exato de estimabilidade de Gauss-Markov."""
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


def decompor_espectro_schur(X):
    """Realiza a decomposição espectral da penalidade de Schur:
    Phi = m_A - (P_coletivo + P_vazamento)
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
    penalidade_total = float(s_vec.T @ np.linalg.pinv(M_aux) @ s_vec)
    phi = float(mA - penalidade_total)

    # Modo coletivo v1 = 1_p / sqrt(p)
    v1 = np.ones(p) / np.sqrt(p)
    mu1 = float(v1.T @ M_aux @ v1)  # Média da soma de linhas de M_aux
    p_coletivo = float(((v1.T @ s_vec) ** 2) / mu1) if mu1 > 0 else np.inf
    p_vazamento = max(0.0, penalidade_total - p_coletivo)

    # Verifica se v1 é autovetor exato de M_aux e se s_vec é uniforme
    residuo_autovetor = float(np.linalg.norm(M_aux @ v1 - mu1 * v1))
    isotropico_1a_ordem = (std_s < 1e-7) and (residuo_autovetor < 1e-7)

    return {
        'mA': mA,
        'mB': mB,
        's_bar': s_bar,
        'std_s': std_s,
        'mu1': mu1,
        'razao_s_mu1': s_bar / mu1 if mu1 > 0 else 0.0,
        'penalidade_total': penalidade_total,
        'p_coletivo': p_coletivo,
        'p_vazamento': p_vazamento,
        'phi': phi,
        'isotropico_1a_ordem': isotropico_1a_ordem,
    }


def gerar_delineamento_pascal(N, m_target):
    """Varre as partições (mA, mB) e seleciona a que maximiza a informação líquida Phi
    (desempatando pelo menor vazamento parasita).
    """
    p = N - 1
    best_key = (np.inf, np.inf)
    best_X = None
    best_info = None

    max_mB_possivel = (2 ** p) - 1
    min_mA = 1
    max_mA = min(2 ** p, m_target - 1)

    for mA in range(min_mA, max_mA + 1):
        mB = m_target - mA
        if mB < 1 or mB > max_mB_possivel:
            continue

        X_A = construir_bloco_hierarquico(p, mA, inclui_m1=True, N=N)
        X_B = construir_bloco_hierarquico(p, mB, inclui_m1=False, N=N)
        X_cand = np.vstack([X_A, X_B])

        sig = compute_uncertainty(X_cand)
        if np.isinf(sig):
            continue

        info = decompor_espectro_schur(X_cand)
        # Prioriza menor incerteza (6 casas) e desempata por menor vazamento espectral
        cand_key = (round(sig, 6), round(info['p_vazamento'], 6))
        if cand_key < best_key:
            best_key = cand_key
            best_X = X_cand
            best_info = info
            best_info['sig1_fator'] = sig

    if best_X is None:
        return None, None

    return best_X, best_info


def otimizar_rota_pesagens(X):
    """Reordena as pesagens minimizando a distância de Hamming acumulada
    (Vizinho Mais Próximo + refinamento local 2-Opt a partir da balança vazia).
    """
    m, N = X.shape
    row_masks = [int(sum(int(X[i, j]) << j for j in range(N))) for i in range(m)]

    def dist_hamming(mask_a, mask_b):
        return (mask_a ^ mask_b).bit_count()

    # 1. Construção gulosa (Vizinho Mais Próximo partindo do estado 0 = balança vazia)
    rota = []
    nao_visitados = list(range(m))
    estado_atual = 0

    while nao_visitados:
        melhor_idx = min(
            nao_visitados,
            key=lambda idx: (dist_hamming(row_masks[idx], estado_atual), row_masks[idx].bit_count()),
        )
        rota.append(melhor_idx)
        estado_atual = row_masks[melhor_idx]
        nao_visitados.remove(melhor_idx)

    # 2. Refinamento 2-Opt para reduzir trocas físicas na bancada
    def custo_rota(seq):
        custo = dist_hamming(0, row_masks[seq[0]])
        for k in range(len(seq) - 1):
            custo += dist_hamming(row_masks[seq[k]], row_masks[seq[k + 1]])
        return custo

    melhorou = True
    while melhorou:
        melhorou = False
        custo_atual = custo_rota(rota)
        for i in range(m - 1):
            for j in range(i + 1, m):
                nova_rota = rota[:i] + rota[i:j + 1][::-1] + rota[j + 1:]
                if custo_rota(nova_rota) < custo_atual:
                    rota = nova_rota
                    custo_atual = custo_rota(rota)
                    melhorou = True

    return X[rota, :], [row_masks[r] for r in rota], custo_rota(rota)


# =======================================================================
# 3. INTERFACE E ROTINA OPERACIONAL DE BANCADA
# =======================================================================
def main():
    N = len(ITENS)
    p = N - 1
    m = M_TARGET

    if N < 2:
        print(f'{COR_ERRO}Erro: Defina pelo menos 2 corpos em ITENS.{RESET}')
        return

    if m < N or m > (2 ** N) - 1:
        print(
            f'{COR_ERRO}Erro: O orçamento m ({m}) deve estar entre N ({N})'
            f' e 2^N - 1 ({(2**N)-1}).{RESET}'
        )
        return

    X_otimo, info = gerar_delineamento_pascal(N, m)
    if X_otimo is None:
        print(
            f'{COR_ERRO}Erro: Não foi possível construir um delineamento estimável'
            f' para N={N} e m={m}.{RESET}'
        )
        return

    X_exec, seq_mascaras, total_movimentos = otimizar_rota_pesagens(X_otimo)

    # Classificação teórica do ponto de operação
    if info['isotropico_1a_ordem'] and abs(info['razao_s_mu1'] - 1.0 / N) < 1e-6:
        regime_str = f'Ponto Limpo na Crista de Lagrange (s_bar/mu1 = 1/{N})'
    elif info['isotropico_1a_ordem']:
        regime_str = 'Isotropia Orbital Estrita (Vazamento Parasita = 0)'
    else:
        regime_str = 'Orçamento Fracionário Sub-Orbital (k < p)'

    print(f'{COR_TITULO}====================================================================={RESET}')
    print(f'{COR_TITULO} PROCEDIMENTO DE PESAGEM COMBINATÓRIA DE PASCAL{RESET}')
    print(f'{COR_TITULO}====================================================================={RESET}')
    print(f" Corpo de Interesse (m1) : {COR_PROMPT}{ITENS[0]}{RESET}")
    print(f" Corpos Auxiliares (p={p}) : {', '.join(ITENS[1:])}")
    print(f" Orçamento Total (m)     : {m} pesagens (Bloco A: {info['mA']} | Bloco B: {info['mB']})")
    print(f" Regime Espectral        : {COR_DESTAQUE}{regime_str}{RESET}")
    print(f" Informação Líquida (Phi): {info['phi']:.5f} (Modo Coletivo: -{info['p_coletivo']:.5f} | Vazamento: -{info['p_vazamento']:.5f})")
    print(f" Fator u(m1)/sigma       : {COR_RESULT}{info['sig1_fator']:.5f} x sigma{RESET}")
    print(f" Movimentos de Bancada   : {total_movimentos} ações de colocar/retirar otimizadas (Gray + 2-Opt)")
    print(f'{COR_TITULO}====================================================================={RESET}\n')
    print(
        f'{COR_ERRO}ATENÇÃO: Certifique-se de que a balança está VAZIA e TARADA'
        f' antes de iniciar.{RESET}\n'
    )

    medicoes = [0.0] * m
    i = 0
    estado_balanca = 0

    while i < m:
        proximo_estado = seq_mascaras[i]

        itens_colocar = [
            ITENS[b]
            for b in range(N)
            if (proximo_estado & (1 << b)) and not (estado_balanca & (1 << b))
        ]
        itens_retirar = [
            ITENS[b]
            for b in range(N)
            if (estado_balanca & (1 << b)) and not (proximo_estado & (1 << b))
        ]
        itens_no_prato = [
            ITENS[b] for b in range(N) if (proximo_estado & (1 << b))
        ]

        acoes = []
        if itens_colocar:
            acoes.append(f"{COR_COLOCAR}[+] COLOQUE: {', '.join(itens_colocar)}{RESET}")
        if itens_retirar:
            acoes.append(f"{COR_RETIRAR}[-] RETIRE : {', '.join(itens_retirar)}{RESET}")

        msg_acao = ' | '.join(acoes) if acoes else 'Mantenha os mesmos corpos'
        print(f'\n{COR_TITULO}--- Pesagem {i + 1} de {m} ---{RESET}')
        print(f'-> {msg_acao}')
        print(f"   {COR_DESTAQUE}Prato deve conter: [ {', '.join(itens_no_prato)} ]{RESET}")

        while True:
            entrada = input(
                f"{COR_PROMPT}Digite o peso lido (ou 'z' para desfazer a anterior): {RESET}"
            ).strip().lower()

            if entrada == 'z':
                if i == 0:
                    print(f'{COR_ERRO}Você já está na primeira pesagem.{RESET}')
                    continue
                i -= 1
                estado_balanca = seq_mascaras[i - 1] if i > 0 else 0
                itens_revertidos = [
                    ITENS[b] for b in range(N) if (estado_balanca & (1 << b))
                ]
                texto_rev = ', '.join(itens_revertidos) if itens_revertidos else 'VAZIA'
                print(f'\n{COR_AVISO}<<< VOLTANDO UM PASSO <<<{RESET}')
                print(f'{COR_AVISO}Ajuste a balança para conter APENAS: [ {texto_rev} ]{RESET}')
                break

            try:
                peso = float(entrada.replace(',', '.'))
                medicoes[i] = peso
                estado_balanca = proximo_estado
                i += 1
                break
            except ValueError:
                print(f"{COR_ERRO}Valor inválido! Digite um número real ou 'z'.{RESET}")

    # =======================================================================
    # 4. RESOLUÇÃO METROLÓGICA (GAUSS-MARKOV + FÓRMULA FECHADA DE BANCADA)
    # =======================================================================
    print(f'\n{COR_TITULO}====================================================================={RESET}')
    print(f'{COR_TITULO} RESULTADOS METROLÓGICOS E DIAGNÓSTICO ESPECTRAL{RESET}')
    print(f'{COR_TITULO}====================================================================={RESET}\n')

    y = np.array(medicoes, dtype=float)
    X = X_exec

    # 1. Estimador Matricial Geral de Gauss-Markov (Mínimos Quadrados)
    beta_hat, _, rank_X, _ = np.linalg.lstsq(X, y, rcond=None)
    XtX = X.T @ X
    pinv_XtX = np.linalg.pinv(XtX)

    residuos = y - X @ beta_hat
    soma_quad_res = float(np.sum(residuos ** 2))
    graus_liberdade = int(m - rank_X)

    if graus_liberdade > 0:
        sigma_balanca = float(np.sqrt(soma_quad_res / graus_liberdade))
        cov_beta = (sigma_balanca ** 2) * pinv_XtX
        u_beta = np.sqrt(np.maximum(0.0, np.diag(cov_beta)))
        incerteza_disponivel = True
    else:
        sigma_balanca = 0.0
        u_beta = np.sqrt(np.maximum(0.0, np.diag(pinv_XtX)))
        incerteza_disponivel = False

    # 2. Estimador Analítico pela Fórmula Fechada de Bancada (Seção 5.2 do Artigo)
    mask_A_exec = X[:, 0] == 1.0
    w_aux_exec = np.sum(X[:, 1:], axis=1)
    soma_y_A = float(np.sum(y[mask_A_exec]))
    soma_pond_aux = float(np.sum(w_aux_exec * y))
    m1_formula_fechada = (
        soma_y_A - info['razao_s_mu1'] * soma_pond_aux
    ) / info['phi']

    # Apresentação da Massa Principal (m1)
    print(
        f"{COR_RESULT}>> MASSA DE INTERESSE (m1 - Gauss-Markov):{RESET} "
        f"{COR_PROMPT}{ITENS[0]}{RESET} = "
        f"{COR_RESULT}{beta_hat[0]:.5f}{RESET}"
        + (
            f" ± {COR_AVISO}{u_beta[0]:.5f}{RESET}"
            if incerteza_disponivel
            else f" (fator: {u_beta[0]:.5f} * sigma)"
        )
    )

    if info['isotropico_1a_ordem']:
        print(
            f"{COR_DESTAQUE}>> FÓRMULA FECHADA DE BANCADA (Sem Matriz):{RESET} "
            f"{ITENS[0]} = {m1_formula_fechada:.5f} "
            f"(s_bar/mu1 = {info['razao_s_mu1']:.5f})"
        )

    print(f'\n{COR_TITULO}>> MASSAS AUXILIARES CALIBRADAS SIMULTANEAMENTE:{RESET}')
    for k in range(1, N):
        txt_inc = (
            f' ± {u_beta[k]:.5f}'
            if incerteza_disponivel
            else f' (fator: {u_beta[k]:.5f} * sigma)'
        )
        print(f'   {ITENS[k].ljust(18)}: {beta_hat[k]:.5f}{txt_inc}')

    print(f'\n{COR_TITULO}>> DIAGNÓSTICO METROLÓGICO DO EXPERIMENTO:{RESET}')
    reducao_teorica = (1.0 - info['sig1_fator']) * 100.0
    print(f"   - Fator de incerteza u(m1)/sigma : {info['sig1_fator']:.5f} (Redução de {reducao_teorica:.1f}% vs. 1 pesagem direta)")
    if incerteza_disponivel:
        print(f'   - Graus de liberdade (m - N)     : {graus_liberdade}')
        print(f'   - Desvio padrão da balança (s)   : {sigma_balanca:.5f} unidades')
    else:
        print('   - m = N: Sistema saturado (graus de liberdade = 0).')
        print('   - Para estimar o desvio s da balança experimentalmente, utilize m > N.')


if __name__ == '__main__':
    main()