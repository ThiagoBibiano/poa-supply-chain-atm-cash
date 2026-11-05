"""MILP estocástico de dois estágios (proposta, seção 4).

Equivalente determinístico por cenários, resolvido com HiGHS toda noite
(horizonte rolante de H = 10 dias, só o 1º estágio é executado).

Notação (seção 4.1–4.3):
- b ∈ 𝓑 agências (tesouraria + GO); j ∈ 𝓙_b ATMs internos; k ∈ 𝓚 externos;
  n ∈ 𝓝_c = 𝓑 ∪ 𝓚 pontos supridos pela central via transportadora;
  m ∈ {C, E} modalidade; ω ∈ Ω cenários com probabilidade π_ω.
- v^m_{n,t} visita (binária) pedida em t e executada em t + L_m; x^m_{n,t} valor;
  r_{n,t} recolhimento em t; y_{j,t}, q_{j,t}, s_{j,t} agenda do GO;
  β_{c,t}, b_{c,t} retirada na custódia; δ_{c,t}, g_{c,t} devolução;
  I, T, Z saldos de fim de dia; u, w, u^T, o, e, e^CD folgas.

Restrições implementadas: R1–R14 da proposta. Decisões de implementação
(documentadas no README):
- R9 (não antecipatividade) é imposta compartilhando as variáveis do 1º
  estágio entre os cenários: pedidos à transportadora emitidos hoje,
  recolhimentos e agenda do GO de amanhã e pedido/devolução à custódia de hoje.
- Devolução à custódia g_{c,t}: emitida em t, sai da central no dia de chegada
  a(t) = t + L_BC úteis (mesma defasagem da retirada).
- Capacidade na carga (R7b): saldo da manhã ≤ Cap_a depois do abastecimento.
  Nos ATMs externos, o que não couber volta à central (ov), com penalidade p^o
  — evita inviabilidade quando um pedido já emitido encontra o ATM cheio.
- R12 com folga penalizada (p^o) e "falta para despachar" na central com
  penalidade alta: o estado real herdado pode tornar a versão rígida inviável.
- Integralidade só nos primeiros ``dias_inteiros`` dias; o resto relaxado
  (seção 4.6, "Escala").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pulp

from .calendario import CalendarioOperacional
from .config import Config
from .estado import Decisao, Decisoes, EstadoRede
from .pulp_compat import contar, expr, n_restricoes, nova_var, resolver, restricao, valor
from .rede import RedePiloto


@dataclass
class InstanciaMILP:
    k0: int
    H: int
    estado: EstadoRede
    D: np.ndarray                     # (S, n_atms, H) saques
    probs: np.ndarray                 # (S,)
    G: np.ndarray | None = None       # (S, n_atms, H) depósitos (recicladores)
    F: np.ndarray | None = None       # (S, n_tes, H) guichê líquido
    EM: dict[str, list[float]] | None = None    # tesouraria → EM_{b,τ}, τ = 1..H
    EMCD: list[float] | None = None
    rotulo: str = ""


@dataclass
class SolucaoMILP:
    decisoes: Decisoes
    stats: dict
    duais: list[dict] = field(default_factory=list)
    # valores por "papel" (tipo, ponto, dia) no cenário 0 — usados como partida a quente
    papeis: dict = field(default_factory=dict)


def to_int(v) -> bool:
    """A variável é inteira (binária)?"""
    cat = getattr(v, "cat", None)
    return cat in ("Integer", "Binary") or str(cat).endswith("Integer")


def _nm(s: str) -> str:
    return s.replace(".", "_").replace(":", "_")


class ModeloNumerario:
    def __init__(self, cfg: Config, rede: RedePiloto, cal: CalendarioOperacional):
        self.cfg, self.rede, self.cal = cfg, rede, cal

    # ------------------------------------------------------------------
    def resolver(self, inst: InstanciaMILP, inicial: dict | None = None) -> SolucaoMILP:
        """Resolve a instância. ``inicial``: valores por papel de uma solução
        anterior (ex.: a determinística da mediana) para a partida a quente."""
        t_ini = time.time()
        cfg, rede, cal = self.cfg, self.rede, self.cal
        c = cfg.custos
        sv = cfg.solver
        k0, H = inst.k0, inst.H
        S_ = inst.D.shape[0]
        Om = range(S_)
        pi = inst.probs
        atms = rede.atms
        ia = {a: i for i, a in enumerate(atms)}
        tes = rede.tesourarias
        ib = {b: i for i, b in enumerate(tes)}
        ext = rede.externos
        pts = rede.pontos_transp
        L = {"C": c.L_C, "E": c.L_E}
        K = {"C": c.K_C, "E": c.K_E}
        est = inst.estado
        Theta = rede.Theta
        tcf = rede.theta_cf
        di = sv.dias_inteiros
        h = c.h
        rec = rede.recicladores

        def dia_ok(tau: int) -> bool:
            return 0 <= k0 + tau < len(cal)

        go = {tau: bool(cal.go[k0 + tau]) if dia_ok(tau) else False for tau in range(0, H + 1)}
        tr = {tau: bool(cal.transp[k0 + tau]) if dia_ok(tau) else False for tau in range(0, H + 1)}

        def visita_possivel(n: str, tau: int) -> bool:
            if tau < 1 or tau > H:
                return False
            return (tr[tau] and go[tau]) if n in ib else tr[tau]

        M = {n: (rede.teto[n] if n in ib else rede.cap[n]) for n in pts}
        # pendências herdadas
        pend_x = {(n, tau): dict(est.visitas.get(k0 + tau, {}).get(n, {})) for n in pts for tau in range(1, H + 1)}
        pend_n = {(n, tau): len(pend_x[(n, tau)]) for n in pts for tau in range(1, H + 1)}
        bc_in = {tau: est.bc_chegada.get(k0 + tau, 0.0) for tau in range(1, H + 1)}
        bc_out = {tau: est.bc_saida.get(k0 + tau, 0.0) for tau in range(1, H + 1)}
        proc = {tau: est.processamento.get(k0 + tau, 0.0) for tau in range(1, H + 1)}

        prob = pulp.LpProblem(f"numerario_{inst.rotulo}_{k0}", pulp.LpMinimize)
        V = lambda nome, lb=0.0, ub=None, cat="Continuous": nova_var(prob, nome, lb, ub, cat)  # noqa: E731

        # ---------------- variáveis do 1º estágio e dos cenários ----------------
        # visitas da transportadora: v[(n, m, τo, ω)]; τo = 0 compartilhada
        v, x = {}, {}
        for n in pts:
            for m in ("C", "E"):
                for to in range(0, H - L[m] + 1):
                    if not visita_possivel(n, to + L[m]):
                        continue
                    # Dominância: no 2º estágio cada cenário enxerga o futuro, então
                    # a especial pedida em τo ≥ 2 perde para a convencional pedida
                    # em τo − 1 (mesmo dia de execução, mais barata). Só a especial
                    # de hoje e a de amanhã são recurso de fato.
                    if m == "E" and to >= 2 and sv.especial_so_recurso:
                        continue
                    cat = "Binary" if to < di else "Continuous"
                    ub = None if cat == "Binary" else 1.0
                    if to == 0:
                        vv = V(f"v_{_nm(n)}_{m}_0", 0.0, ub, cat)
                        xx = V(f"x_{_nm(n)}_{m}_0")
                        for w in Om:
                            v[(n, m, 0, w)] = vv; x[(n, m, 0, w)] = xx
                    else:
                        for w in Om:
                            v[(n, m, to, w)] = V(f"v_{_nm(n)}_{m}_{to}_{w}", 0.0, ub, cat)
                            x[(n, m, to, w)] = V(f"x_{_nm(n)}_{m}_{to}_{w}")
        # entregas de pedidos pendentes (≤ valor pedido)
        xp = {}
        for (n, tau), d in pend_x.items():
            tot = float(sum(d.values()))
            if tot > 0:
                for w in Om:
                    xp[(n, tau, w)] = V(f"xp_{_nm(n)}_{tau}_{w}", 0.0, tot) if tau > 1 else None
                if tau == 1:
                    xx = V(f"xp_{_nm(n)}_1", 0.0, tot)
                    for w in Om:
                        xp[(n, 1, w)] = xx

        def nu_termos(n, tau, w):
            """(constante pendente, termos de visita) do que executa em τ."""
            ter = []
            for m in ("C", "E"):
                to = tau - L[m]
                if (n, m, to, w) in v:
                    ter.append((v[(n, m, to, w)], 1.0))
            return pend_n.get((n, tau), 0), ter

        def S_termos(n, tau, w):
            ter = []
            for m in ("C", "E"):
                to = tau - L[m]
                if (n, m, to, w) in x:
                    ter.append((x[(n, m, to, w)], 1.0))
            if (n, tau, w) in xp and xp[(n, tau, w)] is not None:
                ter.append((xp[(n, tau, w)], 1.0))
            return ter

        # recolhimentos r[(n, τ, ω)]
        r = {}
        for n in pts:
            for tau in range(1, H + 1):
                possivel = pend_n.get((n, tau), 0) > 0 or any(
                    (n, m, tau - L[m], 0) in v for m in ("C", "E"))
                if not possivel:
                    continue
                if tau == 1:
                    rr = V(f"r_{_nm(n)}_1")
                    for w in Om:
                        r[(n, 1, w)] = rr
                else:
                    for w in Om:
                        r[(n, tau, w)] = V(f"r_{_nm(n)}_{tau}_{w}")

        # GO: y, q, s
        y, q, s = {}, {}, {}
        for b in tes:
            for j in rede.internos[b]:
                for tau in range(1, H + 1):
                    if not go[tau]:
                        continue
                    cat = "Binary" if tau <= di else "Continuous"
                    ub = None if cat == "Binary" else 1.0
                    if tau == 1:
                        yy = V(f"y_{_nm(j)}_1", 0.0, ub, cat); qq = V(f"q_{_nm(j)}_1")
                        ss = V(f"s_{_nm(j)}_1") if j in rec else None
                        for w in Om:
                            y[(j, 1, w)] = yy; q[(j, 1, w)] = qq; s[(j, 1, w)] = ss
                    else:
                        for w in Om:
                            y[(j, tau, w)] = V(f"y_{_nm(j)}_{tau}_{w}", 0.0, ub, cat)
                            q[(j, tau, w)] = V(f"q_{_nm(j)}_{tau}_{w}")
                            s[(j, tau, w)] = V(f"s_{_nm(j)}_{tau}_{w}") if j in rec else None

        # custódia: β, b, δ, g por dia de pedido τo (chegada/saída em a(τo))
        bet, bb, dlt, gg, chega = {}, {}, {}, {}, {}
        for to in range(0, H):
            kk = k0 + to
            if not cal.pode_pedir_bc(kk):
                continue
            a = cal.chegada_bc(kk) - k0
            if a > H:
                continue
            chega[to] = a
            cat = "Binary" if to < di else "Continuous"
            ub = None if cat == "Binary" else 1.0
            if to == 0:
                b1, b2, d1, d2 = V("beta_0", 0.0, ub, cat), V("b_0"), V("delta_0", 0.0, ub, cat), V("g_0")
                for w in Om:
                    bet[(to, w)], bb[(to, w)], dlt[(to, w)], gg[(to, w)] = b1, b2, d1, d2
            else:
                for w in Om:
                    bet[(to, w)] = V(f"beta_{to}_{w}", 0.0, ub, cat); bb[(to, w)] = V(f"b_{to}_{w}")
                    dlt[(to, w)] = V(f"delta_{to}_{w}", 0.0, ub, cat); gg[(to, w)] = V(f"g_{to}_{w}")

        # ---------------- restrições ----------------
        obj = []
        for w in Om:
            pw = float(pi[w])
            I_prev = {a: float(est.I[a]) for a in atms}
            T_prev = {b: float(est.T[b]) for b in tes}
            Z_prev = float(est.Z)
            for tau in range(1, H + 1):
                ti = tau - 1
                # ---- estado do dia
                I = {a: V(f"I_{_nm(a)}_{tau}_{w}", 0.0, rede.cap[a]) for a in atms}
                u = {a: V(f"u_{_nm(a)}_{tau}_{w}", 0.0, float(max(inst.D[w, ia[a], ti], 0.0))) for a in atms}
                wv = {}
                if inst.G is not None:
                    for a in atms:
                        if a in rec:
                            wv[a] = V(f"w_{_nm(a)}_{tau}_{w}", 0.0, float(max(inst.G[w, ia[a], ti], 0.0)))
                T = {b: V(f"T_{_nm(b)}_{tau}_{w}") for b in tes}
                o = {b: V(f"o_{_nm(b)}_{tau}_{w}") for b in tes}
                e = {b: V(f"e_{_nm(b)}_{tau}_{w}") for b in tes} if inst.EM is not None else {}
                uT = {}
                if inst.F is not None:
                    for b in tes:
                        lim = float(max(-inst.F[w, ib[b], ti], 0.0))
                        if lim > 0:
                            uT[b] = V(f"uT_{_nm(b)}_{tau}_{w}", 0.0, lim)
                Z = V(f"Z_{tau}_{w}")
                oCD = V(f"oCD_{tau}_{w}")
                # folgas de viabilidade só onde o despacho já está decidido
                # (pendências e pedidos de hoje: τ ≤ L_C); depois disso o próprio
                # cenário ajusta o despacho
                fixo = tau <= c.L_C
                fcd = V(f"fcd_{tau}_{w}") if fixo else None
                eCD = V(f"eCD_{tau}_{w}") if inst.EMCD is not None else None
                ov = {k: V(f"ov_{_nm(k)}_{tau}_{w}") for k in ext if fixo and S_termos(k, tau, w)}

                # ---- R1: ATMs internos
                for b in tes:
                    for j in rede.internos[b]:
                        dj = float(inst.D[w, ia[j], ti])
                        gj = float(inst.G[w, ia[j], ti]) if (inst.G is not None and j in rec) else 0.0
                        ter = [(I[j], 1.0), (u[j], -1.0)]
                        if (j, tau, w) in q:
                            ter.append((q[(j, tau, w)], -1.0))
                            if s.get((j, tau, w)) is not None:
                                ter.append((s[(j, tau, w)], 1.0))
                        if j in wv:
                            ter.append((wv[j], 1.0))
                        rhs = I_prev[j] + gj - dj if isinstance(I_prev[j], float) else None
                        if rhs is not None:
                            prob += expr(ter) == rhs
                        else:
                            prob += expr(ter + [(I_prev[j], -1.0)]) == gj - dj
                        # R7b: capacidade na carga (manhã) e R6: ligação com y
                        if (j, tau, w) in q and (tau > 1 or w == 0):
                            ter_c = [(q[(j, tau, w)], 1.0)]
                            if s.get((j, tau, w)) is not None:
                                ter_c.append((s[(j, tau, w)], -1.0))
                            if isinstance(I_prev[j], float):
                                prob += expr(ter_c) <= rede.cap[j] - I_prev[j]
                            else:
                                prob += expr(ter_c + [(I_prev[j], 1.0)]) <= rede.cap[j]
                            ter_y = [(q[(j, tau, w)], 1.0), (y[(j, tau, w)], -rede.cap[j])]
                            if s.get((j, tau, w)) is not None:
                                ter_y.append((s[(j, tau, w)], 1.0))
                            prob += expr(ter_y) <= 0
                # ---- R2: ATMs externos
                for k in ext:
                    dk = float(inst.D[w, ia[k], ti])
                    gk = float(inst.G[w, ia[k], ti]) if (inst.G is not None and k in rec) else 0.0
                    Ster = S_termos(k, tau, w)
                    ter = [(I[k], 1.0), (u[k], -1.0)] + [(vv, -cc) for vv, cc in Ster]
                    if k in ov:
                        ter.append((ov[k], 1.0))
                    if (k, tau, w) in r:
                        ter.append((r[(k, tau, w)], 1.0))
                    if k in wv:
                        ter.append((wv[k], 1.0))
                    if isinstance(I_prev[k], float):
                        prob += expr(ter) == I_prev[k] + gk - dk
                    else:
                        prob += expr(ter + [(I_prev[k], -1.0)]) == gk - dk
                    # R7b: carga cabe no ATM (o excedente volta: ov)
                    if Ster:
                        ter_c = list(Ster) + ([(r[(k, tau, w)], -1.0)] if (k, tau, w) in r else []) + \
                            ([(ov[k], -1.0)] if k in ov else [])
                        if isinstance(I_prev[k], float):
                            prob += expr(ter_c) <= rede.cap[k] - I_prev[k]
                        else:
                            prob += expr(ter_c + [(I_prev[k], 1.0)]) <= rede.cap[k]
                        if k in ov:
                            prob += expr([(ov[k], 1.0)] + [(vv, -cc) for vv, cc in Ster]) <= 0
                    # recolhimento ≤ saldo antes da visita
                    if (k, tau, w) in r and (tau > 1 or w == 0):
                        if isinstance(I_prev[k], float):
                            prob += expr([(r[(k, tau, w)], 1.0)]) <= I_prev[k]
                        else:
                            prob += expr([(r[(k, tau, w)], 1.0), (I_prev[k], -1.0)]) <= 0
                # ---- R3, R4, R5: tesourarias
                for b in tes:
                    js = rede.internos[b]
                    Fb = float(inst.F[w, ib[b], ti]) if inst.F is not None else 0.0
                    Ster = S_termos(b, tau, w)
                    rter = [(r[(b, tau, w)], 1.0)] if (b, tau, w) in r else []
                    qter = [(q[(j, tau, w)], 1.0) for j in js if (j, tau, w) in q]
                    ster = [(s[(j, tau, w)], 1.0) for j in js if s.get((j, tau, w)) is not None]
                    ter = [(T[b], 1.0)] + [(vv, -cc) for vv, cc in Ster] + rter + qter + [(vv, -1.0) for vv, _ in ster]
                    if b in uT:
                        ter.append((uT[b], -1.0))
                    if isinstance(T_prev[b], float):
                        prob += expr(ter) == T_prev[b] + Fb
                    else:
                        prob += expr(ter + [(T_prev[b], -1.0)]) == Fb
                    # R4: o GO só abastece com o que já está no cofre
                    if qter and (tau > 1 or w == 0):
                        ter4 = qter + [(vv, -cc) for vv, cc in Ster] + rter + [(vv, -1.0) for vv, _ in ster]
                        nome = f"R4_{_nm(b)}_1" if tau == 1 else None
                        if isinstance(T_prev[b], float):
                            cons = expr(ter4) <= T_prev[b]
                        else:
                            cons = expr(ter4 + [(T_prev[b], -1.0)]) <= 0
                        if nome:
                            prob += cons, nome
                        else:
                            prob += cons
                    # R5: capacidade do GO (minutos), inclusive receber a transportadora
                    if go[tau] and (tau > 1 or w == 0):
                        npend, nter = nu_termos(b, tau, w)
                        ter5 = [(y[(j, tau, w)], rede.theta[j]) for j in js if (j, tau, w) in y] + [(vv, tcf) for vv, _ in nter]
                        if ter5:
                            cons = expr(ter5) <= Theta - tcf * npend
                            if tau == 1:
                                prob += cons, f"R5_{_nm(b)}_1"
                            else:
                                prob += cons
                    # recolhimento ≤ saldo antes da visita
                    if rter and (tau > 1 or w == 0):
                        if isinstance(T_prev[b], float):
                            prob += expr(rter) <= T_prev[b]
                        else:
                            prob += expr(rter + [(T_prev[b], -1.0)]) <= 0
                    # R7: teto flexível
                    prob += expr([(T[b], 1.0), (o[b], -1.0)]) <= rede.teto[b]
                    # R10: estoque mínimo flexível
                    if e:
                        prob += expr([(T[b], 1.0), (e[b], 1.0)]) >= float(inst.EM[b][ti])
                # ---- R6: ligações volume–decisão (transportadora), uma vez por variável
                for n in pts:
                    for m in ("C", "E"):
                        to = tau - L[m]
                        key = (n, m, to, w)
                        if key in x and (to > 0 or w == 0):
                            prob += expr([(x[key], 1.0), (v[key], -M[n])]) <= 0
                    if (n, tau, w) in r and (tau > 1 or w == 0):
                        npend, nter = nu_termos(n, tau, w)
                        prob += expr([(r[(n, tau, w)], 1.0)] + [(vv, -M[n]) for vv, _ in nter]) <= M[n] * npend
                # ---- R11–R14: tesouraria central
                ter = [(Z, 1.0)] + ([(fcd, -1.0)] if fcd is not None else [])
                cte = bc_in[tau] - bc_out[tau] + proc[tau]
                for to, a in chega.items():
                    if a == tau:
                        ter += [(bb[(to, w)], -1.0), (gg[(to, w)], 1.0)]
                for n in pts:
                    if tau - c.L_P >= 1 and (n, tau - c.L_P, w) in r:
                        ter.append((r[(n, tau - c.L_P, w)], -1.0))
                    for vv, cc in S_termos(n, tau, w):
                        ter.append((vv, cc))
                    if n in ov:
                        ter.append((ov[n], -1.0))
                if isinstance(Z_prev, float):
                    prob += expr(ter) == Z_prev + cte
                else:
                    prob += expr(ter + [(Z_prev, -1.0)]) == cte
                prob += expr([(Z, 1.0), (oCD, -1.0)]) <= rede.Zmax          # R12 (flexível)
                if eCD is not None:
                    prob += expr([(Z, 1.0), (eCD, 1.0)]) >= float(inst.EMCD[ti])  # R14
                # R13: ligação da custódia (calendário já filtrado na criação)
                for to in chega:
                    if (to == 0 and w == 0 and tau == 1) or (to > 0 and tau == 1):
                        prob += expr([(bb[(to, w)], 1.0), (bet[(to, w)], -rede.Zmax)]) <= 0
                        prob += expr([(gg[(to, w)], 1.0), (dlt[(to, w)], -rede.Zmax)]) <= 0

                # ---- objetivo do dia (ponderado por π_ω)
                for a in atms:
                    obj.append((I[a], pw * h))
                    obj.append((u[a], pw * c.p))
                    if a in wv:
                        obj.append((wv[a], pw * c.p_w))
                for b in tes:
                    obj += [(T[b], pw * h), (o[b], pw * c.p_o)]
                    if e:
                        obj.append((e[b], pw * c.p_EM))
                    if b in uT:
                        obj.append((uT[b], pw * c.p_T))
                obj += [(Z, pw * h), (oCD, pw * c.p_o)]
                if fcd is not None:
                    obj.append((fcd, pw * c.p_falta_central))
                if eCD is not None:
                    obj.append((eCD, pw * c.p_EM))
                for k, vv in ov.items():
                    obj.append((vv, pw * (c.p_o + c.kappa)))
                for n in pts:
                    for vv, cc in S_termos(n, tau, w):
                        obj.append((vv, pw * c.kappa))
                    if (n, tau, w) in r:
                        obj.append((r[(n, tau, w)], pw * (c.kappa + c.c_P)))
                    npend, nter = nu_termos(n, tau, w)
                    if n in ib:
                        for vv, _ in nter:
                            obj.append((vv, pw * c.c_GO * tcf))
                for b in tes:
                    for j in rede.internos[b]:
                        if (j, tau, w) in y:
                            obj.append((y[(j, tau, w)], pw * c.c_GO * rede.theta[j]))
                # avança o estado
                I_prev, T_prev, Z_prev = I, T, Z

            # custos fixos de visitas e da custódia (por dia de pedido)
            for (n, m, to, ww), vv in v.items():
                if ww == w:
                    obj.append((vv, pw * K[m]))
            for to in chega:
                obj += [(bet[(to, w)], pw * c.K_BC), (dlt[(to, w)], pw * c.K_BC),
                        (bb[(to, w)], pw * c.c_BC), (gg[(to, w)], pw * c.c_rBC)]

        prob += expr(obj)
        t_build = time.time() - t_ini

        # ---------------- partida a quente ----------------
        def _papeis_vars():
            for (n, m, to, w), vv in v.items():
                yield ("v", n, m, to), w, vv
                yield ("x", n, m, to), w, x[(n, m, to, w)]
            for (n, tau, w), vv in r.items():
                yield ("r", n, tau), w, vv
            for (j, tau, w), vv in y.items():
                yield ("y", j, tau), w, vv
                yield ("q", j, tau), w, q[(j, tau, w)]
                if s.get((j, tau, w)) is not None:
                    yield ("s", j, tau), w, s[(j, tau, w)]
            for (to, w), vv in bet.items():
                yield ("beta", to), w, vv
                yield ("b", to), w, bb[(to, w)]
                yield ("delta", to), w, dlt[(to, w)]
                yield ("g", to), w, gg[(to, w)]

        usar_ws = False
        if inicial:
            for papel, w, vv in _papeis_vars():
                if papel in inicial:
                    val = float(inicial[papel])
                    lb = vv.lowBound if vv.lowBound is not None else -np.inf
                    ub = vv.upBound if vv.upBound is not None else np.inf
                    val = min(max(val, lb), ub)
                    if papel[0] in ("v", "y", "beta", "delta") and to_int(vv):
                        val = float(round(val))
                    vv.setInitialValue(val)
                    usar_ws = True
        nvar, nint = contar(prob)
        st = resolver(prob, sv.tempo_limite_s, sv.gap_relativo, sv.threads, sv.msg,
                      opcoes=sv.opcoes_highs, warm_start=usar_ws)
        st.update(t_construcao=t_build, variaveis=nvar, binarias=nint, cenarios=S_,
                  restricoes=n_restricoes(prob))
        if not st["ok"]:
            st["erro"] = "sem solução"
            return SolucaoMILP(Decisoes(k0=k0, info={"milp": st}), st)

        # ---------------- extração ----------------
        dec = Decisoes(k0=k0)
        for n in pts:
            for m in ("C", "E"):
                key = (n, m, 0, 0)
                if key in v and valor(v[key]) > 0.5:
                    dec.pedidos.setdefault(n, {})[m] = max(0.0, valor(x[key]))
            if (n, 1, 0) in r:
                rv = valor(r[(n, 1, 0)])
                if rv > 1e-6:
                    dec.recolher[n] = rv
        for b in tes:
            for j in rede.internos[b]:
                if (j, 1, 0) in y and valor(y[(j, 1, 0)]) > 0.5:
                    qq = valor(q[(j, 1, 0)])
                    ss = valor(s[(j, 1, 0)]) if s.get((j, 1, 0)) is not None else 0.0
                    if qq > 1e-6 or ss > 1e-6:
                        dec.go[j] = (qq, ss)
        if 0 in chega:
            if valor(bet[(0, 0)]) > 0.5:
                dec.bc_retirada = valor(bb[(0, 0)])
            if valor(dlt[(0, 0)]) > 0.5:
                dec.bc_devolucao = valor(gg[(0, 0)])

        # plano (2º estágio): valores esperados nos cenários
        plano = []
        for n in pts:
            for m in ("C", "E"):
                for to in range(0, H - L[m] + 1):
                    if (n, m, to, 0) not in v:
                        continue
                    pv = sum(pi[w] * valor(v[(n, m, to, w)]) for w in Om)
                    if pv < 0.05:
                        continue
                    xv = sum(pi[w] * valor(x[(n, m, to, w)]) for w in Om)
                    plano.append(Decisao(n, k0, k0 + to + L[m], m, xv / max(pv, 1e-9), 0.0,
                                         1 if to == 0 else 2, pv))
            for tau in range(1, H + 1):
                if (n, tau, 0) in r:
                    rv = sum(pi[w] * valor(r[(n, tau, w)]) for w in Om)
                    if rv > 1e-3:
                        plano.append(Decisao(n, k0, k0 + tau, "REC", rv, 0.0, 1 if tau == 1 else 2, 1.0))
        for b in tes:
            for j in rede.internos[b]:
                for tau in range(1, H + 1):
                    if (j, tau, 0) not in y:
                        continue
                    py = sum(pi[w] * valor(y[(j, tau, w)]) for w in Om)
                    if py < 0.05:
                        continue
                    qv = sum(pi[w] * valor(q[(j, tau, w)]) for w in Om)
                    plano.append(Decisao(j, k0, k0 + tau, "GO", qv / max(py, 1e-9), rede.theta[j],
                                         1 if tau == 1 else 2, py))
        for to, a in chega.items():
            pb_ = sum(pi[w] * valor(bet[(to, w)]) for w in Om)
            if pb_ >= 0.05:
                plano.append(Decisao("CD", k0, k0 + a, "BC", sum(pi[w] * valor(bb[(to, w)]) for w in Om) / max(pb_, 1e-9),
                                     0.0, 1 if to == 0 else 2, pb_))
            pd_ = sum(pi[w] * valor(dlt[(to, w)]) for w in Om)
            if pd_ >= 0.05:
                plano.append(Decisao("CD", k0, k0 + a, "DEV", sum(pi[w] * valor(gg[(to, w)]) for w in Om) / max(pd_, 1e-9),
                                     0.0, 1 if to == 0 else 2, pd_))
        dec.plano = plano
        dec.info["milp"] = st
        papeis = {papel: valor(vv) for papel, w, vv in _papeis_vars() if w == 0}

        # ---------------- duais com as binárias fixadas ----------------
        duais = []
        if sv.calcular_duais:
            try:
                for vv in prob.variables():
                    if to_int(vv):
                        val = round(valor(vv))
                        vv.lowBound = val; vv.upBound = val; vv.cat = "Continuous"
                st2 = resolver(prob, sv.tempo_limite_s, None, sv.threads, sv.msg, mip=False,
                               opcoes=sv.opcoes_highs)
                if st2["ok"]:
                    for b in tes:
                        cons = restricao(prob, f"R5_{_nm(b)}_1")
                        if cons is not None:
                            duais.append(dict(restricao="R5", ponto=b, dual=float(cons.pi or 0.0),
                                              folga=float(cons.slack or 0.0)))
                        cons = restricao(prob, f"R4_{_nm(b)}_1")
                        if cons is not None:
                            duais.append(dict(restricao="R4", ponto=b, dual=float(cons.pi or 0.0),
                                              folga=float(cons.slack or 0.0)))
            except Exception as exc:  # duais são diagnósticos: não derrubam a rodada
                st["aviso_duais"] = str(exc)
        st["t_total"] = time.time() - t_ini
        st["partida_quente"] = usar_ws
        return SolucaoMILP(dec, st, duais, papeis)


__all__ = ["ModeloNumerario", "InstanciaMILP", "SolucaoMILP"]
