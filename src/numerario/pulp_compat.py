"""Camada fina de compatibilidade com o PuLP.

O PuLP 4 (núcleo em Rust) mudou a API: variáveis nascem do problema
(``prob.add_variable``), ``solve`` devolve um objeto de estatísticas e
``LpStatus`` saiu. O projeto fixa ``pulp==4.2.0`` (Python >= 3.12) e cai para
``pulp==3.3.2`` em Python mais antigo; a 3.3 é de transição (já tem
``add_variable``, mas ainda devolve o status clássico). Esta camada detecta cada
recurso em vez de confiar só no número da versão, e aceita também o PuLP 2.8.
"""

from __future__ import annotations

import re

import pulp

PULP_4 = int(re.match(r"\d+", pulp.__version__).group()) >= 4
_ADD_VAR = hasattr(pulp.LpProblem, "add_variable")
_FROM_LIST = hasattr(pulp.LpAffineExpression, "from_list")
_GET_CONS = hasattr(pulp.LpProblem, "get_constraint_by_name")


def nova_var(prob: pulp.LpProblem, nome: str, lb: float | None = 0.0, ub: float | None = None,
             cat: str = "Continuous"):
    if _ADD_VAR:
        return prob.add_variable(nome, lowBound=lb, upBound=ub, cat=cat)
    return pulp.LpVariable(nome, lowBound=lb, upBound=ub, cat=cat)


def expr(termos: list[tuple], constante: float = 0.0):
    """Expressão linear a partir de pares (variável, coeficiente).

    No PuLP 4, ``lpSum`` com tuplas soma os elementos da tupla (não o produto);
    por isso a expressão é montada por ``from_list`` com coeficientes já somados
    para variáveis repetidas."""
    acc: dict = {}
    ordem = []
    for v, c in termos:
        if v is None:
            continue
        chave = id(v)
        if chave not in acc:
            acc[chave] = [v, 0.0]
            ordem.append(chave)
        acc[chave][1] += float(c)
    pares = [(acc[k][0], acc[k][1]) for k in ordem]
    if _FROM_LIST:
        return pulp.LpAffineExpression.from_list(pares, float(constante))
    return pulp.LpAffineExpression(pares, constant=float(constante))


def resolver(prob: pulp.LpProblem, tempo_limite: float | None, gap: float | None,
             threads: int | None = None, msg: bool = False, mip: bool = True,
             opcoes: dict | None = None, warm_start: bool = False) -> dict:
    """Resolve com HiGHS e devolve um dicionário uniforme de estatísticas.

    ``opcoes``: opções nativas do HiGHS (ex.: {"mip_heuristic_run_rins": False})."""
    kw = dict(msg=msg, timeLimit=tempo_limite, gapRel=gap, mip=mip)
    if threads:
        kw["threads"] = threads
    if warm_start:
        kw["warmStart"] = True
    for chave, val in (opcoes or {}).items():
        kw[chave] = val
    solver = pulp.HiGHS(**kw)
    res = prob.solve(solver)
    if PULP_4:
        st = int(res.status)
        nomes = {s.value: s.name for s in pulp.LpSolveStatus}
        out = dict(status=nomes.get(st, str(st)), ok=bool(res.has_solution),
                   objetivo=float(res.objective) if res.objective is not None else float("nan"),
                   gap=float(res.gap_rel) if getattr(res, "gap_rel", None) is not None else 0.0,
                   tempo=float(res.time), nos=int(res.nodes or 0) if getattr(res, "nodes", None) is not None else 0)
        return out
    # API clássica (2.8–3.x): status do PuLP e, quando o modelo do HiGHS está
    # acessível, o status real (a 3.x reporta limite de tempo como "Optimal")
    st = prob.status
    sol = getattr(prob, "sol_status", None)
    viavel = (getattr(pulp, "LpSolutionOptimal", 1), getattr(pulp, "LpSolutionIntegerFeasible", 2))
    obj = pulp.value(prob.objective) if prob.objective is not None else None
    out = dict(status=pulp.LpStatus.get(st, str(st)),
               ok=bool(sol in viavel) if sol is not None else bool(st == 1 and obj is not None),
               objetivo=float(obj) if obj is not None else float("nan"), gap=0.0,
               tempo=float(getattr(prob, "solutionTime", float("nan")) or float("nan")), nos=0)
    h = getattr(prob, "solverModel", None)
    if h is not None and hasattr(h, "getModelStatus"):
        try:
            nome = getattr(h.getModelStatus(), "name", "")
            if nome:
                out["status"] = nome[1:] if nome.startswith("k") else nome
            info = h.getInfo()
            if mip:
                out["gap"] = float(info.mip_gap) if info.mip_gap < 1e30 else float("nan")
                out["nos"] = int(info.mip_node_count)
            out["ok"] = out["ok"] or int(info.primal_solution_status) == 2
        except Exception:
            pass
    return out


def n_restricoes(prob: pulp.LpProblem) -> int:
    c = prob.constraints          # 4.x: método; 3.3: mapeamento chamável; 2.x: dicionário
    return len(c()) if callable(c) else len(c)


def restricao(prob: pulp.LpProblem, nome: str):
    if _GET_CONS:
        return prob.get_constraint_by_name(nome)
    return prob.constraints.get(nome)


def valor(v) -> float:
    if v is None:
        return 0.0
    x = v.varValue if hasattr(v, "varValue") else v
    return float(x) if x is not None else 0.0


def contar(prob: pulp.LpProblem) -> tuple[int, int]:
    """(nº de variáveis, nº de inteiras)."""
    vs = prob.variables()
    nint = sum(1 for v in vs if getattr(v, "cat", None) in ("Integer", pulp.LpInteger, "Binary"))
    return len(vs), nint


__all__ = ["nova_var", "expr", "resolver", "restricao", "valor", "contar", "n_restricoes", "PULP_4"]
