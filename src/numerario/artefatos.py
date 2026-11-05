"""Artefatos: contrato de dados do painel (seção 7) e o HTML final.

A simulação grava dez tabelas — pontos, estoque_minimo, custodia, previsoes,
realizado, decisoes, estados, custos, modelo, execucoes — em Parquet (para
análises futuras) e um JSON compacto embutido num único arquivo HTML, que só
lê resultados (não roda modelo).

Unidades: valores em R$ mil; custos em R$.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from .config import z_alpha
from .metricas import (comparar_politicas, divergencia_oraculo, especiais_reativas, kpis_politica,
                       metricas_previsao, mudancas_decisao, nervosismo)
from .politicas import NOMES_POLITICAS
from .previsao import rotulos_quantis

DIAS_SEMANA = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


def _r(v, n=2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not np.isfinite(f):
        return None
    return round(f, n)


# ---------------------------------------------------------------------------
# Tabelas do contrato
# ---------------------------------------------------------------------------
def tabelas_contrato(p, resultados: dict, k_ini: int, k_fim: int) -> dict[str, pd.DataFrame]:
    from .pipeline import tabela_estoque_minimo

    cal, rede, cfg = p.cal, p.rede, p.cfg
    t = {}
    t["pontos"] = rede.tabela_pontos(cfg)
    em = tabela_estoque_minimo(p)
    t["estoque_minimo"] = em
    # custódia: remessas executadas (chegadas e devoluções na central)
    linhas = []
    for nome, r in resultados.items():
        cd = r.estados[r.estados["ponto"] == "CD"]
        for x in cd.itertuples():
            if x.entrada_custodia > 0:
                linhas.append(dict(politica=nome, central="CD", data=cal.data(int(x.k)), tipo="retirada",
                                   valor=x.entrada_custodia, tarifa=cfg.custos.c_BC * x.entrada_custodia,
                                   fixo=cfg.custos.K_BC))
            if x.devolucao > 0:
                linhas.append(dict(politica=nome, central="CD", data=cal.data(int(x.k)), tipo="devolucao",
                                   valor=x.devolucao, tarifa=cfg.custos.c_rBC * x.devolucao, fixo=cfg.custos.K_BC))
    t["custodia"] = pd.DataFrame(linhas)
    conj = p.previsoes[cfg.previsao.modelo]
    prev = conj.tabela(cal)
    t["previsoes"] = prev[prev["origem"] >= cal.data(p.k_cal_ini - 1)].reset_index(drop=True)
    real = []
    for s in p.idx.series:
        for k in range(p.k_cal_ini, k_fim + 1):
            real.append(dict(ponto=s, data=cal.data(k), saques=float(p.real[s].values[k]),
                             indisponivel=bool(np.isnan(p.contexto[s].values[k]))))
    t["realizado"] = pd.DataFrame(real)
    for chave in ["decisoes", "estados", "custos", "modelo", "execucoes"]:
        t[chave] = pd.concat([getattr(r, chave) for r in resultados.values()], ignore_index=True)
    return t


def gravar_tabelas(tabelas: dict, pasta: str) -> dict:
    os.makedirs(pasta, exist_ok=True)
    out = {}
    for nome, df in tabelas.items():
        arq = os.path.join(pasta, f"{nome}.parquet")
        d = df.copy()
        for c in d.columns:
            if d[c].dtype == object:
                d[c] = d[c].astype(str)
        d.to_parquet(arq, index=False)
        out[nome] = arq
    return out


# ---------------------------------------------------------------------------
# Indicadores consolidados
# ---------------------------------------------------------------------------
def consolidar(p, resultados: dict, k_ini: int, k_fim: int) -> dict:
    cfg = p.cfg
    kp = {n: kpis_politica(r, p, k_ini, k_fim) for n, r in resultados.items()}
    mud = {n: mudancas_decisao(r, p, k_ini, k_fim) for n, r in resultados.items()}
    for n in kp:
        kp[n]["chg"] = int(len(mud[n]))
        kp[n]["nervosismo"] = nervosismo(resultados[n], p, k_ini, k_fim)
        kp[n]["especiais_reativas"] = especiais_reativas(resultados[n], p, k_ini, k_fim)
        if "P4" in resultados:
            kp[n]["divergencia_oraculo"] = divergencia_oraculo(resultados[n], resultados["P4"], p, k_ini, k_fim)
        ex = resultados[n].execucoes
        if len(ex) and ex["status"].ne("regra").any():
            exm = ex[ex["status"] != "regra"]
            kp[n]["solver"] = dict(segundos_medio=float(exm["segundos"].mean()),
                                   segundos_max=float(exm["segundos"].max()),
                                   gap_medio=float(pd.to_numeric(exm["gap"], errors="coerce").mean()),
                                   otimos=int((exm["status"] == "Optimal").sum()), noites=int(len(exm)),
                                   fallback=int(exm["fallback"].sum()))
    comp = comparar_politicas(kp)
    conj = p.previsoes[cfg.previsao.modelo]
    origens_teste = list(range(p.k_teste_ini - 1, p.k_teste_fim))
    fer = p.cal.feriado
    mp = {"teste": metricas_previsao(conj, p.contexto, origens_teste, p.k_teste_fim, feriados=fer),
          "calibracao": metricas_previsao(conj, p.contexto, p.origens_calibracao, p.k_teste_ini - 1, feriados=fer)}
    for outro, cj in p.previsoes.items():
        if outro != cfg.previsao.modelo:
            mp[f"teste_{outro}"] = metricas_previsao(cj, p.contexto, origens_teste, p.k_teste_fim, feriados=fer)
    return {"kpis": kp, "comparacao": comp, "mudancas": mud, "previsao": mp}


# ---------------------------------------------------------------------------
# Conferências (critérios de aceite do painel, seção 9)
# ---------------------------------------------------------------------------
def conferencias(p, resultados: dict, cons: dict, k_ini: int, k_fim: int) -> list[dict]:
    out = []
    rede = p.rede

    def add(criterio, ok, detalhe):
        out.append(dict(criterio=criterio, ok=bool(ok), detalhe=detalhe))

    # 1. balanços de ATMs e tesourarias fecham com a tabela estados
    err = 0.0
    for r in resultados.values():
        e = r.estados
        a = e[e["ponto"].isin(rede.atms)]
        d = a["saldo_ini"] + a["deposito"] - a["saque"] + a["nao_atendido"] - a["excesso"] - a["saldo_fim"]
        err = max(err, float(d.abs().max()) if len(d) else 0.0)
        t = e[e["ponto"].isin(rede.tesourarias)]
        d = t["saldo_ini"] - t["abastecimento_go"] + t["guiche"] + t["nao_atendido"] - t["saldo_fim"]
        err = max(err, float(d.abs().max()) if len(d) else 0.0)
    add("Saldos de ATMs e tesourarias fecham com a tabela estados", err < 1e-6, f"maior diferença {err:.2e} R$ mil")
    # 2. previsões nas três antecedências
    conj = p.previsoes[p.cfg.previsao.modelo]
    falt = [k - h for k in range(k_ini, k_fim + 1) for h in (1, 2, 3) if (k - h) not in conj.por_origem]
    add("Previsões disponíveis nas três antecedências para todos os dias", not falt, f"{len(falt)} origens faltando")
    # 3. log de mudanças com classe e motivo
    vazios = sum(int(((m["classe"] == "") | (m["motivo"] == "")).sum()) for m in cons["mudancas"].values() if len(m))
    add("Toda mudança de decisão no log tem classe e motivo", vazios == 0, f"{vazios} linhas sem classe/motivo")
    # 4. soma dos componentes = tabela custos
    dif = 0.0
    for n, r in resultados.items():
        c = r.custos
        tot = float(c[(c["k"] >= k_ini) & (c["k"] <= k_fim)]["valor"].sum())
        dif = max(dif, abs(tot - cons["kpis"][n]["total"]))
    add("Soma dos componentes da faixa macro = tabela custos", dif < 1e-6, f"maior diferença R$ {dif:.2e}")
    # 5. EM (aproximação da seção 5) recalculado de forma independente a partir de previsoes
    ok5, det5 = _conferir_em(p, k_ini)
    add("EM recalculado a partir de previsoes (fórmula 5) bate com estoque_minimo", ok5, det5)
    # 6. custo do erro de P4 = 0; VSS e EVPI conferem
    comp = cons["comparacao"]
    ok6 = True
    det = []
    if "P4" in comp["por_politica"] and comp["por_politica"]["P4"]:
        ok6 &= abs(comp["por_politica"]["P4"]["custo_erro"]) < 1e-9
        det.append("custo do erro de P4 = 0")
    if comp["VSS"] is not None:
        ok6 &= abs(comp["VSS"] - (cons["kpis"]["P1"]["total"] - cons["kpis"]["P3"]["total"])) < 1e-6
        det.append(f"VSS = R$ {comp['VSS']:,.0f}")
    if comp["EVPI"] is not None:
        ok6 &= abs(comp["EVPI"] - (cons["kpis"]["P3"]["total"] - cons["kpis"]["P4"]["total"])) < 1e-6
        det.append(f"EVPI = R$ {comp['EVPI']:,.0f}")
    add("Custo do erro de P4 é zero; VSS e EVPI conferem", ok6, "; ".join(det) or "P1/P3/P4 não rodaram")
    # 7. rupturas e estouros nos dois lados (estados × indicadores)
    ok7 = True
    for n, r in resultados.items():
        e = r.estados[(r.estados["k"] >= k_ini) & (r.estados["k"] <= k_fim)]
        ok7 &= int((e[e["ponto"].isin(rede.atms)]["nao_atendido"] > 0.5).sum()) == cons["kpis"][n]["rupt"]
        ok7 &= int((e[e["ponto"].isin(rede.tesourarias)]["excesso"] > 0).sum()) == cons["kpis"][n]["est"]
    add("Rupturas e estouros contados igual no mapa (estados) e nos indicadores", ok7, "falta e excesso conferidos")
    # 8. saldo da central fecha
    err = 0.0
    for r in resultados.values():
        cd = r.estados[r.estados["ponto"] == "CD"]
        d = (cd["saldo_anterior"] + cd["entrada_custodia"] + cd["recolhimento_processado"] - cd["despacho"]
             - cd["devolucao"] - cd["saldo_fim"])
        err = max(err, float(d.abs().max()) if len(d) else 0.0)
    add("Saldo da central fecha: anterior + custódia + recolhimentos − despachos − devoluções", err < 1e-6,
        f"maior diferença {err:.2e} R$ mil")
    # 9. ganho de centralização: > 0 com ρ < 1 e = 0 com ρ = 1
    ok9, det9 = _conferir_ganho(p, k_ini)
    add("Ganho de centralização positivo com ρ < 1 e zero com ρ = 1", ok9, det9)
    return out


def _sigma_tab(p, serie, origem, alvo):
    conj = p.previsoes[p.cfg.previsao.modelo]
    niv = np.asarray(conj.niveis)
    i05, i95 = int(np.argmin(np.abs(niv - 0.05))), int(np.argmin(np.abs(niv - 0.95)))
    h = alvo - origem
    while h > conj.H:
        h -= 7
    q = conj.quantis(origem)[p.idx.i(serie), h - 1]
    return (q[i95] - q[i05]) / (2 * 1.6448536269514722)


def _conferir_em(p, k: int) -> tuple[bool, str]:
    """Recalcula EM ≈ z√((1−ρ)Σσ² + ρ(Σσ)²) a partir dos quantis, sem usar a
    CalculadoraEM, e compara com a tabela estoque_minimo."""
    em = p.em.get(k)
    if em is None:
        return False, "noite sem EM"
    z = z_alpha(p.cfg.es.alpha)
    rho = p.calc_em.rho
    maior = 0.0
    cal = p.cal
    for b in p.rede.tesourarias:
        P = cal.periodo_protecao_agencia(k)
        sig = []
        for x in P:
            if not cal.go[x]:
                continue
            for j in p.rede.internos[b]:
                for y in range(x, cal.prox_go(x)):
                    sig.append(_sigma_tab(p, j, k, y))
        sig = np.array(sig)
        aprox = max(p.rede.Rb[b], z * np.sqrt((1 - rho) * (sig ** 2).sum() + rho * sig.sum() ** 2)) if len(sig) else p.rede.Rb[b]
        maior = max(maior, abs(aprox - em["agencias"][b][0].em_aprox_rho))
    return maior < 1e-6, f"maior diferença {maior:.2e} R$ mil (noite {cal.data(k).date()})"


def _conferir_ganho(p, k: int) -> tuple[bool, str]:
    cal = p.cal
    z = z_alpha(p.cfg.es.alpha)
    P = cal.periodo_protecao_central(k)
    grupos = []
    for b in p.rede.tesourarias:
        sig = [_sigma_tab(p, j, k, y) for x in P if cal.go[x] for j in p.rede.internos[b] for y in range(x, cal.prox_go(x))]
        grupos.append(np.array(sig))
    for kk in p.rede.externos:
        grupos.append(np.array([_sigma_tab(p, kk, k, x) for x in P]))

    def em(sig, rho):
        return z * np.sqrt((1 - rho) * (sig ** 2).sum() + rho * sig.sum() ** 2) if len(sig) else 0.0
    todos = np.concatenate([g for g in grupos if len(g)])
    rho = p.calc_em.rho
    g_rho = sum(em(g, rho) for g in grupos) - em(todos, rho)
    g_1 = sum(em(g, 1.0) for g in grupos) - em(todos, 1.0)
    ok = (g_rho > 0 if rho < 1 else True) and abs(g_1) < 1e-6
    return ok, f"ganho com ρ = {rho:.2f}: R$ {g_rho:,.0f} mil; com ρ = 1: {g_1:.1e}"


# ---------------------------------------------------------------------------
# JSON do painel
# ---------------------------------------------------------------------------
def json_painel(p, resultados: dict, cons: dict, conf: list, k_ini: int, k_fim: int,
                sensibilidade: pd.DataFrame | None = None) -> dict:
    cfg, cal, rede = p.cfg, p.cal, p.rede
    c = cfg.custos
    dias = list(range(k_ini, k_fim + 1))
    D = len(dias)
    conj = p.previsoes[cfg.previsao.modelo]
    pols = [n for n in resultados]
    meta = {
        "titulo": "Painel da cadeia de numerário",
        "gerado_em": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "experimento": cfg.experimento.nome,
        "modelo_previsao": cfg.previsao.modelo + (f" ({cfg.previsao.chronos_id})" if cfg.previsao.modelo == "chronos2" else ""),
        "fonte_dados": p.diagnosticos.get("fonte_dados", {}).get("fonte"),
        "aviso": p.diagnosticos.get("fonte_dados", {}).get("aviso"),
        "escala": p.dados.escala,
        "config": {
            "LC": c.L_C, "LE": c.L_E, "KC": c.K_C, "KE": c.K_E, "kappa": c.kappa, "Theta": rede.Theta,
            "thetaCF": rede.theta_cf, "cGO": c.c_GO, "h": c.h, "p": c.p, "pO": c.p_o, "pEM": c.p_EM,
            "pT": c.p_T, "LBC": c.L_BC, "KBC": c.K_BC, "cBC": c.c_BC, "cRBC": c.c_rBC, "cP": c.c_P,
            "LP": c.L_P, "alfa": cfg.es.alpha, "zAlfa": z_alpha(cfg.es.alpha), "rho": p.calc_em.rho,
            "Zmax": rede.Zmax, "Rc": rede.Rc, "taxa": c.taxa_anual, "S": cfg.cenarios.n_cenarios,
            "H": cfg.previsao.horizonte, "configuracao_rede": cfg.es.configuracao_rede,
            "margem_p0": p.margem_p0,
        },
        "politicas": [{"id": n, "nome": NOMES_POLITICAS.get(n, (n, ""))[0], "desc": NOMES_POLITICAS.get(n, ("", n))[1]}
                      for n in pols],
        "dias": [{"data": str(cal.data(k).date()), "rot": f"{DIAS_SEMANA[cal.data(k).dayofweek]} {cal.data(k).day:02d}/{cal.data(k).month:02d}",
                  "transp": bool(cal.transp[k]), "go": bool(cal.go[k]), "bc": bool(cal.bc[k]), "fer": bool(cal.feriado[k])}
                 for k in dias],
        "baumol": {"W": p.calc_em.W_bar, "Q": p.calc_em.Q_baumol, "MO": p.calc_em.mo_z_menos_L},
        "VSS": cons["comparacao"]["VSS"], "EVPI": cons["comparacao"]["EVPI"],
        "periodo": [str(cal.data(k_ini).date()), str(cal.data(k_fim).date())],
    }
    pontos = []
    tp = rede.tabela_pontos(cfg)
    for r in tp.itertuples():
        d = {"id": r.ponto, "tipo": r.tipo, "nome": r.nome}
        if r.tipo == "tes":
            d.update(ag=int(r.agencia), teto=r.teto, Rb=r.reserva, filhos=rede.internos[r.ponto])
        elif r.tipo == "int":
            d.update(ag=int(r.agencia), cap=r.capacidade, theta=r.theta_min, rec=bool(r.reciclador), m=r.media)
        elif r.tipo == "ext":
            d.update(cap=r.capacidade, rec=bool(r.reciclador), m=r.media)
        elif r.tipo == "cd":
            d.update(teto=r.teto, Rc=r.reserva)
        pontos.append({k: _r(v, 3) if isinstance(v, float) else v for k, v in d.items()})
    real = {a: [_r(p.real[a].values[k], 2) for k in dias] for a in rede.atms}
    indisp = {a: [int(np.isnan(p.contexto[a].values[k])) for k in dias] for a in rede.atms}
    rot = rotulos_quantis(conj.niveis)
    prev = {}
    for a in rede.atms:
        i = p.idx.i(a)
        prev[a] = {}
        for h in (1, 2, 3):
            arr = []
            for k in dias:
                o = k - h
                if o in conj.por_origem:
                    arr.append([_r(v, 1) for v in conj.quantis(o)[i, h - 1]])
                else:
                    arr.append(None)
            prev[a][str(h)] = arr
    # estados, custos e EM usados
    estados, custos, em_usado = {}, {}, {}
    campos = {"ini": "saldo_ini", "fim": "saldo_fim", "cheg": "chegada", "rec": "recolhimento", "go": "abastecimento_go",
              "na": "nao_atendido", "exc": "excesso", "min": "minutos_go", "gui": "guiche", "ant": "saldo_anterior"}
    for n, r in resultados.items():
        e = r.estados[(r.estados["k"] >= k_ini) & (r.estados["k"] <= k_fim)].set_index(["ponto", "k"]).sort_index()
        estados[n] = {}
        for pid in rede.atms + rede.tesourarias + ["CD"]:
            sub = e.loc[pid] if pid in e.index.get_level_values(0) else None
            if sub is None:
                continue
            sub = sub.reindex(dias)
            d = {kk: [_r(v, 2) for v in sub[col].fillna(0).values] for kk, col in campos.items()}
            d["fila"] = [int(bool(v)) for v in sub["fila_go"].fillna(False).values]
            d["vC"] = [int(bool(v)) for v in sub["visita_C"].fillna(False).values]
            d["vE"] = [int(bool(v)) for v in sub["visita_E"].fillna(False).values]
            if pid == "CD":
                for kk, col in {"bc": "entrada_custodia", "dev": "devolucao", "ret": "recolhimento_processado",
                                "desp": "despacho", "short": "falta_despachar"}.items():
                    d[kk] = [_r(v, 2) for v in sub[col].fillna(0).values]
            estados[n][pid] = d
        cu = r.custos[(r.custos["k"] >= k_ini) & (r.custos["k"] <= k_fim)]
        dia = {comp: [0.0] * D for comp in ["conv", "esp", "go", "opor", "rupt", "estouro", "num"]}
        for x in cu.itertuples():
            dia[x.componente][int(x.k) - k_ini] += x.valor
        pp = cu.groupby("ponto")["valor"].sum().to_dict()
        pdia = {}
        for (pid, kk), v in cu.groupby(["ponto", "k"])["valor"].sum().items():
            pdia.setdefault(pid, [0.0] * D)[int(kk) - k_ini] = _r(v, 1)
        custos[n] = {"dia": {k2: [_r(v, 1) for v in vv] for k2, vv in dia.items()},
                     "ponto": {k2: _r(v, 1) for k2, v in pp.items()}, "ponto_dia": pdia}
        em_usado[n] = {}
        for b in rede.tesourarias + ["CD"]:
            vals = []
            for k in dias:
                em = p.em.get(k)
                if n == "P0":
                    v = (cfg.es.reserva_fixa_p0_central_dias * sum(rede.media.values()) if b == "CD" else
                         cfg.es.reserva_fixa_p0_tesouraria_dias * sum(rede.media[j] for j in rede.internos[b]))
                elif n in ("P1", "P4", "P3_sem_EM"):
                    v = rede.Rc if b == "CD" else rede.Rb[b]
                else:
                    v = (em["central"][0].em_formula if b == "CD" else em["agencias"][b][0].em_formula) if em else None
                vals.append(_r(v, 1))
            em_usado[n][b] = vals
    emf = {}
    for b in rede.tesourarias + ["CD"]:
        d = {kk: [] for kk in ["em", "pp", "med", "ind", "com", "aprox", "ndias", "pini", "pfim", "sep", "ganho", "sepA", "ganhoA"]}
        for k in dias:
            em = p.em.get(k)
            x = (em["central"][0] if b == "CD" else em["agencias"][b][0]) if em else None
            if x is None:
                for kk in d:
                    d[kk].append(None)
                continue
            d["em"].append(_r(x.em_formula, 1)); d["pp"].append(_r(x.pp, 1)); d["med"].append(_r(x.saida_mediana, 1))
            d["ind"].append(_r(x.em_independencia, 1)); d["com"].append(_r(x.em_soma_quantis, 1))
            d["aprox"].append(_r(x.em_aprox_rho, 1)); d["ndias"].append(len(x.dias_protecao))
            d["pini"].append((x.dias_protecao[0] - k_ini) if x.dias_protecao else None)
            d["pfim"].append((x.dias_protecao[-1] - k_ini) if x.dias_protecao else None)
            d["sep"].append(_r(x.total_separado, 1)); d["ganho"].append(_r(x.ganho_centralizacao, 1))
            d["sepA"].append(_r(x.total_separado_aprox, 1)); d["ganhoA"].append(_r(x.ganho_aprox, 1))
        emf[b] = d
    kpis = {n: {k2: (_r(v, 4) if isinstance(v, (int, float, np.floating)) else v) for k2, v in kk.items()}
            for n, kk in cons["kpis"].items()}
    for n in kpis:
        kpis[n]["comp"] = {k2: _r(v, 1) for k2, v in cons["kpis"][n]["comp"].items()}
    mud = {}
    for n, m in cons["mudancas"].items():
        mud[n] = [] if not len(m) else [
            {"d": int(x.k) - k_ini, "id": x.ponto, "plano": x.plano, "fin": x.executado, "cls": x.classe,
             "mot": x.motivo, "lado": x.lado} for x in m.itertuples()]
    prevm = {}
    for chave, mp in cons["previsao"].items():
        if mp.get("n", 0) == 0:
            continue
        prevm[chave] = {
            "n": mp["n"], "WQL": _r(mp["WQL"], 4), "MASE": _r(mp["MASE"], 4), "vies": _r(mp["vies_pct"], 4),
            "cob80": _r(mp["cobertura_80"], 4), "cob90": _r(mp["cobertura_90"], 4),
            "cobertura": {k2: _r(v, 4) for k2, v in mp["cobertura"].items()},
            "caudas": mp["caudas"].round(4).to_dict(orient="records"),
            "por_h": mp["por_h"].round(4).to_dict(orient="records"),
            "pit": [_r(v, 4) for v in mp["pit_hist"]],
            "fer": _r(mp.get("WQL_feriado"), 4), "comum": _r(mp.get("WQL_dia_comum"), 4),
        }
    modelo = {}
    for n, r in resultados.items():
        ex = r.execucoes[(r.execucoes["k"] >= k_ini - 1) & (r.execucoes["k"] <= k_fim - 1)]
        mo = r.modelo
        r5d, r5a = {}, {}
        for b in rede.tesourarias:
            r5d[b] = [None] * D
            r5a[b] = [None] * D
        if len(mo):
            for x in mo[(mo["restricao"] == "R5")].itertuples():
                kk = int(x.k) - k_ini
                if 0 <= kk < D:
                    if x.fonte == "dual_lp":
                        r5d[x.ponto][kk] = _r(x.dual, 3)
                    else:
                        r5a[x.ponto][kk] = int(bool(x.ativa))
        modelo[n] = {"exec": [{"d": int(x.k) + 1 - k_ini, "status": x.status, "gap": _r(x.gap, 4),
                               "seg": _r(x.segundos, 2), "vars": None if pd.isna(x.variaveis) else int(x.variaveis),
                               "bin": None if pd.isna(x.binarias) else int(x.binarias)} for x in ex.itertuples()],
                     "r5_dual": r5d, "r5_ativa": r5a}
    out = {"meta": meta, "pontos": pontos, "real": real, "indisp": indisp, "prev": prev, "niveis": rot,
           "estados": estados, "custos": custos, "em": emf, "em_usado": em_usado, "kpis": kpis,
           "comparacao": cons["comparacao"], "mudancas": mud, "previsao": prevm, "modelo": modelo,
           "conferencias": conf}
    if sensibilidade is not None and len(sensibilidade):
        out["sensibilidade"] = sensibilidade.round(4).to_dict(orient="records")
    return out


def json_estrito(o):
    """Estrutura pronta para JSON estrito — o ``JSON.parse`` do painel recusa NaN:
    NaN e ±inf viram null; tipos do numpy e do pandas viram nativos."""
    if o is None or isinstance(o, str):
        return o
    if o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(_default(k)) if isinstance(k, np.generic) else str(k)): json_estrito(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_estrito(v) for v in o]
    if isinstance(o, np.ndarray):
        return [json_estrito(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    return _default(o)


def gravar_json(dados, caminho: str, indent: int | None = None) -> str:
    """Grava JSON estrito (sem NaN), de forma atômica."""
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    with open(caminho + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(json_estrito(dados), fh, ensure_ascii=False, allow_nan=False, indent=indent)
    os.replace(caminho + ".tmp", caminho)
    return caminho


def escrever_painel(dados: dict, template: str, destino: str) -> str:
    """Injeta o JSON no template HTML (bloco <script type="application/json">)."""
    with open(template, encoding="utf-8") as fh:
        html = fh.read()
    js = json.dumps(json_estrito(dados), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    js = js.replace("</", "<\\/")
    marcador = "__DADOS_DO_NOTEBOOK__"
    if marcador not in html:
        raise ValueError("template sem o marcador __DADOS_DO_NOTEBOOK__")
    html = html.replace(marcador, js)
    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)
    with open(destino, "w", encoding="utf-8") as fh:
        fh.write(html)
    return destino


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return str(o)


# ---------------------------------------------------------------------------
# Resumo em markdown (esqueleto para gerar slides)
# ---------------------------------------------------------------------------
def _br(v: float, dec: int = 0) -> str:
    """Número no formato brasileiro (1.234,5)."""
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def _pct(v: float, dec: int = 1, sinal: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    s = _br(abs(v) * 100, dec) + "%"
    if sinal:
        s = ("+" if v > 0 else "−" if v < 0 else "") + s
    return s


def resumo_markdown(p, cons: dict, conf: list) -> str:
    """Esqueleto em markdown com os resultados, para alimentar a geração de slides."""
    cfg = p.cfg
    k = cons["kpis"]
    comp = cons["comparacao"]
    linhas = [
        "# Otimização estocástica da cadeia de numerário — resultados do backtest",
        "",
        f"- **Período de teste:** {cfg.datas.teste_inicio} a {cfg.datas.teste_fim} ({p.k_teste_fim - p.k_teste_ini + 1} decisões diárias por política)",
        f"- **Previsão:** {cfg.previsao.modelo} · H = {cfg.previsao.horizonte} dias · {len(cfg.previsao.niveis)} quantis",
        f"- **Rede:** 1 tesouraria central · {cfg.rede.n_agencias} agências × {cfg.rede.atms_internos_por_agencia} ATMs internos · {cfg.rede.n_atms_externos} externos",
        f"- **Experimento:** {cfg.experimento.nome} · cenários: {cfg.cenarios.n_cenarios} · α = {_pct(cfg.es.alpha, 0)}",
        f"- **Dados:** {p.diagnosticos.get('fonte_dados', {}).get('fonte')}"
        + (f" — {p.diagnosticos['fonte_dados']['aviso']}" if p.diagnosticos.get('fonte_dados', {}).get('aviso') else ""),
        "",
        "## 1. Comparação das políticas",
        "",
        "| Política | Custo total (R$) | Custo do erro (R$) | Nível de serviço | Rupturas | Especiais | Da custódia (R$ mil) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, x in k.items():
        ce = comp["por_politica"].get(n, {}).get("custo_erro")
        linhas.append(f"| {n} · {NOMES_POLITICAS.get(n, (n,))[0]} | {_br(x['total'])} | "
                      f"{'—' if ce is None or n == 'P4' else _br(ce)} | {_pct(x['fill'], 2)} | {x['rupt']} | {x['esp']} | {_br(x['bcTot'])} |")
    linhas += ["", "## 2. Valor da incerteza", ""]
    if comp["VSS"] is not None:
        linhas.append(f"- **VSS** = custo(P1) − custo(P3) = R$ {_br(comp['VSS'])}")
    if comp["EVPI"] is not None:
        linhas.append(f"- **EVPI** = custo(P3) − custo(P4) = R$ {_br(comp['EVPI'])}")
    mp = cons["previsao"].get("teste", {})
    if mp.get("n"):
        linhas += ["", "## 3. Previsão (janela de teste)", "",
                   f"- WQL = {_br(mp['WQL'], 3)} · MASE = {_br(mp['MASE'], 3)} · viés = {_pct(mp['vies_pct'], 1, True)}",
                   f"- Cobertura do intervalo de 80% = {_pct(mp['cobertura_80'])} · de 90% = {_pct(mp['cobertura_90'])}",
                   "- Cobertura por cauda: " + ", ".join(f"Q{int(round(r['nivel'] * 100)):02d} → {_pct(r['empirica'])}"
                                                       for r in mp["caudas"].to_dict(orient="records"))]
    linhas += ["", "## 4. Replanejamento", ""]
    for n, x in k.items():
        nv = x.get("nervosismo", {})
        linhas.append(f"- {n}: {x.get('chg', 0)} mudanças de decisão · nervosismo do plano {_pct(nv.get('taxa', 0.0))}"
                      f" · {x.get('especiais_reativas', 0)} especiais reativas")
    linhas += ["", "## 5. Conferências", ""]
    for c_ in conf:
        linhas.append(f"- {'✓' if c_['ok'] else '✗'} {c_['criterio']} — {c_['detalhe']}")
    linhas += ["", "## 6. Próximos slides sugeridos", "",
               "- Mapa da cadeia num dia com especial (painel)", "- Cascata do custo do erro: oráculo → P3 → P1",
               "- Calibração por cauda e PIT", "- Ganho de centralização ao longo do teste"]
    return "\n".join(linhas)


__all__ = ["tabelas_contrato", "gravar_tabelas", "consolidar", "conferencias", "json_painel",
           "json_estrito", "gravar_json", "escrever_painel", "resumo_markdown"]
