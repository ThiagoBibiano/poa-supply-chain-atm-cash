"""Métricas e indicadores (proposta, seção 8.4; painel, seção 6).

Quatro camadas — previsão, custo, serviço/operação e capital — mais os
indicadores próprios da programação estocástica (VSS, EVPI) e os de
replanejamento (mudanças de decisão, nervosismo do plano, especiais reativas,
divergência do oráculo). Tudo é calculado só na janela de teste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .previsao import ConjuntoPrevisoes

# ---------------------------------------------------------------------------
# Previsão
# ---------------------------------------------------------------------------


def _pinball(y: np.ndarray, q: np.ndarray, niveis: np.ndarray) -> np.ndarray:
    d = y[..., None] - q
    return np.maximum(niveis * d, (niveis - 1) * d)


def metricas_previsao(conj: ConjuntoPrevisoes, contexto: pd.DataFrame, origens: list[int], limite_k: int,
                      series: list[str] | None = None, feriados: np.ndarray | None = None) -> dict:
    """WQL, MASE, cobertura por nível e por cauda, PIT, viés e erro por antecedência.

    ``contexto`` tem NaN nos dias indisponíveis: esses dias não entram (demanda censurada)."""
    from .cenarios import pit

    niv = np.asarray(conj.niveis)
    series = series or [s for s in conj.series if ":" not in s]
    pos = [conj.series.index(s) for s in series]
    vals = contexto[conj.series].values
    H = conj.H
    linhas = []
    for k in origens:
        if k not in conj.por_origem:
            continue
        q = conj.quantis(k)[pos]
        for h in range(1, H + 1):
            t = k + h
            if t > limite_k:
                break
            y = vals[t, pos]
            for i, s in enumerate(series):
                if np.isnan(y[i]):
                    continue
                linhas.append((k, h, s, t, y[i], *q[i, h - 1]))
    if not linhas:
        return {"n": 0}
    rot = [f"q{int(round(t * 100)):02d}" for t in niv]
    df = pd.DataFrame(linhas, columns=["origem", "h", "serie", "alvo", "y", *rot])
    Q = df[rot].values
    y = df["y"].values
    pin = _pinball(y, Q, niv)
    i50 = int(np.argmin(np.abs(niv - 0.5)))
    # escala do MASE: erro do SeasonalNaive(7) no histórico até a 1ª origem
    k0 = min(origens)
    escala = {}
    for s in series:
        hist = vals[: k0 + 1, conj.series.index(s)]
        d = np.abs(hist[7:] - hist[:-7])
        escala[s] = float(np.nanmean(d)) if np.isfinite(np.nanmean(d)) and np.nanmean(d) > 0 else 1.0
    df["ae"] = np.abs(y - Q[:, i50])
    df["ae_esc"] = df["ae"] / df["serie"].map(escala)
    df["pin_soma"] = pin.mean(axis=1)
    u = pit(Q, niv, y)
    df["pit"] = u
    cobertura = {rot[i]: float((y <= Q[:, i]).mean()) for i in range(len(niv))}
    por_h = df.groupby("h").apply(lambda g: pd.Series({
        "WQL": 2 * g["pin_soma"].sum() / np.abs(g["y"]).sum(),
        "MASE": g["ae_esc"].mean(),
        "cobertura_80": float(((g["y"] >= g["q10"]) & (g["y"] <= g["q90"])).mean()),
        "cobertura_90": float(((g["y"] >= g["q05"]) & (g["y"] <= g["q95"])).mean()),
        "vies_pct": float((g["y"] - g["q50"]).sum() / g["q50"].sum()),
    }), include_groups=False).reset_index()
    caudas = pd.DataFrame([
        {"cauda": "inferior", "nivel": t, "nominal": t, "empirica": cobertura[f"q{int(round(t * 100)):02d}"]}
        for t in (0.01, 0.05, 0.10)] + [
        {"cauda": "superior", "nivel": t, "nominal": t, "empirica": cobertura[f"q{int(round(t * 100)):02d}"]}
        for t in (0.90, 0.95, 0.99)])
    hist_pit, _ = np.histogram(u[~np.isnan(u)], bins=10, range=(0, 1))
    res = {
        "n": int(len(df)),
        "WQL": float(2 * df["pin_soma"].sum() / np.abs(df["y"]).sum()),
        "MASE": float(df["ae_esc"].mean()),
        "vies_pct": float((df["y"] - df["q50"]).sum() / df["q50"].sum()),
        "cobertura": cobertura,
        "cobertura_80": float(((df["y"] >= df["q10"]) & (df["y"] <= df["q90"])).mean()),
        "cobertura_90": float(((df["y"] >= df["q05"]) & (df["y"] <= df["q95"])).mean()),
        "caudas": caudas,
        "por_h": por_h,
        "pit_hist": (hist_pit / max(hist_pit.sum(), 1)).tolist(),
        "por_serie": df.groupby("serie").apply(lambda g: pd.Series({
            "WQL": 2 * g["pin_soma"].sum() / np.abs(g["y"]).sum(), "MASE": g["ae_esc"].mean()}),
            include_groups=False).reset_index(),
    }
    if feriados is not None:
        fer = feriados[df["alvo"].values.astype(int)]
        if fer.any():
            g = df[fer]
            res["WQL_feriado"] = float(2 * g["pin_soma"].sum() / np.abs(g["y"]).sum())
            g = df[~fer]
            res["WQL_dia_comum"] = float(2 * g["pin_soma"].sum() / np.abs(g["y"]).sum())
    res["_df"] = df
    return res


# ---------------------------------------------------------------------------
# Indicadores das políticas
# ---------------------------------------------------------------------------


def _janela(df: pd.DataFrame, k_ini: int, k_fim: int, col: str = "k") -> pd.DataFrame:
    return df[(df[col] >= k_ini) & (df[col] <= k_fim)]


def kpis_politica(res, p, k_ini: int, k_fim: int) -> dict:
    """Indicadores da faixa macro e da comparação de políticas (painel, seções 2 e 6)."""
    rede, cal = p.rede, p.cal
    est = _janela(res.estados, k_ini, k_fim)
    cus = _janela(res.custos, k_ini, k_fim)
    comp = cus.groupby("componente")["valor"].sum().to_dict()
    for c_ in ["conv", "esp", "go", "opor", "rupt", "estouro", "num"]:
        comp.setdefault(c_, 0.0)
    total = float(sum(comp.values()))
    atm = est[est["ponto"].isin(rede.atms)]
    tes = est[est["ponto"].isin(rede.tesourarias)]
    cd = est[est["ponto"] == "CD"]
    dem = float(atm["saque"].sum())
    na = float(atm["nao_atendido"].sum())
    # cobertura: saldo ÷ média do P50 dos 3 dias seguintes (previsão da própria noite)
    parado = 0
    for _, r in atm.iterrows():
        cen = p.noites.get(int(r["k"]))
        if cen is None:
            continue
        i = p.idx.i(r["ponto"])
        m3 = float(np.mean(cen.mediana[i, :3]))
        if m3 > 0 and r["saldo_fim"] / m3 > 6:
            parado += 1
    abaixo, cd_abaixo = 0, 0
    for _, r in tes.iterrows():
        em = p.em.get(int(r["k"]))
        if em and r["saldo_fim"] < em["agencias"][r["ponto"]][0].em_formula - 1e-6:
            abaixo += 1
    for _, r in cd.iterrows():
        em = p.em.get(int(r["k"]))
        if em and r["saldo_fim"] < em["central"][0].em_formula - 1e-6:
            cd_abaixo += 1
    go_dias = tes[tes["k"].map(lambda kk: bool(cal.go[int(kk)]))]
    tot_saldo = float(atm["saldo_fim"].sum() + tes["saldo_fim"].sum() + cd["saldo_fim"].sum())
    desp = float(cd["despacho"].sum()) if "despacho" in cd else 0.0
    ret = float(cd["recolhimento_processado"].sum()) if "recolhimento_processado" in cd else 0.0
    n_dias = k_fim - k_ini + 1
    return {
        "comp": {k: float(v) for k, v in comp.items()},
        "total": total,
        "fill": 1.0 - na / dem if dem > 0 else 1.0,
        "nao_atendido": na,
        "rupt": int((atm["nao_atendido"] > 0.5).sum()),
        "rupt_guiche": int((tes["nao_atendido"] > 0.5).sum()),
        "est": int((tes["excesso"] > 0).sum()),
        "est_valor": float(tes["excesso"].sum()),
        "dep_recusado": float(atm["excesso"].sum()),
        "parado": int(parado),
        "abaixoMin": int(abaixo),
        "cdBelow": int(cd_abaixo),
        "cdShort": int((cd["nao_atendido"] > 0.5).sum()) if len(cd) else 0,
        "esp": int(est["visita_E"].sum()),
        "conv": int(est["visita_C"].sum()),
        "goAvg": float(go_dias["minutos_go"].mean()) if len(go_dias) else 0.0,
        "goMax": float(go_dias["minutos_go"].max()) if len(go_dias) else 0.0,
        "lim": int(go_dias["fila_go"].sum()) if len(go_dias) else 0,
        "bcTot": float(cd["entrada_custodia"].sum()) if "entrada_custodia" in cd else 0.0,
        "bcN": int((cd["entrada_custodia"] > 0).sum()) if "entrada_custodia" in cd else 0,
        "devTot": float(cd["devolucao"].sum()) if "devolucao" in cd else 0.0,
        "retTot": ret,
        "despTot": desp,
        "recirc": ret / desp if desp > 0 else 0.0,
        "shZ": float(cd["saldo_fim"].sum()) / tot_saldo if tot_saldo else 0.0,
        "shT": float(tes["saldo_fim"].sum()) / tot_saldo if tot_saldo else 0.0,
        "shI": float(atm["saldo_fim"].sum()) / tot_saldo if tot_saldo else 0.0,
        "zAvg": float(cd["saldo_fim"].mean()) if len(cd) else 0.0,
        "saldo_medio_rede": tot_saldo / n_dias,
        "dias": n_dias,
    }


def comparar_politicas(kpis: dict) -> dict:
    """Custo do erro (vs oráculo P4), separado nos dois lados, e VSS/EVPI."""
    out = {}
    k4 = kpis.get("P4")
    for nome, k in kpis.items():
        d = {}
        if k4 is not None:
            d["custo_erro"] = k["total"] - k4["total"]
            d["lado_falta"] = (k["comp"]["esp"] + k["comp"]["rupt"]) - (k4["comp"]["esp"] + k4["comp"]["rupt"])
            d["lado_excesso"] = (k["comp"]["opor"] + k["comp"]["estouro"]) - (k4["comp"]["opor"] + k4["comp"]["estouro"])
        out[nome] = d
    vss = evpi = None
    if "P1" in kpis and "P3" in kpis:
        vss = kpis["P1"]["total"] - kpis["P3"]["total"]
    if "P3" in kpis and "P4" in kpis:
        evpi = kpis["P3"]["total"] - kpis["P4"]["total"]
    return {"por_politica": out, "VSS": vss, "EVPI": evpi}


# ---------------------------------------------------------------------------
# Mudanças de decisão (plano anterior × executado)
# ---------------------------------------------------------------------------


def _fmt_pct(v: float) -> str:
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v) * 100:.0f}%"


def _lbl(cal, k: int) -> str:
    d = cal.data(k)
    dias = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
    return f"{dias[d.dayofweek]} {d.day:02d}/{d.month:02d}"


def mudancas_decisao(res, p, k_ini: int, k_fim: int, ant_transp: int = 3, ant_go: int = 2,
                     lim_valor: float = 0.25, lim_erro: float = 0.15, lim_revisao: float = 0.08) -> pd.DataFrame:
    """Para um mesmo ponto e dia-alvo, compara o plano feito antes (D−3 para a
    transportadora, D−2 para o GO) com o executado. Classes e motivos como no
    painel (seção 6)."""
    rede, cal = p.rede, p.cal
    conj = p.previsoes[p.cfg.previsao.modelo]
    dec = res.decisoes
    est = res.estados.set_index(["ponto", "k"])
    plano = dec[dec["estagio"] == 2]

    def p50(serie, origem, alvo):
        if origem not in conj.por_origem or not p.idx.tem(serie):
            return np.nan
        h = alvo - origem
        if h < 1 or h > conj.H:
            return np.nan
        return float(conj.mediana(origem)[p.idx.i(serie), h - 1])

    def erro(n, x):
        series = rede.internos[n] if n in rede.internos else [n]
        r = sum(float(p.real[s].values[x]) for s in series)
        f = sum(p50(s, x - 1, x) for s in series)
        return (r - f) / f if f and np.isfinite(f) and f > 0 else 0.0

    def revisao(n, d, o_ant, o_novo):
        series = rede.internos[n] if n in rede.internos else [n]
        a = sum(p50(s, o_ant, y) for s in series for y in (d, d + 1))
        b = sum(p50(s, o_novo, y) for s in series for y in (d, d + 1))
        return b / a - 1 if a and np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0

    def motivo(n, d, dias_erro, o_ant, fila=False):
        if fila:
            return f"GO no limite de {rede.Theta:.0f} min (R5)"
        melhor = None
        for x in dias_erro:
            e = erro(n, x)
            if melhor is None or abs(e) > abs(melhor[1]):
                melhor = (x, e)
        if melhor and abs(melhor[1]) >= lim_erro:
            return f"saque de {_lbl(cal, melhor[0])} {'acima' if melhor[1] > 0 else 'abaixo'} do P50 ({_fmt_pct(melhor[1])})"
        rv = revisao(n, d, o_ant, d - 1)
        if abs(rv) >= lim_revisao:
            return f"P50 de {_lbl(cal, d)}–{_lbl(cal, d + 1)} revisado {_fmt_pct(rv)}"
        return None

    linhas = []
    for n in rede.pontos_transp:
        for d in range(k_ini, k_fim + 1):
            if not cal.transp[d]:
                continue
            pl = plano[(plano["ponto"] == n) & (plano["k_decisao"] == d - ant_transp) & (plano["k_execucao"] == d)]
            pv = pl[pl["tipo"].isin(["C", "E"]) & (pl["prob"] >= 0.5)]
            planejado = float(pv["valor"].sum()) if len(pv) else 0.0
            tem_plano = len(pv) > 0
            try:
                e = est.loc[(n, d)]
            except KeyError:
                continue
            vC, vE = bool(e["visita_C"]), bool(e["visita_E"])
            chegou, rec = float(e["chegada"]), float(e["recolhimento"])
            so_rec = vC and chegou <= 1e-6 and rec > 0 and not vE
            cls = None
            if so_rec and not tem_plano:
                cls = "Recolhimento incluído"
            elif tem_plano and not vC and not vE:
                cls = "Cancelada"
            elif tem_plano and vE and not vC:
                cls = "Virou especial"
            elif tem_plano and vE and vC:
                cls = "Especial adicional"
            elif not tem_plano and vE:
                cls = "Especial incluída"
            elif not tem_plano and vC and chegou > 1e-6:
                cls = "Incluída"
            elif tem_plano and vC and planejado > 1 and abs(chegou - planejado) / planejado > lim_valor:
                cls = "Valor ajustado " + _fmt_pct(chegou / planejado - 1)
            if not cls:
                continue
            if so_rec:
                mot = "saldo projetado acima do teto (excesso)"
            else:
                mot = motivo(n, d, [d - 2, d - 1] if vE else [d - 2], d - ant_transp) or (
                    "saldo observado acima do projetado no plano" if chegou < planejado else
                    "saldo observado abaixo do projetado no plano")
            descr_plano = "sem visita" if not tem_plano else (
                ("Especial " if "E" in set(pv["tipo"]) and "C" not in set(pv["tipo"]) else "Convencional ") + f"R$ {planejado:,.0f} mil")
            if vE and vC and chegou > 0:
                fin = f"Especial + conv. R$ {chegou:,.0f} mil"
            elif vE:
                fin = f"Especial R$ {chegou:,.0f} mil"
            elif so_rec:
                fin = f"Recolhimento R$ {rec:,.0f} mil"
            elif vC:
                fin = f"Convencional R$ {chegou:,.0f} mil"
            else:
                fin = "sem visita"
            linhas.append(dict(k=d, data=cal.data(d), ponto=n, plano=descr_plano.replace(",", "."),
                               executado=fin.replace(",", "."), classe=cls, motivo=mot,
                               lado="falta" if (vE or chegou > planejado) else "excesso",
                               planejado=planejado, executado_valor=chegou))
    for j in rede.atms_internos:
        for d in range(k_ini, k_fim + 1):
            if not cal.go[d]:
                continue
            pl = plano[(plano["ponto"] == j) & (plano["k_decisao"] == d - ant_go) & (plano["k_execucao"] == d)
                       & (plano["tipo"] == "GO") & (plano["prob"] >= 0.5)]
            planejado = float(pl["valor"].sum()) if len(pl) else 0.0
            try:
                e = est.loc[(j, d)]
            except KeyError:
                continue
            fin = float(e["abastecimento_go"])
            fila = bool(e["fila_go"])
            cls = None
            if planejado > 1 and fin <= 1e-6:
                cls = "Adiada (fila do GO)" if fila else "Cancelada"
            elif planejado <= 1 and fin > 1e-6:
                cls = "Incluída"
            elif planejado > 1 and fin > 1e-6 and abs(fin - planejado) / planejado > lim_valor:
                cls = "Valor ajustado " + _fmt_pct(fin / planejado - 1)
            if not cls:
                continue
            mot = motivo(j, d, [d - 1], d - ant_go, fila and fin <= 1e-6) or (
                "saldo observado acima do projetado no plano" if fin < planejado else
                "saldo observado abaixo do projetado no plano")
            linhas.append(dict(k=d, data=cal.data(d), ponto=j,
                               plano=(f"GO abastece R$ {planejado:,.0f} mil" if planejado > 1 else "sem abastecimento").replace(",", "."),
                               executado=(f"GO abastece R$ {fin:,.0f} mil" if fin > 0 else ("na fila do GO" if fila else "sem abastecimento")).replace(",", "."),
                               classe=cls, motivo=mot, lado="falta" if fin > planejado else "excesso",
                               planejado=planejado, executado_valor=fin))
    df = pd.DataFrame(linhas)
    if len(df):
        df = df.sort_values(["k", "ponto"]).reset_index(drop=True)
    return df


def nervosismo(res, p, k_ini: int, k_fim: int) -> dict:
    """Fração das decisões planejadas (D−2 e D−1) que mudaram ao chegar a hora."""
    dec = res.decisoes
    plano = dec[(dec["estagio"] == 2) & dec["tipo"].isin(["C", "E", "GO"]) & (dec["prob"] >= 0.5)]
    exe = dec[(dec["estagio"] == 1) & dec["tipo"].isin(["C", "E", "GO"])]
    exe_set = {(r.ponto, int(r.k_execucao)) for r in exe.itertuples()}
    tot = mud = 0
    for r in plano.itertuples():
        ke = int(r.k_execucao)
        if not (k_ini <= ke <= k_fim):
            continue
        tot += 1
        if (r.ponto, ke) not in exe_set:
            mud += 1
    return {"planejadas": tot, "canceladas": mud, "taxa": mud / tot if tot else 0.0}


def especiais_reativas(res, p, k_ini: int, k_fim: int, nivel: float = 0.90) -> int:
    """Especiais executadas em d cujo pedido (noite d−1) seguiu um dia d−1 com
    saque acima do P90 previsto na véspera."""
    conj = p.previsoes[p.cfg.previsao.modelo]
    niv = np.asarray(conj.niveis)
    iq = int(np.argmin(np.abs(niv - nivel)))
    est = _janela(res.estados, k_ini, k_fim)
    n = 0
    for r in est[est["visita_E"]].itertuples():
        d = int(r.k)
        series = p.rede.internos[r.ponto] if r.ponto in p.rede.internos else [r.ponto]
        x = d - 1
        if (x - 1) not in conj.por_origem:
            continue
        q = conj.quantis(x - 1)
        real = sum(float(p.real[s].values[x]) for s in series)
        lim = sum(float(q[p.idx.i(s), 0, iq]) for s in series)
        if real > lim:
            n += 1
    return n


def divergencia_oraculo(res, res_oraculo, p, k_ini: int, k_fim: int) -> float:
    """Fração dos pares (ponto, dia) em que a presença de visita/abastecimento
    difere da decisão do oráculo (P4)."""
    def mapa(r):
        e = _janela(r.estados, k_ini, k_fim)
        return {(x.ponto, int(x.k)): (bool(x.visita_C) or bool(x.visita_E) or x.abastecimento_go > 1e-6)
                for x in e.itertuples() if x.ponto != "CD"}
    a, b = mapa(res), mapa(res_oraculo)
    chaves = set(a) & set(b)
    if not chaves:
        return 0.0
    return float(np.mean([a[c] != b[c] for c in chaves]))


__all__ = ["metricas_previsao", "kpis_politica", "comparar_politicas", "mudancas_decisao", "nervosismo",
           "especiais_reativas", "divergencia_oraculo"]
