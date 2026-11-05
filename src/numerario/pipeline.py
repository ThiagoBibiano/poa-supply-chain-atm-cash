"""Orquestração: o notebook chama estas funções, célula a célula.

Cada etapa grava o que é caro em ``raiz/`` (Parquet/pickle) e é pulada se o
arquivo já existir — as células são idempotentes e retomáveis.

    proj = preparar(cfg)                 # dados, calendário, rede
    prever(proj)                         # Chronos-2 (e SeasonalNaive) em cache
    calibrar(proj)                       # PITs, cópula, ρ, margem da P0
    montar_noites(proj)                  # cenários + estoques mínimos por noite
    res = rodar(proj, ["P0", ..., "P4"]) # backtest rolante (checkpoint por noite)
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendario import CalendarioOperacional, conferir_feriados
from .cenarios import Dependencia, estimar_dependencia, gerar_cenarios_origem, pit
from .config import Config
from .dados import DadosProjeto, baixar_nn5, preparar_dados
from .estoque_seguranca import CalculadoraEM, IndiceSeries, em_para_tabela, nome_deposito, nome_guiche, parametros_regiao
from .politicas import ContextoNoite, criar_politica
from .previsao import criar_previsor, prever_origens
from .rede import RedePiloto
from .simulacao import ResultadoPolitica, rodar_politica


@dataclass
class Projeto:
    cfg: Config
    raiz: str
    dados: DadosProjeto
    cal: CalendarioOperacional
    rede: RedePiloto
    idx: IndiceSeries
    contexto: pd.DataFrame          # datas × séries (R$ mil, NaN nos indisponíveis): entrada da previsão
    real: pd.DataFrame              # datas × séries (imputado): demanda verdadeira da simulação
    nao_negativas: np.ndarray
    k_cal_ini: int
    k_cal_fim: int
    k_aq_ini: int
    k_teste_ini: int
    k_teste_fim: int
    previsoes: dict = field(default_factory=dict)      # modelo → ConjuntoPrevisoes
    dep: Dependencia | None = None
    calc_em: CalculadoraEM | None = None
    noites: dict = field(default_factory=dict)         # k → CenariosOrigem
    em: dict = field(default_factory=dict)             # k → saída de CalculadoraEM.horizonte
    margem_p0: float | None = None
    diagnosticos: dict = field(default_factory=dict)
    log: callable = print

    def caminho(self, *partes: str) -> str:
        p = os.path.join(self.raiz, *partes)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    @property
    def origens_calibracao(self) -> list[int]:
        return list(range(self.k_cal_ini - 1, self.k_cal_fim))

    @property
    def noites_simulacao(self) -> list[int]:
        return list(range(self.k_aq_ini - 1, self.k_teste_fim))


# ---------------------------------------------------------------------------
def preparar(cfg: Config, raiz: str | None = None, log=print) -> Projeto:
    if cfg.calendario.ciclo != "todo_dia":
        raise NotImplementedError("só o ciclo de decisão 'todo_dia' está implementado (ver docs/especificacao.md, seção 9)")
    raiz = raiz or cfg.caminhos.raiz
    os.makedirs(raiz, exist_ok=True)
    bruto, meta = baixar_nn5(cfg, os.path.join(raiz, "01_dados", "raw"))
    dados = preparar_dados(bruto, cfg, meta)
    fim_cal = pd.Timestamp(cfg.datas.teste_fim) + pd.Timedelta(days=cfg.previsao.horizonte + 30)
    cal = CalendarioOperacional.construir(cfg, bruto.index[0], fim_cal)
    rede = RedePiloto.construir(cfg, dados)

    series = list(rede.atms)
    ctx_cols, real_cols, nneg = {}, {}, []
    idx_datas = cal.datas
    for a in rede.atms:
        ctx_cols[a] = dados.saques_bruto[a].reindex(idx_datas)
        real_cols[a] = dados.saques[a].reindex(idx_datas)
        nneg.append(True)
    if dados.depositos is not None:
        for a in dados.depositos.columns:
            s = nome_deposito(a)
            series.append(s)
            ctx_cols[s] = dados.depositos[a].reindex(idx_datas)
            real_cols[s] = dados.depositos[a].reindex(idx_datas)
            nneg.append(True)
    if dados.guiche is not None:
        for b in dados.guiche.columns:
            s = nome_guiche(b)
            series.append(s)
            ctx_cols[s] = dados.guiche[b].reindex(idx_datas)
            real_cols[s] = dados.guiche[b].reindex(idx_datas)
            nneg.append(False)
    contexto = pd.DataFrame(ctx_cols)[series]
    real = pd.DataFrame(real_cols)[series]
    # além do fim do NN5: repete a última semana (só para o horizonte do oráculo)
    ult = dados.saques.index[-1]
    for i in range(1, len(idx_datas)):
        d = idx_datas[i]
        if d > ult:
            real.loc[d] = real.loc[d - pd.Timedelta(days=7)]
    p = Projeto(cfg=cfg, raiz=raiz, dados=dados, cal=cal, rede=rede, idx=IndiceSeries(series),
                contexto=contexto, real=real, nao_negativas=np.array(nneg),
                k_cal_ini=cal.k(cfg.datas.calibracao_inicio), k_cal_fim=cal.k(cfg.datas.calibracao_fim),
                k_aq_ini=cal.k(cfg.datas.aquecimento_inicio), k_teste_ini=cal.k(cfg.datas.teste_inicio),
                k_teste_fim=cal.k(cfg.datas.teste_fim), log=log)
    p.diagnosticos["fonte_dados"] = meta
    p.diagnosticos["feriados_divergencias"] = conferir_feriados()
    # grava os dados preparados (contrato: tabela "realizado" sai daqui)
    dados.saques.to_parquet(p.caminho("01_dados", "saques_rs_mil.parquet"))
    dados.indisponivel.to_parquet(p.caminho("01_dados", "indisponivel.parquet"))
    dados.mapa_series.to_parquet(p.caminho("01_dados", "mapa_series.parquet"))
    cal.tabela().to_parquet(p.caminho("01_dados", "calendario.parquet"))
    rede.tabela_pontos(cfg).to_parquet(p.caminho("01_dados", "pontos.parquet"))
    return p


# ---------------------------------------------------------------------------
def prever(p: Projeto, modelos: list[str] | None = None) -> dict:
    """Previsões para as origens de calibração e de simulação, com cache."""
    cfg = p.cfg
    modelos = modelos or [cfg.previsao.modelo]
    origens = sorted(set(p.origens_calibracao) | set(p.noites_simulacao))
    for m in modelos:
        chave = cfg.hash("previsao") + "_" + p.dados.hash()
        arq = p.caminho("02_previsoes", f"{m}__{chave}.parquet")
        aditivas = ~p.nao_negativas
        previsor = criar_previsor(cfg, p.cal, m, aditivas=aditivas)
        t0 = time.time()
        conj = prever_origens(previsor, p.contexto, p.cal, origens, cfg.previsao.horizonte, cfg.previsao.niveis,
                              caminho_cache=arq, nao_negativas=p.nao_negativas, log=p.log)
        p.previsoes[m] = conj
        p.diagnosticos.setdefault("previsao", {})[m] = {
            "arquivo": arq, "origens": len(conj.por_origem), "segundos": round(time.time() - t0, 1),
            **(getattr(previsor, "info", {}) or {})}
    return p.previsoes


# ---------------------------------------------------------------------------
def pits(p: Projeto, modelo: str, origens: list[int], limite_k: int) -> np.ndarray:
    """z = Φ⁻¹(PIT) por (origem, série, h); NaN se o real não existe ou é indisponível."""
    from scipy import stats

    conj = p.previsoes[modelo]
    H = conj.H
    ctx = p.contexto.values  # NaN nos indisponíveis
    Z = np.full((len(origens), len(conj.series), H), np.nan)
    for i, k in enumerate(origens):
        q = conj.quantis(k)
        y = np.full((len(conj.series), H), np.nan)
        for h in range(1, H + 1):
            if k + h <= limite_k:
                y[:, h - 1] = ctx[k + h]
        u = pit(q, conj.niveis, y)
        Z[i] = stats.norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
    return Z


def calibrar(p: Projeto, modelo: str | None = None) -> dict:
    """Janela de calibração: dependência da cópula, ρ, parâmetros da região,
    estoque mínimo e margem da P0. Nada da janela de teste entra aqui."""
    cfg = p.cfg
    modelo = modelo or cfg.previsao.modelo
    limite = p.k_teste_ini - 1  # nada do teste
    Z = pits(p, modelo, p.origens_calibracao, limite)
    p.dep = estimar_dependencia(Z, cfg.cenarios.shrinkage)
    W_bar, var_cd = parametros_regiao(p.dados.saques, cfg.datas.calibracao_fim, p.dados.guiche, p.dados.depositos)
    p.calc_em = CalculadoraEM(cfg, p.rede, p.cal, p.idx, p.dep.rho_medio, W_bar, var_cd)
    if cfg.experimento.calibrar_p0:
        p.margem_p0 = calibrar_margem_p0(p)
    else:
        p.margem_p0 = cfg.experimento.margem_p0_padrao
    p.diagnosticos["calibracao"] = {
        "rho_medio": p.dep.rho_medio, "n_origens": p.dep.n_origens, "shrinkage": p.dep.shrinkage,
        "W_bar_regiao": W_bar, "var_diaria_regiao": var_cd, "Q_baumol": p.calc_em.Q_baumol,
        "miller_orr_z_menos_L": p.calc_em.mo_z_menos_L, "margem_p0": p.margem_p0}
    return p.diagnosticos["calibracao"]


def _media_hist(p: Projeto, k: int, janela: int) -> dict:
    ini = max(0, k - janela + 1)
    return {s: float(np.nanmean(p.real.iloc[ini:k + 1][s].values)) for s in p.idx.series}


def calibrar_margem_p0(p: Projeto) -> float:
    """Escolhe a margem da P0 (média histórica + margem) que minimiza o custo
    simulado na janela de calibração (regra da seção 6.3: só calibração)."""
    cfg = p.cfg
    melhor, custos = None, {}
    k_ini, k_fim = p.k_cal_ini + 14, p.k_cal_fim  # 14 dias de aquecimento dentro da calibração
    D_real = {a: p.real[a].values for a in p.rede.atms}
    G_real = {a: p.real[nome_deposito(a)].values for a in p.rede.recicladores} if p.rede.recicladores else {}
    F_real = {b: p.real[nome_guiche(b)].values for b in p.rede.tesourarias if p.idx.tem(nome_guiche(b))}
    for m in cfg.experimento.margens_p0:
        pol = criar_politica("P0", cfg, p.rede, p.cal, p.idx, p.calc_em, margem_p0=m)
        pol._plano = lambda ctx, dec, noites=2: []  # sem projeção na calibração

        def ctxf(k, est):
            return ContextoNoite(k=k, estado=est, media_hist=_media_hist(p, k, cfg.experimento.janela_media_p0_dias))
        ctxf.fallback = pol
        res = rodar_politica("P0", pol, cfg, p.rede, p.cal, {"D": D_real, "G": G_real, "F": F_real}, ctxf,
                             k_ini, k_fim, caminho_ckpt=None, log=lambda *a: None)
        custos[m] = float(res.custos["valor"].sum())
        if melhor is None or custos[m] < custos[melhor]:
            melhor = m
    p.diagnosticos["calibracao_p0"] = custos
    return float(melhor)


# ---------------------------------------------------------------------------
def montar_noites(p: Projeto, modelo: str | None = None) -> None:
    """Cenários e estoques mínimos de cada noite da simulação (cache em pickle)."""
    cfg = p.cfg
    modelo = modelo or cfg.previsao.modelo
    chave = cfg.hash("previsao", "cenarios", "es", "rede", "datas") + "_" + p.dados.hash() + f"_{modelo}"
    arq = p.caminho("03_cenarios", f"noites__{chave}.pkl")
    if os.path.exists(arq):
        with open(arq, "rb") as fh:
            salvo = pickle.load(fh)
        p.noites, p.em = salvo["noites"], salvo["em"]
        p.log(f"[cenários] {len(p.noites)} noites lidas do cache")
        return
    conj = p.previsoes[modelo]
    chave_series = np.array([i for i, s in enumerate(p.idx.series) if ":" not in s])
    t0 = time.time()
    for i, k in enumerate(p.noites_simulacao):
        q = conj.quantis(k)
        cen = gerar_cenarios_origem(cfg, k, p.idx.series, q, conj.niveis, p.dep, p.nao_negativas, chave_series)
        p.noites[k] = cen
        p.em[k] = p.calc_em.horizonte(k, cen)
        if (i + 1) % 14 == 0:
            p.log(f"[cenários] {i + 1}/{len(p.noites_simulacao)} noites · {time.time() - t0:.0f} s")
    with open(arq + ".tmp", "wb") as fh:
        pickle.dump({"noites": p.noites, "em": p.em}, fh)
    os.replace(arq + ".tmp", arq)


def tabela_estoque_minimo(p: Projeto) -> pd.DataFrame:
    """Tabela ``estoque_minimo`` (contrato): EM e PP de cada tesouraria e da
    central, calculados na noite do próprio dia (τ = 0)."""
    linhas = []
    for k, em in sorted(p.em.items()):
        for b, lst in em["agencias"].items():
            linhas.append(em_para_tabela(lst[0], p.cal))
        linhas.append(em_para_tabela(em["central"][0], p.cal))
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
def _real_futuro(p: Projeto, k: int) -> np.ndarray:
    H = p.cfg.previsao.horizonte
    return p.real.values[k + 1:k + 1 + H].T.copy()


def _necessidade_go(p: Projeto):
    def f(kd: int) -> dict:
        cen = p.noites.get(kd - 1)
        if cen is None or not p.cal.go[kd]:
            return {}
        g = p.cal.prox_go(kd)
        out = {}
        for j in p.rede.atms_internos:
            i = p.idx.i(j)
            out[j] = float(sum(cen.mediana[i, y - (kd - 1) - 1] for y in range(kd, g) if y - (kd - 1) - 1 < cen.mediana.shape[1]))
        return out
    return f


def rodar(p: Projeto, politicas: list[str] | None = None, sufixo: str = "") -> dict[str, ResultadoPolitica]:
    cfg = p.cfg
    politicas = politicas or cfg.experimento.politicas
    D_real = {a: p.real[a].values for a in p.rede.atms}
    G_real = {a: p.real[nome_deposito(a)].values for a in p.rede.recicladores}
    F_real = {b: p.real[nome_guiche(b)].values for b in p.rede.tesourarias if p.idx.tem(nome_guiche(b))}
    demanda = {"D": D_real, "G": G_real, "F": F_real, "necessidade_go": _necessidade_go(p)}
    fallback = criar_politica("P0", cfg, p.rede, p.cal, p.idx, p.calc_em, margem_p0=p.margem_p0)
    resultados = {}
    chave = cfg.hash("custos", "rede", "es", "solver", "cenarios", "previsao", "datas") + "_" + p.dados.hash()
    for nome in politicas:
        pol = criar_politica(nome, cfg, p.rede, p.cal, p.idx, p.calc_em, margem_p0=p.margem_p0)
        arq_res = p.caminho("04_backtest", f"{nome}{sufixo}__{chave}.pkl")
        if os.path.exists(arq_res):
            with open(arq_res, "rb") as fh:
                resultados[nome] = pickle.load(fh)
            p.log(f"[{nome}] resultado lido do cache")
            continue

        def ctxf(k, est, _p=p):
            return ContextoNoite(k=k, estado=est, cen=_p.noites.get(k), em=_p.em.get(k),
                                 real_futuro=_real_futuro(_p, k),
                                 media_hist=_media_hist(_p, k, cfg.experimento.janela_media_p0_dias))
        ctxf.fallback = fallback
        res = rodar_politica(nome, pol, cfg, p.rede, p.cal, demanda, ctxf, p.k_aq_ini, p.k_teste_fim,
                             caminho_ckpt=p.caminho("04_backtest", f"ckpt_{nome}{sufixo}__{chave}.pkl"), log=p.log)
        with open(arq_res + ".tmp", "wb") as fh:
            pickle.dump(res, fh)
        os.replace(arq_res + ".tmp", arq_res)
        resultados[nome] = res
    return resultados


def sensibilidade(p: Projeto, grade: dict[str, list], politicas: list[str] | None = None,
                  n_cenarios: int | None = 10) -> pd.DataFrame:
    """Roda as políticas variando um parâmetro por vez (seção 8.5).

    ``grade``: {"rede.Theta_min_dia": [60, 90, 120], "custos.nivel_servico_alvo": [...]}.
    Reaproveita previsões e dependência; refaz rede, cenários e estoques mínimos
    só quando o parâmetro mexe neles. Custo total × Θ dá o valor de um minuto
    do GO (diferença finita), que o dual de R5 não dá: R5 só tem binárias."""
    import copy

    from .metricas import kpis_politica

    politicas = politicas or ["P1", "P3"]
    linhas = []
    for chave, valores in grade.items():
        for v in valores:
            cfg2 = p.cfg.copia(**{chave: v})
            if n_cenarios:
                cfg2.cenarios.n_cenarios = n_cenarios
            p2 = copy.copy(p)
            p2.cfg = cfg2
            p2.rede = RedePiloto.construir(cfg2, p.dados)
            W_bar, var_cd = parametros_regiao(p.dados.saques, cfg2.datas.calibracao_fim, p.dados.guiche, p.dados.depositos)
            p2.calc_em = CalculadoraEM(cfg2, p2.rede, p.cal, p.idx, p.dep.rho_medio, W_bar, var_cd)
            p2.noites, p2.em = {}, {}
            montar_noites(p2)
            res = rodar(p2, politicas, sufixo="_sens")
            for nome, r in res.items():
                k = kpis_politica(r, p2, p.k_teste_ini, p.k_teste_fim)
                linhas.append({"parametro": chave, "valor": v, "politica": nome, "custo_total": k["total"],
                               "nivel_servico": k["fill"], "especiais": k["esp"], "rupturas": k["rupt"],
                               "minutos_go_medio": k["goAvg"], "dias_go_fila": k["lim"],
                               "saldo_medio_rede": k["saldo_medio_rede"]})
    df = pd.DataFrame(linhas)
    if len(df) and "rede.Theta_min_dia" in grade:
        th = df[df["parametro"] == "rede.Theta_min_dia"].sort_values("valor")
        for nome, g in th.groupby("politica"):
            g = g.sort_values("valor")
            dv = g["valor"].diff()
            dc = g["custo_total"].diff()
            df.loc[g.index, "valor_minuto_go"] = (-dc / dv / max(len(p.cal.go[p.k_teste_ini:p.k_teste_fim + 1].nonzero()[0]), 1)
                                                  / max(len(p.rede.tesourarias), 1)).values
    return df


def salvar_diagnosticos(p: Projeto) -> str:
    from .artefatos import gravar_json
    return gravar_json(p.diagnosticos, p.caminho("06_execucoes", "diagnosticos.json"), indent=1)


__all__ = ["Projeto", "preparar", "prever", "calibrar", "montar_noites", "rodar", "pits", "sensibilidade",
           "tabela_estoque_minimo", "salvar_diagnosticos", "calibrar_margem_p0"]
