"""Previsão probabilística: Chronos-2 zero-shot e SeasonalNaive quantílico.

Cada previsão devolve, para cada série e cada passo h = 1..H, os 21 quantis
nativos do Chronos-2 (0,01 a 0,99). São distribuições marginais: a dependência
entre dias e entre ATMs entra depois, na cópula (seção 6 da proposta).

Detalhes do Chronos-2 (chronos-forecasting 2.x, conferidos no código-fonte):
- ``predict_quantiles`` recebe uma lista de dicionários com ``target`` e,
  opcionalmente, ``past_covariates`` e ``future_covariates``;
- NaN no alvo é mascarado pelo próprio modelo, então dias indisponíveis entram
  como NaN no contexto (as datas precisam ser contínuas);
- os 21 níveis treinados ficam em ``pipeline.quantiles``.

Janela expansiva: cada origem usa todo o histórico até o dia da decisão.
Cache em Parquet por origem: o que já foi previsto não roda de novo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendario import CalendarioOperacional
from .config import Config

def rotulos_quantis(niveis: list[float]) -> list[str]:
    return [f"q{int(round(q * 100)):02d}" for q in niveis]


# ---------------------------------------------------------------------------
# Previsores
# ---------------------------------------------------------------------------
class Previsor:
    nome = "base"

    def prever(self, contexto: np.ndarray, origem_k: int, H: int) -> np.ndarray:
        """contexto: (n_series, k+1) com NaN; devolve (n_series, H, Q)."""
        raise NotImplementedError


class SeasonalNaiveQuantilico(Previsor):
    """Mediana = valor do mesmo dia da semana na semana anterior; quantis pelos
    resíduos log (multiplicativos) do próprio método nas últimas semanas, por
    passo h. Para séries com sinal (guichê), resíduos aditivos."""

    nome = "seasonal_naive"

    def __init__(self, niveis: list[float], semanas: int = 8, aditivas: np.ndarray | None = None):
        self.niveis = np.asarray(niveis)
        self.semanas = semanas
        self.aditivas = aditivas

    @staticmethod
    def _ultimo_valido(y: np.ndarray, i: int) -> float:
        # mesmo dia da semana, recuando de 7 em 7 até achar valor
        j = i
        while j >= 0:
            if not np.isnan(y[j]):
                return float(y[j])
            j -= 7
        v = y[~np.isnan(y)]
        return float(v[-1]) if len(v) else 0.0

    def prever(self, contexto: np.ndarray, origem_k: int, H: int) -> np.ndarray:
        n, T = contexto.shape
        out = np.zeros((n, H, len(self.niveis)))
        W = self.semanas * 7
        for s in range(n):
            y = contexto[s]
            adit = bool(self.aditivas[s]) if self.aditivas is not None else False
            for h in range(1, H + 1):
                lag = 7 * int(np.ceil(h / 7))
                base = self._ultimo_valido(y, T - 1 + h - lag)
                # resíduos históricos do mesmo método e passo
                res = []
                for t in range(max(lag, T - W), T):
                    prev = y[t - lag] if t - lag >= 0 else np.nan
                    if np.isnan(y[t]) or np.isnan(prev):
                        continue
                    if adit:
                        res.append(y[t] - prev)
                    elif y[t] > 0 and prev > 0:
                        res.append(np.log(y[t] / prev))
                res = np.asarray(res) if res else np.zeros(1)
                qs = np.quantile(res, self.niveis)
                out[s, h - 1] = base + qs if adit else max(base, 1e-6) * np.exp(qs)
        return out


class Chronos2Previsor(Previsor):
    nome = "chronos2"

    def __init__(self, cfg: Config, cal: CalendarioOperacional):
        self.cfg = cfg
        self.cal = cal
        self.niveis = list(cfg.previsao.niveis)
        self.cov = cal.covariaveis() if cfg.previsao.covariaveis_calendario else None
        self._pipeline = None
        self.info: dict = {}

    def _carregar(self):
        if self._pipeline is not None:
            return self._pipeline
        from chronos import BaseChronosPipeline

        p = self.cfg.previsao
        disp = p.dispositivo
        if disp == "auto":
            import torch
            disp = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"device_map": disp}
        if p.chronos_revisao:
            kw["revision"] = p.chronos_revisao
        t0 = time.time()
        self._pipeline = BaseChronosPipeline.from_pretrained(p.chronos_id, **kw)
        treinados = list(getattr(self._pipeline, "quantiles", []))
        self.info = {"modelo": p.chronos_id, "revisao": p.chronos_revisao, "dispositivo": disp,
                     "quantis_treinados": treinados, "segundos_carga": round(time.time() - t0, 1)}
        faltam = [q for q in self.niveis if not any(abs(q - t) < 1e-9 for t in treinados)]
        if faltam:
            self.info["aviso"] = f"níveis fora dos treinados (interpolados pelo Chronos): {faltam}"
        return self._pipeline

    def prever(self, contexto: np.ndarray, origem_k: int, H: int) -> np.ndarray:
        pipe = self._carregar()
        p = self.cfg.previsao
        n, T = contexto.shape
        ini = 0 if p.contexto_max is None else max(0, T - p.contexto_max)
        entradas = []
        for s in range(n):
            d = {"target": contexto[s, ini:T].astype(np.float32)}
            if self.cov is not None:
                passado = self.cov.iloc[ini:T]
                futuro = self.cov.iloc[T:T + H]
                d["past_covariates"] = {c: passado[c].values.astype(np.float32) for c in self.cov.columns}
                d["future_covariates"] = {c: futuro[c].values.astype(np.float32) for c in self.cov.columns}
            entradas.append(d)
        quantis, _ = pipe.predict_quantiles(entradas, prediction_length=H, quantile_levels=self.niveis,
                                            batch_size=p.batch_size, cross_learning=p.cross_learning)
        out = np.stack([np.asarray(q.float().cpu().numpy() if hasattr(q, "cpu") else q)[0] for q in quantis])
        return out  # (n, H, Q)


# ---------------------------------------------------------------------------
# Armazém de previsões (cache em Parquet)
# ---------------------------------------------------------------------------
@dataclass
class ConjuntoPrevisoes:
    """Previsões de várias origens: dict origem_k → array (n_series, H, Q)."""

    modelo: str
    series: list[str]
    niveis: list[float]
    H: int
    por_origem: dict[int, np.ndarray]

    def __post_init__(self):
        # O Chronos-2 devolve float32 (e o cache em Parquet preserva o tipo). Somas de
        # centenas de σ em float32 erram na 4ª–5ª casa — foi o que derrubou a conferência
        # do ganho de centralização com ρ = 1. Tudo a jusante trabalha em float64.
        self.por_origem = {int(k): np.asarray(v, dtype=np.float64) for k, v in self.por_origem.items()}

    def quantis(self, origem_k: int) -> np.ndarray:
        return self.por_origem[origem_k]

    def mediana(self, origem_k: int) -> np.ndarray:
        i = int(np.argmin(np.abs(np.asarray(self.niveis) - 0.5)))
        return self.por_origem[origem_k][:, :, i]

    def tabela(self, cal: CalendarioOperacional) -> pd.DataFrame:
        """Formato longo do contrato de dados (tabela ``previsoes``)."""
        rot = rotulos_quantis(self.niveis)
        ks = sorted(self.por_origem)
        if not ks:
            return pd.DataFrame(columns=["modelo", "ponto", "origem", "alvo", "h", *rot])
        A = np.stack([self.por_origem[k] for k in ks])          # (O, S, H, Q)
        O, S, H, Q = A.shape
        kk = np.repeat(np.array(ks), S * H)
        ss = np.tile(np.repeat(np.arange(S), H), O)
        hh = np.tile(np.arange(1, H + 1), O * S)
        origem = cal.datas[kk]
        df = pd.DataFrame({"modelo": self.modelo, "ponto": np.array(self.series, dtype=object)[ss],
                           "origem": origem, "alvo": origem + pd.to_timedelta(hh, unit="D"), "h": hh})
        df[rot] = A.reshape(-1, Q)
        return df

    @classmethod
    def de_tabela(cls, df: pd.DataFrame, cal: CalendarioOperacional, series: list[str],
                  niveis: list[float], H: int) -> "ConjuntoPrevisoes":
        rot = rotulos_quantis(niveis)
        idx = {s: i for i, s in enumerate(series)}
        df = df[df["ponto"].isin(idx) & (df["h"] <= H)]
        ks = df["origem"].map(lambda d: cal.k(d)).values
        si = df["ponto"].map(idx).values
        hi = df["h"].values.astype(int) - 1
        Q = df[rot].values.astype(float)
        por = {}
        for k in np.unique(ks):
            m = ks == k
            arr = np.full((len(series), H, len(niveis)), np.nan)
            arr[si[m], hi[m]] = Q[m]
            por[int(k)] = arr
        modelo = str(df["modelo"].iloc[0]) if len(df) else "?"
        return cls(modelo=modelo, series=series, niveis=niveis, H=H, por_origem=por)


def pos_processar(q: np.ndarray, nao_negativas: np.ndarray | None = None) -> np.ndarray:
    """Quantis monotônicos (float64) e, nas séries de saque, sem valores negativos."""
    q = np.sort(np.asarray(q, dtype=np.float64), axis=-1)
    if nao_negativas is not None:
        q[nao_negativas] = np.maximum(q[nao_negativas], 0.0)
    return q


def prever_origens(previsor: Previsor, contexto_total: pd.DataFrame, cal: CalendarioOperacional,
                   origens: list[int], H: int, niveis: list[float], caminho_cache: str | None = None,
                   nao_negativas: np.ndarray | None = None, log=print) -> ConjuntoPrevisoes:
    """Roda o previsor em cada origem (janela expansiva), com cache idempotente.

    ``contexto_total``: datas × séries (R$ mil, NaN nos indisponíveis), alinhado
    ao calendário. Cada origem k usa as linhas 0..k (nada do futuro entra).
    """
    series = list(contexto_total.columns)
    feitas: dict[int, np.ndarray] = {}
    if caminho_cache and os.path.exists(caminho_cache):
        df = pd.read_parquet(caminho_cache)
        feitas = ConjuntoPrevisoes.de_tabela(df, cal, series, niveis, H).por_origem
    faltam = [k for k in origens if k not in feitas]
    if faltam:
        log(f"[previsão] {previsor.nome}: {len(faltam)} origens a prever ({len(feitas)} no cache)")
    vals = contexto_total.values.T  # (n, T)
    t0 = time.time()
    for i, k in enumerate(faltam):
        ctx = vals[:, : k + 1]
        q = previsor.prever(ctx, k, H)
        feitas[k] = pos_processar(q, nao_negativas)
        if caminho_cache and ((i + 1) % 25 == 0 or i == len(faltam) - 1):
            conj = ConjuntoPrevisoes(previsor.nome, series, niveis, H, dict(feitas))
            _gravar_atomico(conj.tabela(cal), caminho_cache)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(faltam)} origens · {time.time() - t0:.0f} s")
    return ConjuntoPrevisoes(previsor.nome, series, niveis, H, {k: feitas[k] for k in sorted(feitas)})


# ---------------------------------------------------------------------------
# Recalibração quantílica pelos PITs da calibração
# ---------------------------------------------------------------------------
def grupo_serie(nome: str) -> str:
    """Grupo de recalibração: saques de ATM, depósitos (G:) e guichê (F:) têm mapas próprios."""
    return nome.split(":", 1)[0] if ":" in nome else "saque"


@dataclass
class Recalibracao:
    """Mapa nível nominal τ → nível τ' do quantil bruto, por grupo e antecedência.

    Recalibração por PIT (Kuleshov, Fenner e Ermon, 2018, adaptada a quantis):
    se a previsão fosse calibrada, os PITs u = F̂(y) seriam uniformes. Com G a
    distribuição empírica dos PITs da janela de calibração, o quantil
    recalibrado de nível τ é o quantil bruto de nível τ' = G⁻¹(τ). Corrige as
    duas caudas de forma independente (uma cauda fina em cima e larga embaixo
    gera τ' > τ nos níveis altos e τ' ≈ τ nos baixos). O quantil bruto além de
    0,01/0,99 vem da extrapolação de cauda de ``cenarios.avaliar_quantil``.
    """

    niveis: np.ndarray                    # (Q,) nominais
    mapa: dict                            # grupo → (H, Q) níveis τ' do quantil bruto
    n_pits: dict                          # grupo → (H,) PITs usados por antecedência
    origens: tuple                        # (primeira, última) origem usada no ajuste

    def aplicar(self, conj: "ConjuntoPrevisoes", nao_negativas: np.ndarray | None = None) -> "ConjuntoPrevisoes":
        from .cenarios import avaliar_quantil

        grupos = np.array([grupo_serie(s) for s in conj.series])
        novo = {}
        for k, q in conj.por_origem.items():
            q2 = q.copy()
            for g, tau in self.mapa.items():
                m = grupos == g
                if not m.any():
                    continue
                qg = q[m]                                             # (Sg, H, Q)
                u = np.broadcast_to(tau.T[:, None, :], (tau.shape[1], qg.shape[0], qg.shape[1]))  # (Q, Sg, H)
                val = avaliar_quantil(qg, self.niveis, u)              # (Q, Sg, H)
                q2[m] = np.moveaxis(val, 0, -1)
            novo[k] = pos_processar(q2, nao_negativas)
        return ConjuntoPrevisoes(conj.modelo, list(conj.series), list(conj.niveis), conj.H, novo)

    def resumo(self, grupo: str = "saque", niveis=(0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)) -> dict:
        """τ' médio sobre as antecedências, para os níveis de interesse (diagnóstico)."""
        if grupo not in self.mapa:
            return {}
        tau = self.mapa[grupo].mean(axis=0)
        return {f"q{int(round(t * 100)):02d}": float(tau[int(np.argmin(np.abs(self.niveis - t)))]) for t in niveis}


def pits_previsao(conj: "ConjuntoPrevisoes", contexto: pd.DataFrame, origens: list[int], limite_k: int) -> np.ndarray:
    """u = F̂(y) por (origem, série, h); NaN se o real não existe, é indisponível ou passa de ``limite_k``."""
    from .cenarios import pit

    vals = contexto[conj.series].values
    U = np.full((len(origens), len(conj.series), conj.H), np.nan)
    for i, k in enumerate(origens):
        if k not in conj.por_origem:
            continue
        y = np.full((len(conj.series), conj.H), np.nan)
        for h in range(1, conj.H + 1):
            if k + h <= limite_k:
                y[:, h - 1] = vals[k + h]
        U[i] = pit(conj.quantis(k), conj.niveis, y)
    return U


def ajustar_recalibracao(conj: "ConjuntoPrevisoes", contexto: pd.DataFrame, origens: list[int], limite_k: int,
                         min_pits: int = 200) -> Recalibracao:
    """Estima τ' = G⁻¹(τ) por grupo e antecedência nas ``origens`` (só calibração).

    Com menos de ``min_pits`` PITs numa antecedência, usa os PITs de todas as
    antecedências do grupo (mapa único); grupo sem PITs suficientes fica sem
    recalibração."""
    niv = np.asarray(conj.niveis, dtype=float)
    U = pits_previsao(conj, contexto, origens, limite_k)
    grupos = np.array([grupo_serie(s) for s in conj.series])
    mapa, n_pits = {}, {}
    for g in dict.fromkeys(grupos):
        Ug = U[:, grupos == g, :]                                  # (O, Sg, H)
        todos = Ug[~np.isnan(Ug)]
        if len(todos) < min_pits:
            continue
        tau_pool = np.quantile(todos, niv)
        tau = np.empty((conj.H, len(niv)))
        n = np.zeros(conj.H, dtype=int)
        for h in range(conj.H):
            u = Ug[:, :, h]
            u = u[~np.isnan(u)]
            n[h] = len(u)
            tau[h] = np.quantile(u, niv) if len(u) >= min_pits else tau_pool
        mapa[g] = np.maximum.accumulate(np.clip(tau, 1e-6, 1 - 1e-6), axis=1)   # monotônico em τ
        n_pits[g] = n
    return Recalibracao(niv, mapa, n_pits, (min(origens), max(origens)))


def _gravar_atomico(df: pd.DataFrame, caminho: str) -> None:
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    tmp = caminho + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, caminho)


def criar_previsor(cfg: Config, cal: CalendarioOperacional, modelo: str | None = None,
                   aditivas: np.ndarray | None = None) -> Previsor:
    modelo = modelo or cfg.previsao.modelo
    if modelo == "chronos2":
        return Chronos2Previsor(cfg, cal)
    if modelo == "seasonal_naive":
        return SeasonalNaiveQuantilico(cfg.previsao.niveis, cfg.previsao.snaive_semanas, aditivas)
    raise ValueError(f"modelo de previsão desconhecido: {modelo}")


__all__ = ["Previsor", "SeasonalNaiveQuantilico", "Chronos2Previsor", "ConjuntoPrevisoes",
           "prever_origens", "criar_previsor", "rotulos_quantis", "pos_processar", "Recalibracao",
           "ajustar_recalibracao", "pits_previsao", "grupo_serie"]
