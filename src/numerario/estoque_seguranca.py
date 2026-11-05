"""Estoques de segurança das tesourarias (proposta, seções 5.6 a 5.8).

Agências (5.6)
    W_{b,x}   = Σ_{j∈J_b} Σ_{y=x}^{g(x)−1} D_{j,y} − F_{b,x}        (saída do cofre no dia útil x)
    𝒫_{b,t}   = {t+1, …, a_t − 1},  a_t = 1º dia com transportadora ≥ t + R + L_C
    EM_{b,t}  = max{ R_b ; Q_α(Σ_{x∈𝒫} W) − Q_0,5(Σ_{x∈𝒫} W) }       (estoque mínimo)
    PP_{b,t}  = Q_0,5(Σ W) + EM_{b,t}                                   (ponto de pedido)
    Aproximação: EM ≈ z_α √((1−ρ)Σσ² + ρ(Σσ)²), σ_i ≈ (Q95 − Q05)/(2·1,645)

Tesouraria central (5.8)
    W^CD_{c,x} = Σ_b W_{b,x} + Σ_k D_{k,x} − Σ_n r_{n,x−L_P}
    EM^CD      = max{ R_c ; Q_α(Σ W^CD) − Q_0,5(Σ W^CD) }
    Ganho de centralização (Eppen, 1979): G = Σ_n EM_n(sozinho) − EM^CD
    Lote de Baumol: Q* = √(2 K^BC W̄^CD / h)
    Faixa de Miller e Orr: z = L + (3 K^BC σ²/4h)^{1/3}, H = 3z − 2L, L = EM^CD
    Ida e volta (5.7): devolver X compensa se h·X·τ > 2K^BC + (c^rBC + c^BC)·X

Os quantis "da soma" saem das trajetórias da cópula (não da soma dos quantis
diários). Nos recicladores (Exp. B), a saída usa o saque líquido D − G.

Convenção de tempo: o EM "do dia t" é calculado com a previsão feita na noite
de t (origem t) e protege os dias t+1 em diante. Dentro do horizonte rolante,
os dias futuros usam a previsão de t0 (faixas mais largas) e são recalculados
toda noite. Dias de proteção além do horizonte de previsão usam a coluna do
mesmo dia da semana uma semana antes (aproximação documentada).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendario import CalendarioOperacional
from .cenarios import CenariosOrigem
from .config import Config, z_alpha
from .rede import RedePiloto

Z95 = 1.6448536269514722


def _col(arr_last_axis_H: np.ndarray, k0: int, dia: int, H: int):
    """Coluna (dia − k0 − 1) do eixo de horizonte; além de H, recua de 7 em 7."""
    j = dia - k0 - 1
    while j >= H:
        j -= 7
    if j < 0:
        return None
    return arr_last_axis_H[..., j]


@dataclass
class IndiceSeries:
    """Posição de cada série dentro dos arrays de previsão/cenário."""

    series: list[str]

    def __post_init__(self):
        self.pos = {s: i for i, s in enumerate(self.series)}

    def i(self, s: str) -> int:
        return self.pos[s]

    def tem(self, s: str) -> bool:
        return s in self.pos


def nome_deposito(a: str) -> str:
    return f"G:{a}"


def nome_guiche(b: str) -> str:
    return f"F:{b}"


@dataclass
class EMDia:
    """Detalhe do estoque mínimo de um ponto num dia (linha da tabela ``estoque_minimo``)."""

    ponto: str
    dia: int
    dias_protecao: list
    saida_mediana: float
    em_formula: float          # trajetórias, com piso R
    em_bruto: float            # trajetórias, sem piso
    em_independencia: float    # aproximação ρ = 0
    em_soma_quantis: float     # aproximação ρ = 1
    em_aprox_rho: float        # aproximação com o ρ estimado
    pp: float
    alpha: float
    rho: float
    total_separado: float | None = None
    ganho_centralizacao: float | None = None
    total_separado_aprox: float | None = None
    ganho_aprox: float | None = None


class CalculadoraEM:
    def __init__(self, cfg: Config, rede: RedePiloto, cal: CalendarioOperacional, idx: IndiceSeries,
                 rho: float, media_regiao: float, var_regiao: float):
        self.cfg, self.rede, self.cal, self.idx = cfg, rede, cal, idx
        self.alpha = cfg.es.alpha
        self.z = z_alpha(self.alpha)
        self.rho = float(rho)
        self.distribuida = cfg.es.configuracao_rede == "distribuida"
        c = cfg.custos
        # Baumol e Miller–Orr (parâmetros estáticos da região, do histórico)
        self.W_bar = float(media_regiao)
        self.var_cd = float(var_regiao)
        self.Q_baumol = float(np.sqrt(2.0 * c.K_BC * max(self.W_bar, 1e-9) / c.h))
        self.mo_z_menos_L = float((3.0 * c.K_BC * max(self.var_cd, 0.0) / (4.0 * c.h)) ** (1.0 / 3.0))

    # ------------------------------------------------------------------
    def _saida_agencia(self, b: str, x: int, k0: int, traj: np.ndarray, H: int) -> tuple[np.ndarray, list[tuple]]:
        """W_{b,x} por trajetória e os pares (série, dia) que a compõem."""
        N = traj.shape[0]
        W = np.zeros(N)
        pares = []
        if self.cal.go[x]:
            g = self.cal.prox_go(x)
            for j in self.rede.internos[b]:
                for y in range(x, g):
                    col = _col(traj[:, self.idx.i(j), :], k0, y, H)
                    if col is None:
                        continue
                    W += col
                    pares.append((j, y))
                    if self.idx.tem(nome_deposito(j)):
                        cg = _col(traj[:, self.idx.i(nome_deposito(j)), :], k0, y, H)
                        if cg is not None:
                            W -= cg
                            pares.append((nome_deposito(j), y))
            if self.idx.tem(nome_guiche(b)):
                cf = _col(traj[:, self.idx.i(nome_guiche(b)), :], k0, x, H)
                if cf is not None:
                    W -= cf
                    pares.append((nome_guiche(b), x))
        return W, pares

    def _saida_externo(self, k: str, x: int, k0: int, traj: np.ndarray, H: int) -> tuple[np.ndarray, list[tuple]]:
        N = traj.shape[0]
        W = np.zeros(N)
        pares = []
        col = _col(traj[:, self.idx.i(k), :], k0, x, H)
        if col is not None:
            W += col
            pares.append((k, x))
        if self.idx.tem(nome_deposito(k)):
            cg = _col(traj[:, self.idx.i(nome_deposito(k)), :], k0, x, H)
            if cg is not None:
                W -= cg
                pares.append((nome_deposito(k), x))
        return W, pares

    def _sigma(self, q: np.ndarray, pares: list[tuple], k0: int, H: int) -> np.ndarray:
        niv = np.asarray(self.cfg.previsao.niveis)
        i05 = int(np.argmin(np.abs(niv - 0.05)))
        i95 = int(np.argmin(np.abs(niv - 0.95)))
        out = []
        for s, y in pares:
            qq = _col(np.moveaxis(q[self.idx.i(s)], 0, -1), k0, y, H)  # (Q,)
            if qq is None:
                continue
            out.append((qq[i95] - qq[i05]) / (2.0 * Z95))
        return np.asarray(out, dtype=float)

    def _aprox(self, sig: np.ndarray, rho: float) -> float:
        if len(sig) == 0:
            return 0.0
        s2, s1 = float((sig ** 2).sum()), float(sig.sum())
        return self.z * float(np.sqrt(max((1 - rho) * s2 + rho * s1 * s1, 0.0)))

    # ------------------------------------------------------------------
    def agencia(self, b: str, t: int, k0: int, cen: CenariosOrigem) -> EMDia:
        H = cen.trajetorias.shape[2]
        L_extra = self.cfg.custos.L_BC if self.distribuida else 0
        P = self.cal.periodo_protecao_agencia(t, L_extra)
        S = np.zeros(cen.trajetorias.shape[0])
        pares = []
        for x in P:
            W, p = self._saida_agencia(b, x, k0, cen.trajetorias, H)
            S += W
            pares += p
        med = float(np.quantile(S, 0.5))
        qa = float(np.quantile(S, self.alpha))
        bruto = qa - med
        Rb = self.rede.Rb[b]
        em = max(Rb, bruto)
        sig = self._sigma(cen.quantis, pares, k0, H)
        return EMDia(ponto=b, dia=t, dias_protecao=P, saida_mediana=med, em_formula=em, em_bruto=bruto,
                     em_independencia=max(Rb, self._aprox(sig, 0.0)), em_soma_quantis=max(Rb, self._aprox(sig, 1.0)),
                     em_aprox_rho=max(Rb, self._aprox(sig, self.rho)), pp=med + em, alpha=self.alpha, rho=self.rho)

    def central(self, t: int, k0: int, cen: CenariosOrigem) -> EMDia:
        H = cen.trajetorias.shape[2]
        P = self.cal.periodo_protecao_central(t)
        N = cen.trajetorias.shape[0]
        S = np.zeros(N)
        sep_traj = 0.0
        pares_todos = []
        sep_aprox = 0.0
        for b in self.rede.tesourarias:
            Sb = np.zeros(N); pb = []
            for x in P:
                W, p = self._saida_agencia(b, x, k0, cen.trajetorias, H)
                Sb += W; pb += p
            S += Sb
            pares_todos += pb
            sep_traj += float(np.quantile(Sb, self.alpha) - np.quantile(Sb, 0.5))
            sep_aprox += self._aprox(self._sigma(cen.quantis, pb, k0, H), self.rho)
        for k in self.rede.externos:
            Sk = np.zeros(N); pk = []
            for x in P:
                W, p = self._saida_externo(k, x, k0, cen.trajetorias, H)
                Sk += W; pk += p
            S += Sk
            pares_todos += pk
            sep_traj += float(np.quantile(Sk, self.alpha) - np.quantile(Sk, 0.5))
            sep_aprox += self._aprox(self._sigma(cen.quantis, pk, k0, H), self.rho)
        med = float(np.quantile(S, 0.5))
        bruto = float(np.quantile(S, self.alpha)) - med
        Rc = self.rede.Rc
        em = Rc if self.distribuida else max(Rc, bruto)
        sig = self._sigma(cen.quantis, pares_todos, k0, H)
        em_rho = self._aprox(sig, self.rho)
        return EMDia(ponto="CD", dia=t, dias_protecao=P, saida_mediana=med, em_formula=em, em_bruto=bruto,
                     em_independencia=max(Rc, self._aprox(sig, 0.0)), em_soma_quantis=max(Rc, self._aprox(sig, 1.0)),
                     em_aprox_rho=max(Rc, em_rho), pp=max(med, 0.0) + em, alpha=self.alpha, rho=self.rho,
                     total_separado=sep_traj, ganho_centralizacao=sep_traj - bruto,
                     total_separado_aprox=sep_aprox, ganho_aprox=sep_aprox - em_rho)

    # ------------------------------------------------------------------
    def horizonte(self, k0: int, cen: CenariosOrigem) -> dict:
        """EM e PP de cada tesouraria e da central para t = k0..k0+H.

        Devolve {"agencias": {b: [EMDia τ=0..H]}, "central": [EMDia τ=0..H]}."""
        H = cen.trajetorias.shape[2]
        ag = {b: [self.agencia(b, k0 + tau, k0, cen) for tau in range(H + 1)] for b in self.rede.tesourarias}
        cd = [self.central(k0 + tau, k0, cen) for tau in range(H + 1)]
        return {"agencias": ag, "central": cd}

    # ------------------------------------------------------------------
    def faixa_miller_orr(self, L: float) -> tuple[float, float]:
        """(z, H) da faixa de Miller e Orr com limite inferior L."""
        zz = L + self.mo_z_menos_L
        return zz, 3.0 * zz - 2.0 * L

    def compensa_devolver(self, X: float) -> bool:
        """Critério de ida e volta da seção 5.7, com τ = X / W̄^CD."""
        c = self.cfg.custos
        tau = X / max(self.W_bar, 1e-9)
        return c.h * X * tau > 2.0 * c.K_BC + (c.c_rBC + c.c_BC) * X


def parametros_regiao(dados_saques: pd.DataFrame, fim_hist: str, guiche: pd.DataFrame | None = None,
                      depositos: pd.DataFrame | None = None) -> tuple[float, float]:
    """W̄^CD (consumo líquido médio diário) e σ² diário dessazonalizado da região,
    calculados só com o histórico (até o fim da calibração)."""
    tot = dados_saques.loc[:fim_hist].sum(axis=1)
    if depositos is not None:
        tot = tot - depositos.loc[:fim_hist].sum(axis=1).reindex(tot.index, fill_value=0.0)
    if guiche is not None:
        tot = tot - guiche.loc[:fim_hist].sum(axis=1).reindex(tot.index, fill_value=0.0)
    ult = tot.iloc[-112:]
    perfil = ult.groupby(ult.index.dayofweek).transform("mean")
    resid = ult - perfil
    return float(ult.mean()), float(resid.var())


def em_para_tabela(em: EMDia, cal: CalendarioOperacional) -> dict:
    d = {
        "ponto": em.ponto, "data": cal.data(em.dia),
        "dias_protecao": len(em.dias_protecao),
        "protecao_inicio": cal.data(em.dias_protecao[0]) if em.dias_protecao else pd.NaT,
        "protecao_fim": cal.data(em.dias_protecao[-1]) if em.dias_protecao else pd.NaT,
        "saida_mediana": em.saida_mediana, "em_formula": em.em_formula, "em_bruto": em.em_bruto,
        "em_independencia": em.em_independencia, "em_soma_quantis": em.em_soma_quantis,
        "em_aprox_rho": em.em_aprox_rho, "pp": em.pp, "alpha": em.alpha, "rho": em.rho,
        "total_separado": em.total_separado, "ganho_centralizacao": em.ganho_centralizacao,
        "total_separado_aprox": em.total_separado_aprox, "ganho_aprox": em.ganho_aprox,
    }
    return d


__all__ = ["CalculadoraEM", "EMDia", "IndiceSeries", "parametros_regiao", "em_para_tabela",
           "nome_deposito", "nome_guiche"]
