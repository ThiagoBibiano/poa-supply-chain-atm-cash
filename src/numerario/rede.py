"""Rede multiescalão do piloto: custódia do BC → tesouraria central →
tesourarias das agências → ATMs.

- ATMs internos: abastecidos pelo GO a partir da tesouraria da agência.
- ATMs externos: supridos pela transportadora direto da tesouraria central.
- A transportadora não é ponto de estoque: executa os fluxos entre central,
  agências e ATMs externos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .dados import DadosProjeto


def _arred(v: float, passo: float = 10.0) -> float:
    return float(np.ceil(v / passo) * passo)


@dataclass
class RedePiloto:
    tesourarias: list[str]
    internos: dict[str, list[str]]          # tesouraria → ATMs internos
    externos: list[str]
    recicladores: set[str]
    media: dict[str, float]                 # saque médio diário (R$ mil)
    cap: dict[str, float]                   # Cap_a
    teto: dict[str, float]                  # T^max_b
    Rb: dict[str, float]                    # reserva operacional R_b
    theta: dict[str, float]                 # θ_j (min)
    theta_cf: float
    Theta: float
    Zmax: float
    Rc: float
    saldo_inicial: dict[str, float] = field(default_factory=dict)
    agencia_de: dict[str, str] = field(default_factory=dict)  # ATM interno → tesouraria

    # ------------------------------------------------------------------
    @property
    def atms(self) -> list[str]:
        return [j for b in self.tesourarias for j in self.internos[b]] + list(self.externos)

    @property
    def atms_internos(self) -> list[str]:
        return [j for b in self.tesourarias for j in self.internos[b]]

    @property
    def pontos_transp(self) -> list[str]:
        """𝒩_c: agências e ATMs externos supridos pela central via transportadora."""
        return list(self.tesourarias) + list(self.externos)

    def eh_reciclador(self, a: str) -> bool:
        return a in self.recicladores

    # ------------------------------------------------------------------
    @classmethod
    def construir(cls, cfg: Config, dados: DadosProjeto) -> "RedePiloto":
        r = cfg.rede
        hist = dados.saques.loc[:cfg.datas.calibracao_fim]
        media = {a: float(hist[a].mean()) for a in dados.saques.columns}
        soma3 = hist.rolling(3).sum()
        tes = [f"T{b}" for b in range(1, r.n_agencias + 1)]
        internos = {f"T{b}": [f"A{b}.{j}" for j in range(1, r.atms_internos_por_agencia + 1)] for b in range(1, r.n_agencias + 1)}
        externos = [f"E{k}" for k in range(1, r.n_atms_externos + 1)]
        agencia_de = {j: b for b, js in internos.items() for j in js}

        cap, theta = {}, {}
        for a in media:
            dias = r.cap_dias_interno if a.startswith("A") else r.cap_dias_externo
            p99_3d = float(np.nanquantile(soma3[a].values, 0.99))
            cap[a] = _arred(max(dias * media[a], 1.1 * p99_3d))
        for b, js in internos.items():
            nb = int(b[1:])
            th = r.theta_min_por_agencia.get(nb, r.theta_min_padrao)
            for i, j in enumerate(js):
                theta[j] = float(th[i % len(th)])

        teto, Rb = {}, {}
        for b, js in internos.items():
            m = sum(media[j] for j in js)
            teto[b] = _arred(r.teto_dias_tesouraria * m)
            Rb[b] = _arred(r.reserva_dias_tesouraria * m, 5.0)
        m_reg = sum(media.values())
        Zmax = _arred(r.teto_dias_central * m_reg, 100.0)
        Rc = _arred(r.reserva_dias_central * m_reg, 50.0)

        rec = set(dados.depositos.columns) if dados.depositos is not None else set()

        saldo = {}
        for a in media:
            saldo[a] = min(cap[a], r.saldo_inicial_atm_dias * media[a])
        for b, js in internos.items():
            saldo[b] = min(teto[b], r.saldo_inicial_tesouraria_dias * sum(media[j] for j in js))
        saldo["CD"] = min(Zmax, r.saldo_inicial_central_dias * m_reg)

        return cls(tesourarias=tes, internos=internos, externos=externos, recicladores=rec,
                   media=media, cap=cap, teto=teto, Rb=Rb, theta=theta, theta_cf=r.theta_cf_min,
                   Theta=r.Theta_min_dia, Zmax=Zmax, Rc=Rc, saldo_inicial=saldo, agencia_de=agencia_de)

    # ------------------------------------------------------------------
    def tabela_pontos(self, cfg: Config) -> pd.DataFrame:
        """Tabela ``pontos`` do contrato de dados."""
        linhas = [
            dict(ponto="BC", tipo="bc", nome="Custódia do BC (operada pelo BB)", central="CD", agencia=None,
                 reciclador=False, capacidade=None, teto=None, reserva=None, alpha=None, theta_min=None, media=None),
            dict(ponto="CD", tipo="cd", nome="Tesouraria central", central="CD", agencia=None, reciclador=False,
                 capacidade=None, teto=self.Zmax, reserva=self.Rc, alpha=cfg.es.alpha, theta_min=None,
                 media=sum(self.media.values())),
        ]
        for b in self.tesourarias:
            linhas.append(dict(ponto=b, tipo="tes", nome=f"Tesouraria · Agência {b[1:]}", central="CD", agencia=int(b[1:]),
                               reciclador=False, capacidade=None, teto=self.teto[b], reserva=self.Rb[b],
                               alpha=cfg.es.alpha, theta_min=None, media=sum(self.media[j] for j in self.internos[b])))
        for b in self.tesourarias:
            for j in self.internos[b]:
                linhas.append(dict(ponto=j, tipo="int", nome=f"ATM interno {j[1:]}", central="CD", agencia=int(b[1:]),
                                   reciclador=self.eh_reciclador(j), capacidade=self.cap[j], teto=None, reserva=None,
                                   alpha=None, theta_min=self.theta[j], media=self.media[j]))
        for k in self.externos:
            linhas.append(dict(ponto=k, tipo="ext", nome=f"ATM externo {k[1:]}", central="CD", agencia=None,
                               reciclador=self.eh_reciclador(k), capacidade=self.cap[k], teto=None, reserva=None,
                               alpha=None, theta_min=None, media=self.media[k]))
        return pd.DataFrame(linhas)


__all__ = ["RedePiloto"]
