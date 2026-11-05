"""Calendário operacional: quem opera em que dia.

- ATMs operam todos os dias (o NN5 tem saque no fim de semana).
- GO e agências: dias úteis (seg–sex, exceto feriados bancários).
- Custódia do BC: dias úteis.
- Transportadora: seg–sáb, exceto feriados (configurável).

Feriados bancários da Inglaterra e do País de Gales em 1995–1998 (o NN5 é de
ATMs ingleses). A lista é fixa no código e conferida contra o pacote
``holidays`` quando ele está instalado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

# Feriados bancários da Inglaterra e do País de Gales (gov.uk), 1995–1998
FERIADOS_EW = [
    "1995-01-02", "1995-04-14", "1995-04-17", "1995-05-08", "1995-05-29", "1995-08-28",
    "1995-12-25", "1995-12-26",
    "1996-01-01", "1996-04-05", "1996-04-08", "1996-05-06", "1996-05-27", "1996-08-26",
    "1996-12-25", "1996-12-26",
    "1997-01-01", "1997-03-28", "1997-03-31", "1997-05-05", "1997-05-26", "1997-08-25",
    "1997-12-25", "1997-12-26",
    "1998-01-01", "1998-04-10", "1998-04-13", "1998-05-04", "1998-05-25", "1998-08-31",
    "1998-12-25", "1998-12-28",
]


def feriados_ew() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(FERIADOS_EW))


def conferir_feriados() -> list[str]:
    """Compara a lista fixa com o pacote ``holidays``; devolve as divergências."""
    try:
        import holidays  # type: ignore
    except ImportError:
        return ["pacote holidays indisponível: conferência não feita"]
    h = holidays.UnitedKingdom(subdiv="ENG", years=range(1995, 1999))
    pacote = {pd.Timestamp(d) for d in h if pd.Timestamp(d).dayofweek < 5}
    fixa = set(feriados_ew())
    dif = []
    for d in sorted(pacote - fixa):
        dif.append(f"no pacote e fora da lista: {d.date()}")
    for d in sorted(fixa - pacote):
        dif.append(f"na lista e fora do pacote: {d.date()}")
    return dif


@dataclass
class CalendarioOperacional:
    """Flags diárias sobre um índice de datas contínuo (índice inteiro k)."""

    datas: pd.DatetimeIndex
    feriado: np.ndarray
    transp: np.ndarray
    go: np.ndarray
    bc: np.ndarray
    L_BC: int
    L_C: int
    R: int

    # ------------------------------------------------------------------
    @classmethod
    def construir(cls, cfg: Config, inicio: str | pd.Timestamp, fim: str | pd.Timestamp) -> "CalendarioOperacional":
        datas = pd.date_range(inicio, fim, freq="D")
        fer = datas.isin(feriados_ew())
        dow = datas.dayofweek.values
        cal = cfg.calendario
        transp = np.isin(dow, cal.transportadora_dias_semana)
        if not cal.transportadora_opera_feriado:
            transp &= ~fer
        go = np.isin(dow, cal.go_dias_semana) & ~fer
        bc = np.isin(dow, cal.custodia_dias_semana) & ~fer
        return cls(datas=datas, feriado=np.asarray(fer), transp=transp, go=go, bc=bc,
                   L_BC=cfg.custos.L_BC, L_C=cfg.custos.L_C, R=cfg.es.R_revisao)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.datas)

    def k(self, data: str | pd.Timestamp) -> int:
        """Índice inteiro de uma data."""
        return int(self.datas.get_loc(pd.Timestamp(data)))

    def data(self, k: int) -> pd.Timestamp:
        return self.datas[k]

    def _proximo(self, flag: np.ndarray, k: int, inclusive: bool = False) -> int:
        i = k if inclusive else k + 1
        n = len(flag)
        while i < n and not flag[i]:
            i += 1
        return i  # pode ser n (fora do calendário): o chamador trata

    def recebe_tesouraria(self, k: int) -> bool:
        """A tesouraria só recebe transportadora com o GO presente (R5)."""
        return bool(self.transp[k] and self.go[k]) if 0 <= k < len(self) else False

    def prox_go(self, k: int) -> int:
        return self._proximo(self.go, k)

    def prox_transp(self, k: int) -> int:
        return self._proximo(self.transp, k)

    def prox_bc(self, k: int) -> int:
        return self._proximo(self.bc, k)

    def prox_recebe_tesouraria(self, k: int, inclusive: bool = False) -> int:
        flag = self.transp & self.go
        return self._proximo(flag, k, inclusive)

    def chegada_bc(self, k: int) -> int:
        """Dia em que chega uma remessa pedida à custódia na noite de k:
        o L_BC-ésimo dia útil da custódia depois de k."""
        i = k
        for _ in range(self.L_BC):
            i = self.prox_bc(i)
        return i

    def pode_pedir_bc(self, k: int) -> bool:
        """R13: só se pede em dia de custódia e com chegada em dia de custódia."""
        if not (0 <= k < len(self)) or not self.bc[k]:
            return False
        a = self.chegada_bc(k)
        return a < len(self) and bool(self.bc[a])

    # ------------------------------------------------------------------
    # Períodos de proteção (seções 5.6 e 5.8)
    # ------------------------------------------------------------------
    def periodo_protecao_agencia(self, t: int, L_extra: int = 0) -> list[int]:
        """𝒫_{b,t} = {t+1, …, a_t − 1}, a_t = 1º dia em que a tesouraria pode
        receber a transportadora ≥ t + R + L_C (+ L_extra na rede distribuída)."""
        alvo = t + self.R + self.L_C + L_extra
        a = self.prox_recebe_tesouraria(alvo - 1) if alvo - 1 >= 0 else alvo
        return list(range(t + 1, a))

    def periodo_protecao_central(self, t: int) -> list[int]:
        """𝒫^CD_{c,t} = {t+1, …, a^BC_t − 1}, a^BC_t = chegada de um pedido
        feito no próximo dia útil da custódia."""
        n = self.prox_bc(t)
        a = self.chegada_bc(n)
        return list(range(t + 1, a))

    # ------------------------------------------------------------------
    def covariaveis(self) -> pd.DataFrame:
        """Covariáveis de calendário conhecidas no futuro (para o Chronos-2):
        feriado e dias até/desde o feriado, truncados em 7."""
        n = len(self)
        fer = self.feriado.astype(int)
        ate = np.full(n, 7, dtype=float)
        desde = np.full(n, 7, dtype=float)
        ultimo = -10_000
        for i in range(n):
            if fer[i]:
                ultimo = i
            desde[i] = min(7, i - ultimo)
        proximo = 10_000
        for i in range(n - 1, -1, -1):
            if fer[i]:
                proximo = i
            ate[i] = min(7, proximo - i)
        return pd.DataFrame({"feriado": fer.astype(float), "dias_ate_feriado": ate,
                             "dias_desde_feriado": desde}, index=self.datas)

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame({
            "data": self.datas, "dia_semana": self.datas.dayofweek, "feriado": self.feriado,
            "transportadora": self.transp, "go": self.go, "custodia": self.bc,
        })


__all__ = ["CalendarioOperacional", "FERIADOS_EW", "feriados_ew", "conferir_feriados"]
