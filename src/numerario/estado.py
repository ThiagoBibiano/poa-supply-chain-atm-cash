"""Estado da rede no fim de um dia e decisões emitidas numa noite.

O simulador mantém o ``EstadoRede``; as políticas leem o estado e devolvem
``Decisoes`` (o que é executado: 1º estágio) mais um plano dos dias seguintes
(2º estágio), que o painel usa para reconstruir a evolução do plano.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class EstadoRede:
    """Saldos de fim de dia e pedidos em andamento (dias em índice absoluto k)."""

    k: int                                   # último dia fechado
    I: dict[str, float]                      # ATMs
    T: dict[str, float]                      # tesourarias das agências
    Z: float                                 # tesouraria central
    # visitas da transportadora já pedidas: {k_exec: {ponto: {"C": valor, "E": valor}}}
    visitas: dict[int, dict[str, dict[str, float]]] = field(default_factory=dict)
    # recolhimentos decididos: {k_exec: {ponto: valor}}
    recolhimentos: dict[int, dict[str, float]] = field(default_factory=dict)
    # agenda do GO: {k_exec: {atm: (q, s)}}
    go: dict[int, dict[str, tuple[float, float]]] = field(default_factory=dict)
    # custódia: retiradas que chegam e devoluções que saem: {k: valor}
    bc_chegada: dict[int, float] = field(default_factory=dict)
    bc_saida: dict[int, float] = field(default_factory=dict)
    # recolhimentos em processamento na central: {k_disponível: valor}
    processamento: dict[int, float] = field(default_factory=dict)

    def copia(self) -> "EstadoRede":
        return copy.deepcopy(self)

    def visitas_em(self, k: int) -> dict[str, dict[str, float]]:
        return self.visitas.get(k, {})

    def chegada_prevista(self, ponto: str, k: int) -> float:
        v = self.visitas.get(k, {}).get(ponto, {})
        return float(sum(v.values()))

    def tem_visita(self, ponto: str, k: int, mod: str | None = None) -> bool:
        v = self.visitas.get(k, {}).get(ponto)
        if not v:
            return False
        return (mod in v) if mod else True


@dataclass
class Decisao:
    """Uma linha da tabela ``decisoes`` (contrato de dados)."""

    ponto: str
    k_decisao: int
    k_execucao: int
    tipo: str            # C, E, GO, REC, BC, DEV, RGO (retirada do reciclador pelo GO)
    valor: float
    minutos: float = 0.0
    estagio: int = 1     # 1 = executada; 2 = plano futuro
    prob: float = 1.0    # no plano do MILP estocástico: probabilidade da visita nos cenários


@dataclass
class Decisoes:
    """O que uma política decide numa noite k0."""

    k0: int
    # 1º estágio
    pedidos: dict[str, dict[str, float]] = field(default_factory=dict)       # ponto → {"C": x, "E": x}
    recolher: dict[str, float] = field(default_factory=dict)                 # ponto → r em k0+1
    recolher_futuro: dict[tuple, float] = field(default_factory=dict)        # (ponto, k_exec) → r (regras)
    go: dict[str, tuple[float, float]] = field(default_factory=dict)         # atm → (q, s) em k0+1
    bc_retirada: float = 0.0                                                 # chega em chegada_bc(k0)
    bc_devolucao: float = 0.0                                                # sai em chegada_bc(k0)
    # 2º estágio (plano)
    plano: list[Decisao] = field(default_factory=list)
    # diagnóstico
    info: dict = field(default_factory=dict)


__all__ = ["EstadoRede", "Decisao", "Decisoes"]
