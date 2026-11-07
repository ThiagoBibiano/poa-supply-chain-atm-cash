"""Bloco CONFIG central do projeto.

Tudo o que é parâmetro de política, custo, rede, datas, previsão, cenários e
solver mora aqui, num só lugar. O notebook cria um ``Config()`` e altera só o
que precisar; o resto do pacote lê deste objeto.

Unidades (contrato de dados do painel, seção 7):
- valores monetários (saldos, fluxos, estoques) em **R$ mil**;
- custos em **R$**.

Os valores padrão vêm da proposta (seções 4, 5 e 8) e do protótipo do painel
(``CFG`` do HTML publicado). Onde a proposta pede um número que o NN5 não tem
(tarifas, tempos do GO, escala monetária), o valor é ilustrativo e entra na
análise de sensibilidade.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Datas do backtest (conversa "Pipeline de previsão de numerário", 01/11/2026)
# ---------------------------------------------------------------------------
@dataclass
class Datas:
    """Divisão do tempo sem sobreposição.

    | Bloco       | Período                 | Dias | Uso                                   |
    |-------------|-------------------------|------|---------------------------------------|
    | Histórico   | 18/03/1996 – 16/11/1997 | ~609 | só contexto da previsão               |
    | Calibração  | 17/11/1997 – 08/03/1998 | 112  | cobertura, cópula, limiares da P0     |
    | Aquecimento | 09/03 – 22/03/1998      | 14   | simulação rodando, resultado descartado |
    | Teste       | 23/03 – 17/05/1998      | 56   | todos os indicadores                  |
    """

    inicio_nn5: str = "1996-03-18"
    calibracao_inicio: str = "1997-11-17"
    calibracao_fim: str = "1998-03-08"
    aquecimento_inicio: str = "1998-03-09"
    teste_inicio: str = "1998-03-23"
    teste_fim: str = "1998-05-17"


# ---------------------------------------------------------------------------
# Rede piloto
# ---------------------------------------------------------------------------
@dataclass
class Rede:
    """Cluster-piloto: 1 tesouraria central, 4 agências × 3 ATMs internos e
    8 ATMs externos (20 séries do NN5)."""

    n_agencias: int = 4
    atms_internos_por_agencia: int = 3
    n_atms_externos: int = 8
    semente_series: int = 2026          # sorteio das 20 séries (semente fixa)
    max_faltantes_frac: float = 0.05    # fração máxima de faltantes/zeros na janela calibração+teste

    # Escala monetária: o NN5 tem unidade própria. A média dos saques diários
    # das 20 séries (histórico até o fim da calibração) passa a valer este
    # número em R$ mil. É o item da seção 3 da conversa de 30/10 ("saque médio
    # de um ATM em R$") que precisa vir da operação.
    saque_medio_atm_rs_mil: float = 50.0

    # Capacidades e tetos derivados do NN5, em dias de saque médio
    cap_dias_interno: float = 5.0       # Cap_a dos ATMs internos
    cap_dias_externo: float = 6.0       # Cap_a dos ATMs externos
    teto_dias_tesouraria: float = 5.0   # T^max_b (saque médio dos internos da agência)
    reserva_dias_tesouraria: float = 0.5  # R_b (reserva operacional da norma interna)
    teto_dias_central: float = 4.5      # Z^max_c (saque médio da região)
    reserva_dias_central: float = 0.3   # R_c

    # GO (gerente operacional)
    theta_min_por_agencia: dict = field(default_factory=lambda: {
        1: [30.0, 30.0, 30.0],          # agência 1: ATMs em outro endereço (fila às sextas)
    })
    theta_min_padrao: list = field(default_factory=lambda: [20.0, 22.0, 25.0])
    theta_cf_min: float = 15.0          # θ^CF: receber e conferir a transportadora
    Theta_min_dia: float = 90.0         # Θ: limite diário de minutos do GO em abastecimento

    # Saldos iniciais (o aquecimento remove o efeito)
    saldo_inicial_atm_dias: float = 2.5
    saldo_inicial_tesouraria_dias: float = 3.0
    saldo_inicial_central_dias: float = 2.0


# ---------------------------------------------------------------------------
# Calendário
# ---------------------------------------------------------------------------
@dataclass
class Calendario:
    """Quem opera em que dia. Feriados bancários da Inglaterra e País de Gales."""

    transportadora_dias_semana: tuple = (0, 1, 2, 3, 4, 5)   # seg–sáb (0 = segunda)
    go_dias_semana: tuple = (0, 1, 2, 3, 4)                  # seg–sex
    custodia_dias_semana: tuple = (0, 1, 2, 3, 4)            # seg–sex
    transportadora_opera_feriado: bool = False
    # Ciclo de decisão: a política decide toda noite corrida (como no protótipo).
    # A alternativa "dia_util" (sábado e domingo herdam o plano de sexta) ficou
    # em aberto na conversa de 01/11 e não está implementada.
    ciclo: str = "todo_dia"


# ---------------------------------------------------------------------------
# Custos (R$) e prazos
# ---------------------------------------------------------------------------
@dataclass
class Custos:
    # Transportadora (Castro, 2009: fixo por visita + componente por valor)
    L_C: int = 2                    # prazo convencional (dias)
    L_E: int = 1                    # prazo especial (dias)
    K_C: float = 600.0              # R$ por visita convencional
    K_E: float = 1500.0             # R$ por visita especial (2,5 × K_C)
    kappa: float = 0.4              # R$ por R$ mil transportado (supre e recolhe)

    # GO (Kaplan e Anderson, 2004 — cenário-base: margem perdida por minuto)
    c_GO: float = 3.5               # R$ por minuto

    # Custo de oportunidade (Baumol, 1952): h = (1+i)^(1/365) − 1
    taxa_anual: float = 0.15        # CDI / custo de captação
    # Nível de serviço-alvo para calibrar p pela razão crítica (Phan et al., 2026)
    nivel_servico_alvo: float = 0.99

    # Penalidades (R$ por R$ mil)
    p_fator_guiche: float = 1.25    # p^T = fator × p (cliente no guichê pesa mais)
    p_fator_deposito: float = 0.5   # p_w = fator × p (depósito recusado no reciclador)
    p_o: float = 5.0                # excesso sobre o teto da tesouraria, por dia
    p_EM: float = 5.0               # saldo abaixo do estoque mínimo, por dia (R10/R14)
    p_falta_central: float = 400.0  # folga de viabilidade: falta para despachar na central

    # Custódia do BC (Resolução BCB nº 134/2021) e tesouraria central
    L_BC: int = 1                   # prazo da custódia, em dias úteis
    K_BC: float = 800.0             # R$ por remessa custódia ↔ central
    c_BC: float = 1.2               # R$ por R$ mil retirado (0,12%)
    c_rBC: float = 0.8              # R$ por R$ mil devolvido (0,08%)
    c_P: float = 0.4                # R$ por R$ mil de recolhimento processado
    L_P: int = 1                    # dias até o recolhimento virar numerário disponível

    @property
    def h(self) -> float:
        """Custo de oportunidade diário em R$ por R$ mil parado."""
        return ((1.0 + self.taxa_anual) ** (1.0 / 365.0) - 1.0) * 1000.0

    @property
    def p(self) -> float:
        """Penalidade por R$ mil de saque não atendido: p ≈ h·ρ*/(1−ρ*)."""
        r = self.nivel_servico_alvo
        return self.h * r / (1.0 - r)

    @property
    def p_T(self) -> float:
        return self.p_fator_guiche * self.p

    @property
    def p_w(self) -> float:
        return self.p_fator_deposito * self.p


# ---------------------------------------------------------------------------
# Estoque de segurança (seções 5.6 e 5.8)
# ---------------------------------------------------------------------------
@dataclass
class EstoqueSeguranca:
    alpha: float = 0.95             # nível de serviço das tesourarias (90/95/99 na sensibilidade)
    R_revisao: int = 1              # revisão diária
    configuracao_rede: str = "concentrada"   # "concentrada" | "distribuida" (seção 8.1)
    reserva_fixa_p0_tesouraria_dias: float = 0.75   # P0: reserva fixa da regra atual
    reserva_fixa_p0_central_dias: float = 1.0
    # Miller–Orr: na ausência de fluxo nos dois sentidos (Exp. A) a faixa usa a
    # variância do consumo líquido diário da região.


# ---------------------------------------------------------------------------
# Previsão
# ---------------------------------------------------------------------------
NIVEIS_QUANTIS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


@dataclass
class Previsao:
    modelo: str = "chronos2"            # "chronos2" | "seasonal_naive"
    chronos_id: str = "amazon/chronos-2"
    chronos_revisao: str | None = None  # fixe um commit do Hugging Face para reprodutibilidade
    horizonte: int = 10                 # H do MILP (Bilir e Döşeyen, 2018)
    niveis: list = field(default_factory=lambda: list(NIVEIS_QUANTIS))
    covariaveis_calendario: bool = True # feriado, dias até/desde feriado (truncados em 7)
    cross_learning: bool = False
    batch_size: int = 256
    dispositivo: str = "auto"           # "auto" | "cuda" | "cpu"
    contexto_max: int | None = None     # None = histórico inteiro (janela expansiva)
    # SeasonalNaive quantílico (fallback e variante "valor do Chronos em custo")
    snaive_semanas: int = 8             # semanas de resíduos para os quantis
    # Recalibração por PIT na janela de calibração (as duas caudas, mapa por antecedência).
    # Não entra na chave do cache das previsões brutas: ligar/desligar não refaz o Chronos-2.
    recalibrar_quantis: bool = False


# ---------------------------------------------------------------------------
# Cenários (seção 6)
# ---------------------------------------------------------------------------
@dataclass
class Cenarios:
    n_trajetorias: int = 1000
    n_cenarios: int = 30
    copula: str = "gaussiana"           # "gaussiana" | "t"
    nu_t: float = 5.0                   # graus de liberdade da cópula t
    shrinkage: float = 0.2              # encolhimento das correlações em direção à identidade
    frac_cauda: float = 0.05            # 5% de cima e 5% de baixo entram como grupos próprios
    cenarios_por_cauda: int = 3
    dias_cauda: int = 3                 # dias usados para ordenar as trajetórias (proteção)
    semente: int = 7


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class Solver:
    tempo_limite_s: float = 60.0
    gap_relativo: float = 0.01
    threads: int | None = None
    dias_inteiros: int = 2              # binárias só nos primeiros dias; o resto relaxado (seção 4.6)
    especial_so_recurso: bool = True    # especial só em τo ∈ {0, 1} (dominância no 2º estágio)
    partida_quente: bool = True         # P3 parte da solução determinística da mediana
    calcular_duais: bool = False        # duais (R4, R5) com as binárias fixadas; R5 só tem binárias → dual 0
    msg: bool = False
    # Opções repassadas ao HiGHS. Padrão do HiGHS (RINS e RENS ligados): com a
    # partida a quente, a raiz do P3 consome quase todo o tempo em cortes, e são
    # essas heurísticas que melhoram o incumbente em parte das noites.
    opcoes_highs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
@dataclass
class Experimento:
    nome: str = "A"                     # "A" (empírico) | "B" (recicladores + guichê)
    politicas: list = field(default_factory=lambda: ["P0", "P1", "P2", "P3", "P4"])
    # Exp. B (sintético a partir de outras séries do NN5)
    frac_recicladores: float = 0.5
    razao_deposito_saque: float = 0.4
    guiche_medio_rs_mil: list = field(default_factory=lambda: [-20.0, 30.0, 60.0, 280.0])
    guiche_desvio_rs_mil: float = 40.0
    # Calibração da P0 na janela de calibração (margem sobre a média histórica)
    calibrar_p0: bool = True
    margens_p0: list = field(default_factory=lambda: [0.0, 0.10, 0.15, 0.25, 0.40])
    margem_p0_padrao: float = 0.15
    janela_media_p0_dias: int = 56


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
@dataclass
class Caminhos:
    raiz: str = "poa_numerario"         # no Colab: /content/drive/MyDrive/poa_numerario
    dados_fonte: str = "zenodo"         # "zenodo" | "huggingface" | "arquivo" | "sintetico"
    arquivo_nn5: str | None = None      # .zip ou .tsf já baixado (dados_fonte="arquivo")


@dataclass
class Config:
    datas: Datas = field(default_factory=Datas)
    rede: Rede = field(default_factory=Rede)
    calendario: Calendario = field(default_factory=Calendario)
    custos: Custos = field(default_factory=Custos)
    es: EstoqueSeguranca = field(default_factory=EstoqueSeguranca)
    previsao: Previsao = field(default_factory=Previsao)
    cenarios: Cenarios = field(default_factory=Cenarios)
    solver: Solver = field(default_factory=Solver)
    experimento: Experimento = field(default_factory=Experimento)
    caminhos: Caminhos = field(default_factory=Caminhos)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["custos"]["h"] = self.custos.h
        d["custos"]["p"] = self.custos.p
        d["custos"]["p_T"] = self.custos.p_T
        d["custos"]["p_w"] = self.custos.p_w
        return d

    def hash(self, *partes: str) -> str:
        """Hash curto da configuração (ou de partes dela) para chaves de cache."""
        d = self.to_dict()
        if partes:
            d = {k: d[k] for k in partes}
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    def copia(self, **alteracoes: Any) -> "Config":
        """Cópia com alterações em notação 'secao.campo' (ex.: {'custos.K_E': 1200})."""
        c = copy.deepcopy(self)
        for chave, valor in alteracoes.items():
            secao, campo = chave.split(".", 1)
            setattr(getattr(c, secao), campo, valor)
        return c

    def resumo(self) -> list[tuple[str, str]]:
        """Linhas (parâmetro, valor) para exibir no notebook."""
        c = self.custos
        return [
            ("Experimento", self.experimento.nome),
            ("Políticas", ", ".join(self.experimento.politicas)),
            ("Rede", f"1 central · {self.rede.n_agencias} agências × {self.rede.atms_internos_por_agencia} internos · {self.rede.n_atms_externos} externos"),
            ("Teste", f"{self.datas.teste_inicio} a {self.datas.teste_fim} (aquecimento desde {self.datas.aquecimento_inicio})"),
            ("Previsão", f"{self.previsao.modelo} · H = {self.previsao.horizonte} · {len(self.previsao.niveis)} quantis"),
            ("Cenários", f"{self.cenarios.n_trajetorias} trajetórias → {self.cenarios.n_cenarios} cenários · cópula {self.cenarios.copula}"),
            ("Transportadora", f"L_C = {c.L_C}, L_E = {c.L_E}, K_C = R$ {c.K_C:.0f}, K_E = R$ {c.K_E:.0f}, κ = R$ {c.kappa}/R$ mil"),
            ("GO", f"Θ = {self.rede.Theta_min_dia:.0f} min/dia, θ^CF = {self.rede.theta_cf_min:.0f} min, c_GO = R$ {c.c_GO}/min"),
            ("Custo de oportunidade", f"i = {c.taxa_anual:.0%} a.a. → h = R$ {c.h:.3f} por R$ mil/dia"),
            ("Ruptura", f"ρ* = {c.nivel_servico_alvo:.0%} → p = R$ {c.p:.1f} por R$ mil"),
            ("Custódia", f"L_BC = {c.L_BC} útil, K_BC = R$ {c.K_BC:.0f}, c_BC = {c.c_BC}, c_rBC = {c.c_rBC}, c_P = {c.c_P}"),
            ("Estoque de segurança", f"α = {self.es.alpha:.0%}, rede {self.es.configuracao_rede}"),
            ("Solver", f"HiGHS · {self.solver.tempo_limite_s:.0f} s · gap {self.solver.gap_relativo:.0%} · binárias nos {self.solver.dias_inteiros} primeiros dias"),
        ]


def z_alpha(alpha: float) -> float:
    """Quantil da normal padrão (sem depender do scipy aqui)."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(alpha)


def h_diario(taxa_anual: float) -> float:
    return (1.0 + taxa_anual) ** (1.0 / 365.0) - 1.0


__all__ = [
    "Config", "Datas", "Rede", "Calendario", "Custos", "EstoqueSeguranca",
    "Previsao", "Cenarios", "Solver", "Experimento", "Caminhos",
    "NIVEIS_QUANTIS", "z_alpha", "h_diario",
]
