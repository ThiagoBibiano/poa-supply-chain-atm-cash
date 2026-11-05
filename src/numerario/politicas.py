"""Políticas comparadas (proposta, seção 8.2).

| Política | Método                                                                 | Para que serve           |
|----------|------------------------------------------------------------------------|--------------------------|
| P0       | Regra fixa: média histórica + margem, sem sazonalidade; reservas fixas | referência operacional   |
| P1       | Mesmo MILP com cenário único = mediana (P50); sem estoque de segurança | base do VSS              |
| P2       | Políticas clássicas por quantil: base-stock pelo quantil da soma       | benchmark da literatura  |
|          | (Phan et al., 2026) com limite superior pelo quantil inferior (Zhou et |                          |
|          | al., 2020); especial como 2º índice (Veeraraghavan e Scheller-Wolf,    |                          |
|          | 2008); GO por menor cobertura até Θ; central (s,Q) com lote de Baumol  |                          |
|          | e faixa de Miller e Orr; tesourarias com PP como gatilho e EM de piso  |                          |
| P3       | MILP estocástico de dois estágios (seção 4), com R10 e R14             | proposta do projeto      |
| P4       | Oráculo: MILP com a demanda real conhecida                             | limite inferior; EVPI    |

Estoque mínimo por política (seção 8.2): P0 reserva fixa; P1 e P4 só R_b e
R_c; P2 a fórmula das seções 5.6/5.8; P3 a restrição flexível R10/R14.

As regras de P0 e P2 compartilham o mesmo motor (como no protótipo do painel):
só mudam os estimadores de demanda e os estoques mínimos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calendario import CalendarioOperacional
from .cenarios import CenariosOrigem
from .config import Config
from .estado import Decisao, Decisoes, EstadoRede
from .estoque_seguranca import CalculadoraEM, IndiceSeries, _col, nome_deposito, nome_guiche
from .modelo import InstanciaMILP, ModeloNumerario
from .rede import RedePiloto


@dataclass
class ContextoNoite:
    """O que a política enxerga na noite k (fim do dia k)."""

    k: int
    estado: EstadoRede
    cen: CenariosOrigem | None = None
    em: dict | None = None                 # CalculadoraEM.horizonte(k, cen)
    real_futuro: np.ndarray | None = None  # (séries, H): só o oráculo usa
    media_hist: dict | None = None         # P0: média histórica por série


# ---------------------------------------------------------------------------
# Políticas baseadas no MILP (P1, P3, P4)
# ---------------------------------------------------------------------------
class PoliticaMILP:
    def __init__(self, nome: str, cfg: Config, rede: RedePiloto, cal: CalendarioOperacional,
                 idx: IndiceSeries, modo: str, usar_em: bool):
        self.nome, self.cfg, self.rede, self.cal, self.idx = nome, cfg, rede, cal, idx
        self.modo = modo                  # "mediana" | "estocastico" | "oraculo"
        self.usar_em = usar_em
        self.modelo = ModeloNumerario(cfg, rede, cal)

    def _arrays(self, cen_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Separa saques (D), depósitos (G) e guichê (F) de um array (S, séries, H)."""
        S, _, H = cen_arr.shape
        atms = self.rede.atms
        D = np.stack([cen_arr[:, self.idx.i(a), :] for a in atms], axis=1)
        G = None
        if any(self.idx.tem(nome_deposito(a)) for a in atms):
            G = np.zeros_like(D)
            for i, a in enumerate(atms):
                if self.idx.tem(nome_deposito(a)):
                    G[:, i, :] = cen_arr[:, self.idx.i(nome_deposito(a)), :]
        F = None
        if any(self.idx.tem(nome_guiche(b)) for b in self.rede.tesourarias):
            F = np.zeros((S, len(self.rede.tesourarias), H))
            for i, b in enumerate(self.rede.tesourarias):
                if self.idx.tem(nome_guiche(b)):
                    F[:, i, :] = cen_arr[:, self.idx.i(nome_guiche(b)), :]
        return np.maximum(D, 0.0), (None if G is None else np.maximum(G, 0.0)), F

    def _em(self, ctx: ContextoNoite, H: int) -> tuple[dict, list]:
        rede = self.rede
        if self.usar_em and ctx.em is not None:
            EM = {b: [ctx.em["agencias"][b][tau].em_formula for tau in range(1, H + 1)] for b in rede.tesourarias}
            EMCD = [ctx.em["central"][tau].em_formula for tau in range(1, H + 1)]
        else:
            EM = {b: [rede.Rb[b]] * H for b in rede.tesourarias}
            EMCD = [rede.Rc] * H
        return EM, EMCD

    def decidir(self, ctx: ContextoNoite) -> Decisoes:
        H = self.cfg.previsao.horizonte
        EM, EMCD = self._em(ctx, H)
        if self.modo == "oraculo":
            arr = ctx.real_futuro[None]
            probs = np.ones(1)
        elif self.modo == "mediana":
            arr = ctx.cen.mediana[None]
            probs = np.ones(1)
        else:
            arr = ctx.cen.cenarios
            probs = ctx.cen.probs
        D, G, F = self._arrays(arr)
        inst = InstanciaMILP(k0=ctx.k, H=H, estado=ctx.estado, D=D, G=G, F=F, probs=probs,
                             EM=EM, EMCD=EMCD, rotulo=self.nome)
        inicial = None
        if self.modo == "estocastico" and self.cfg.solver.partida_quente:
            Dm, Gm, Fm = self._arrays(ctx.cen.mediana[None])
            det = InstanciaMILP(k0=ctx.k, H=H, estado=ctx.estado, D=Dm, G=Gm, F=Fm, probs=np.ones(1),
                                EM=EM, EMCD=EMCD, rotulo=self.nome + "_det")
            inicial = self.modelo.resolver(det).papeis
        sol = self.modelo.resolver(inst, inicial=inicial)
        dec = sol.decisoes
        dec.info["duais"] = sol.duais
        if not sol.stats.get("ok"):
            # sem solução: cai para a regra da P0 nesta noite (registrado em execucoes)
            dec.info["fallback"] = True
        return dec


# ---------------------------------------------------------------------------
# Motor das regras (P0 e P2)
# ---------------------------------------------------------------------------
class Estimador:
    """Estimativas de somas de demanda: (série, dia, sinal) → valor."""

    def soma(self, pares: list[tuple], nivel: float) -> float:
        raise NotImplementedError


class EstimadorMedia(Estimador):
    """P0: média histórica (sem sazonalidade); níveis acima de 0,5 levam a margem."""

    def __init__(self, media: dict, margem: float):
        self.media, self.margem = media, margem

    def soma(self, pares, nivel):
        f = 1.0 + self.margem if nivel > 0.5 else (1.0 if nivel == 0.5 else max(0.0, 1.0 - self.margem))
        tot = 0.0
        for s, _, sinal in pares:
            m = self.media.get(s, 0.0)
            tot += sinal * m * (f if sinal > 0 else 1.0)
        return tot


class EstimadorTrajetorias(Estimador):
    """P2: quantil da soma calculado nas trajetórias da cópula."""

    def __init__(self, cen: CenariosOrigem, idx: IndiceSeries, k0: int):
        self.traj, self.idx, self.k0 = cen.trajetorias, idx, k0
        self.H = cen.trajetorias.shape[2]
        self._cache: dict = {}

    def vetor(self, pares) -> np.ndarray:
        tot = np.zeros(self.traj.shape[0])
        for s, y, sinal in pares:
            if not self.idx.tem(s):
                continue
            col = _col(self.traj[:, self.idx.i(s), :], self.k0, y, self.H)
            if col is not None:
                tot += sinal * col
        return tot

    def soma(self, pares, nivel):
        if not pares:
            return 0.0
        chave = (tuple(pares), nivel)
        if chave not in self._cache:
            self._cache[chave] = float(np.quantile(self.vetor(pares), nivel))
        return self._cache[chave]


class PoliticaRegra:
    """Regras de P0 e P2 (motor comum)."""

    def __init__(self, nome: str, cfg: Config, rede: RedePiloto, cal: CalendarioOperacional,
                 idx: IndiceSeries, calc_em: CalculadoraEM, margem_p0: float | None = None):
        self.nome, self.cfg, self.rede, self.cal, self.idx = nome, cfg, rede, cal, idx
        self.calc_em = calc_em
        self.margem = cfg.experimento.margem_p0_padrao if margem_p0 is None else margem_p0
        c = cfg.custos
        self.L = {"C": c.L_C, "E": c.L_E}
        self.alpha = cfg.es.alpha
        self.cobertura_ciclo = 3  # dias extras cobertos pelo pedido convencional (amortiza o fixo)

    # ---------------- pares (série, dia, sinal) ----------------
    def pares_atm(self, a: str, dias) -> list[tuple]:
        out = [(a, y, 1.0) for y in dias]
        if self.idx.tem(nome_deposito(a)):
            out += [(nome_deposito(a), y, -1.0) for y in dias]
        return out

    def pares_tes(self, b: str, dias_x) -> list[tuple]:
        out = []
        for x in dias_x:
            if not (0 <= x < len(self.cal)) or not self.cal.go[x]:
                continue
            g = self.cal.prox_go(x)
            for j in self.rede.internos[b]:
                out += self.pares_atm(j, range(x, g))
            if self.idx.tem(nome_guiche(b)):
                out.append((nome_guiche(b), x, -1.0))
        return out

    # ---------------- estimadores por política ----------------
    def _estimador(self, ctx: ContextoNoite) -> Estimador:
        if self.nome == "P0":
            return EstimadorMedia(ctx.media_hist, self.margem)
        return EstimadorTrajetorias(ctx.cen, self.idx, ctx.k)

    def _em_tes(self, ctx, b: str, tau: int = 0) -> tuple[float, float | None]:
        """(EM, PP) da tesouraria na política; PP None quando a política não usa."""
        if self.nome == "P0":
            m = sum(self.rede.media[j] for j in self.rede.internos[b])
            return self.cfg.es.reserva_fixa_p0_tesouraria_dias * m, None
        e = ctx.em["agencias"][b][tau]
        return e.em_formula, e.pp

    def _em_cd(self, ctx, tau: int = 0) -> tuple[float, float | None]:
        if self.nome == "P0":
            return self.cfg.es.reserva_fixa_p0_central_dias * sum(self.rede.media.values()), None
        e = ctx.em["central"][tau]
        return e.em_formula, e.pp

    # ---------------- decisão de uma noite ----------------
    def decidir(self, ctx: ContextoNoite) -> Decisoes:
        dec = self._decidir_noite(ctx, ctx.estado, ctx.k)
        dec.plano = self._plano(ctx, dec)
        return dec

    def _decidir_noite(self, ctx: ContextoNoite, est: EstadoRede, k: int) -> Decisoes:
        cal, rede = self.cal, self.rede
        E = self._estimador(ctx)
        a_hi, a_lo = (self.alpha, 1 - self.alpha) if self.nome != "P0" else (0.95, 0.05)
        dec = Decisoes(k0=k)
        LC, LE = self.L["C"], self.L["E"]

        def chegada(n, y):
            return est.chegada_prevista(n, y) + sum(dec.pedidos.get(n, {}).get(m, 0.0) for m in ("C", "E")
                                                     if y == k + self.L[m])

        # ---------- tesourarias ----------
        for b in rede.tesourarias:
            T = est.T[b]
            rec1 = est.recolhimentos.get(k + 1, {}).get(b, 0.0)
            EMb, PPb = self._em_tes(ctx, b)
            teto = rede.teto[b]
            # especial para amanhã (2º índice): o cofre não aguenta o GO de amanhã
            if cal.recebe_tesouraria(k + 1) and LE == 1:
                disp = T + est.chegada_prevista(b, k + 1) - rec1
                prox_conv = cal.prox_recebe_tesouraria(k + 1 + LC - 1)
                sai_amanha = E.soma(self.pares_tes(b, [k + 1]), a_hi)
                if disp < sai_amanha + rede.Rb[b]:
                    alvo = E.soma(self.pares_tes(b, range(k + 1, prox_conv)), a_hi) + EMb
                    x = min(teto, alvo) - disp
                    if x > 1.0:
                        dec.pedidos.setdefault(b, {})["E"] = x
            # convencional para k + L_C (gatilho PP; teto pelo quantil baixo)
            a = k + LC
            if cal.recebe_tesouraria(a) and not est.tem_visita(b, a, "C"):
                ip = T + chegada(b, k + 1) - rec1
                proj = ip - E.soma(self.pares_tes(b, range(k + 1, a)), 0.5)
                prox = cal.prox_recebe_tesouraria(a)
                if PPb is None:  # P0: cobre até a próxima convencional + reserva fixa
                    need = EMb + E.soma(self.pares_tes(b, range(a, prox)), a_hi)
                    gatilho = proj < need
                else:            # P2: posição de estoque abaixo do ponto de pedido
                    gatilho = ip < PPb
                if gatilho:
                    alvo = EMb + E.soma(self.pares_tes(b, range(a, a + 1 + self.cobertura_ciclo)), a_hi)
                    teto_ef = teto
                    if self.nome != "P0":  # Zhou et al.: não estourar o teto se a demanda vier baixa
                        teto_ef = teto + E.soma(self.pares_tes(b, range(k + 1, a)), a_lo) - E.soma(
                            self.pares_tes(b, range(k + 1, a)), 0.5)
                    x = min(teto_ef, alvo) - max(proj, 0.0)
                    if x > 1.0:
                        dec.pedidos.setdefault(b, {})["C"] = x
                else:
                    # lado do excesso: recolher se o cofre ficar acima do teto
                    limite = 0.92 * teto if self.nome == "P0" else teto
                    baixo = proj if self.nome == "P0" else ip - E.soma(self.pares_tes(b, range(k + 1, a)), a_lo)
                    if baixo > limite:
                        manter = EMb + E.soma(self.pares_tes(b, range(a, a + 3)), 0.5)
                        r = max(0.0, proj - manter)
                        if r > 1.0:
                            dec.pedidos.setdefault(b, {})["C"] = 0.0
                            dec.recolher_futuro[(b, a)] = r

        # ---------- GO: agenda de amanhã ----------
        if cal.go[k + 1] if k + 1 < len(cal) else False:
            g = cal.prox_go(k + 1)
            for b in rede.tesourarias:
                nvis = len(est.visitas.get(k + 1, {}).get(b, {})) + (1 if "E" in dec.pedidos.get(b, {}) else 0)
                minutos = rede.theta_cf * nvis
                caixa = est.T[b] + chegada(b, k + 1) - est.recolhimentos.get(k + 1, {}).get(b, 0.0) - rede.Rb[b]
                cands = []
                for j in rede.internos[b]:
                    ini = est.I[j]
                    need = E.soma(self.pares_atm(j, range(k + 1, g)), a_hi)
                    if ini < need:
                        alvo = min(rede.cap[j], E.soma(self.pares_atm(j, range(k + 1, g + 1)), a_hi))
                        med_dia = max(E.soma(self.pares_atm(j, [k + 1]), 0.5), 1e-6)
                        cands.append((ini / med_dia, j, ini, alvo))
                cands.sort()
                for _, j, ini, alvo in cands:  # menor cobertura primeiro, até Θ
                    if minutos + rede.theta[j] > rede.Theta:
                        continue
                    qv = min(alvo - ini, max(caixa, 0.0))
                    if qv > 1.0:
                        dec.go[j] = (qv, 0.0)
                        caixa -= qv
                        minutos += rede.theta[j]

        # ---------- ATMs externos ----------
        for kk in rede.externos:
            I = est.I[kk]
            cap = rede.cap[kk]
            # especial para amanhã
            if cal.transp[k + 1] and LE == 1:
                disp = I + est.chegada_prevista(kk, k + 1)
                if disp < E.soma(self.pares_atm(kk, [k + 1]), a_hi):
                    prox_conv = cal._proximo(cal.transp, k + 1 + LC - 1)
                    alvo = min(cap, E.soma(self.pares_atm(kk, range(k + 1, prox_conv)), a_hi))
                    x = alvo - disp
                    if x > 1.0:
                        dec.pedidos.setdefault(kk, {})["E"] = x
            a = k + LC
            if a < len(cal) and cal.transp[a] and not est.tem_visita(kk, a, "C"):
                ip = I + chegada(kk, k + 1)
                proj = ip - E.soma(self.pares_atm(kk, range(k + 1, a)), 0.5)
                prox = cal._proximo(cal.transp, a)
                if self.nome == "P0":
                    gatilho = proj < E.soma(self.pares_atm(kk, range(a, prox)), a_hi)
                else:  # base-stock: posição abaixo do quantil da demanda até a próxima chegada possível
                    gatilho = ip < E.soma(self.pares_atm(kk, range(k + 1, prox)), a_hi)
                if gatilho:
                    alvo = E.soma(self.pares_atm(kk, range(a, a + 1 + self.cobertura_ciclo)), a_hi)
                    teto_ef = cap
                    if self.nome != "P0":
                        teto_ef = cap + E.soma(self.pares_atm(kk, range(k + 1, a)), a_lo) - E.soma(
                            self.pares_atm(kk, range(k + 1, a)), 0.5)
                    x = min(teto_ef, alvo + max(proj, 0.0)) - max(proj, 0.0)
                    x = min(x, cap - max(proj, 0.0))
                    if x > 1.0:
                        dec.pedidos.setdefault(kk, {})["C"] = x


        # ---------- tesouraria central e custódia ----------
        if cal.pode_pedir_bc(k):
            a = cal.chegada_bc(k)
            EMc, PPc = self._em_cd(ctx)
            proj = est.Z
            for y in range(k + 1, a + 1):
                proj += est.bc_chegada.get(y, 0.0) - est.bc_saida.get(y, 0.0) + est.processamento.get(y, 0.0)
                if y < a:
                    proj -= self._despacho_previsto(ctx, E, est, dec, y, k)
            a2 = cal.chegada_bc(cal.prox_bc(k))
            prox_saida = sum(self._despacho_previsto(ctx, E, est, dec, y, k, nivel=0.5) for y in range(a, a2))
            need = EMc + prox_saida
            if proj < need:
                falta = need - proj
                qx = falta if self.nome == "P0" else max(self.calc_em.Q_baumol, falta)
                qx = min(qx, rede.Zmax - max(proj, 0.0))
                if qx > 1.0:
                    dec.bc_retirada = qx
            else:
                if self.nome == "P0":
                    if proj > 0.9 * rede.Zmax:
                        dec.bc_devolucao = proj - 0.7 * rede.Zmax
                else:  # faixa de Miller e Orr + critério de ida e volta (seção 5.7)
                    z_ret, H_mo = self.calc_em.faixa_miller_orr(need)
                    if proj > H_mo:
                        X = proj - z_ret
                        if X > 1.0 and self.calc_em.compensa_devolver(X):
                            dec.bc_devolucao = X
        return dec

    def _despacho_previsto(self, ctx, E: Estimador, est: EstadoRede, dec: Decisoes, y: int, k: int,
                           nivel: float | None = None) -> float:
        """Saída da central no dia y: pedidos já feitos (até k + L_C) ou a saída
        líquida estimada da região."""
        if y <= k + self.L["C"]:
            tot = sum(sum(v.values()) for v in est.visitas.get(y, {}).values())
            for n, mods in dec.pedidos.items():
                for m, x in mods.items():
                    if k + self.L[m] == y:
                        tot += x
            tot -= sum(est.recolhimentos.get(y - self.cfg.custos.L_P, {}).values())
            return tot
        nv = 0.5 if nivel is None else nivel
        pares = []
        for b in self.rede.tesourarias:
            pares += self.pares_tes(b, [y])
        for kk in self.rede.externos:
            pares += self.pares_atm(kk, [y])
        return E.soma(pares, nv)

    # ---------------- plano (2º estágio) por projeção ----------------
    def _plano(self, ctx: ContextoNoite, dec0: Decisoes, noites: int = 2) -> list[Decisao]:
        """Projeta as próximas noites com a demanda mediana e as mesmas regras,
        para registrar o plano que a política faria hoje para os dias seguintes."""
        from .simulacao import aplicar_decisoes, executar_dia_projetado

        plano = []
        est = ctx.estado.copia()
        aplicar_decisoes(est, dec0, self.cfg, self.cal, self.rede)
        E = self._estimador(ctx)
        for d in range(1, noites + 1):
            kk = ctx.k + d
            executar_dia_projetado(est, kk, self, E, self.cfg, self.cal, self.rede)
            dec = self._decidir_noite(ctx, est, kk)
            for n, mods in dec.pedidos.items():
                for m, x in mods.items():
                    plano.append(Decisao(n, ctx.k, kk + self.L[m], m, x, 0.0, 2, 1.0))
            for j, (qv, sv) in dec.go.items():
                plano.append(Decisao(j, ctx.k, kk + 1, "GO", qv, self.rede.theta[j], 2, 1.0))
            for n, r in dec.recolher.items():
                plano.append(Decisao(n, ctx.k, kk + 1, "REC", r, 0.0, 2, 1.0))
            for (n, dia), r in dec.recolher_futuro.items():
                plano.append(Decisao(n, ctx.k, dia, "REC", r, 0.0, 2, 1.0))
            if dec.bc_retirada > 0:
                plano.append(Decisao("CD", ctx.k, self.cal.chegada_bc(kk), "BC", dec.bc_retirada, 0.0, 2, 1.0))
            if dec.bc_devolucao > 0:
                plano.append(Decisao("CD", ctx.k, self.cal.chegada_bc(kk), "DEV", dec.bc_devolucao, 0.0, 2, 1.0))
            aplicar_decisoes(est, dec, self.cfg, self.cal, self.rede)
        # decisões executadas também entram (estágio 1) para o log
        return plano


def criar_politica(nome: str, cfg: Config, rede: RedePiloto, cal: CalendarioOperacional, idx: IndiceSeries,
                   calc_em: CalculadoraEM, margem_p0: float | None = None):
    if nome == "P0":
        return PoliticaRegra("P0", cfg, rede, cal, idx, calc_em, margem_p0)
    if nome == "P1":
        return PoliticaMILP("P1", cfg, rede, cal, idx, "mediana", usar_em=False)
    if nome == "P2":
        return PoliticaRegra("P2", cfg, rede, cal, idx, calc_em)
    if nome == "P3":
        return PoliticaMILP("P3", cfg, rede, cal, idx, "estocastico", usar_em=True)
    if nome == "P3_sem_EM":
        return PoliticaMILP("P3_sem_EM", cfg, rede, cal, idx, "estocastico", usar_em=False)
    if nome == "P4":
        return PoliticaMILP("P4", cfg, rede, cal, idx, "oraculo", usar_em=False)
    raise ValueError(f"política desconhecida: {nome}")


NOMES_POLITICAS = {
    "P0": ("Regra fixa", "média histórica + margem, sem sazonalidade; reservas fixas"),
    "P1": ("Mediana (P50)", "mesmo MILP com cenário único = P50 do Chronos-2"),
    "P2": ("Clássicas por quantil", "base-stock pelo quantil da soma, dual-index, (s,Q) com Baumol e Miller–Orr"),
    "P3": ("MILP estocástico", "dois estágios, 30 cenários da cópula, R10 e R14"),
    "P3_sem_EM": ("MILP sem estoque mínimo", "P3 sem R10/R14: mede o preço do estoque mínimo"),
    "P4": ("Oráculo", "demanda real conhecida (referência de custo mínimo)"),
}

__all__ = ["ContextoNoite", "PoliticaMILP", "PoliticaRegra", "criar_politica", "NOMES_POLITICAS",
           "EstimadorMedia", "EstimadorTrajetorias"]
