"""Simulação em horizonte rolante (proposta, seções 1 e 8.3).

Ordem dos eventos no dia t:
1. Central: chega a retirada pedida à custódia, os recolhimentos processados
   (de t − L_P) viram numerário e sai a devolução programada.
2. A central despacha as visitas que vencem em t; a transportadora entrega o
   suprimento e traz o recolhimento (tesourarias e ATMs externos). Se faltar
   dinheiro na central, a entrega é rateada ("falta para despachar").
3. Em dia de expediente, o GO abastece os ATMs internos com o que está no cofre
   (e retira excesso dos recicladores), sem passar de Θ minutos.
4. Saques e depósitos nos ATMs e fluxo de guichê. Saque sem saldo vira ruptura;
   depósito além da capacidade é recusado.
5. Fecham-se os saldos. À noite, a política decide (só o 1º estágio é executado).

A cada noite: prever (já feito, em cache) → gerar cenários → resolver a
política → implementar só as decisões do dia → revelar a demanda real.
Nenhum parâmetro é escolhido olhando a janela de teste.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field

import pandas as pd

from .calendario import CalendarioOperacional
from .config import Config
from .estado import Decisoes, EstadoRede
from .rede import RedePiloto

COMPONENTES = ["conv", "esp", "go", "opor", "rupt", "estouro", "num"]


# ---------------------------------------------------------------------------
# Aplicar decisões e executar um dia
# ---------------------------------------------------------------------------
def aplicar_decisoes(est: EstadoRede, dec: Decisoes, cfg: Config, cal: CalendarioOperacional,
                     rede: RedePiloto) -> None:
    """Registra no estado o que foi decidido na noite ``dec.k0``."""
    k = dec.k0
    L = {"C": cfg.custos.L_C, "E": cfg.custos.L_E}
    for n, mods in dec.pedidos.items():
        for m, x in mods.items():
            ke = k + L[m]
            if ke >= len(cal) or not cal.transp[ke]:
                continue  # R8: não há transportadora no dia de execução
            if n in rede.teto and not cal.recebe_tesouraria(ke):
                continue  # R5: agência fechada não recebe
            d = est.visitas.setdefault(ke, {}).setdefault(n, {})
            d[m] = d.get(m, 0.0) + max(float(x), 0.0)
    for n, r in dec.recolher.items():
        if r > 0:
            est.recolhimentos.setdefault(k + 1, {})[n] = float(r)
    for (n, ke), r in dec.recolher_futuro.items():
        if r > 0:
            est.recolhimentos.setdefault(ke, {})[n] = float(r)
    if dec.go:
        est.go[k + 1] = dict(dec.go)
    if (dec.bc_retirada > 0 or dec.bc_devolucao > 0) and cal.pode_pedir_bc(k):
        a = cal.chegada_bc(k)
        if dec.bc_retirada > 0:
            est.bc_chegada[a] = est.bc_chegada.get(a, 0.0) + float(dec.bc_retirada)
        if dec.bc_devolucao > 0:
            est.bc_saida[a] = est.bc_saida.get(a, 0.0) + float(dec.bc_devolucao)


@dataclass
class ResultadoDia:
    estados: list[dict] = field(default_factory=list)
    custos: list[dict] = field(default_factory=list)
    ativas: list[dict] = field(default_factory=list)


def executar_dia(est: EstadoRede, k: int, D: dict, G: dict, F: dict, cfg: Config,
                 cal: CalendarioOperacional, rede: RedePiloto, necessidade_go: dict | None = None,
                 registrar: bool = True) -> ResultadoDia:
    """Executa o dia k sobre o estado (in place) com a demanda dada (R$ mil)."""
    c = cfg.custos
    res = ResultadoDia()
    custo = {}

    def C(ponto, comp, v):
        if v:
            custo[(ponto, comp)] = custo.get((ponto, comp), 0.0) + float(v)

    # 1) tesouraria central
    Z0 = est.Z
    chegada_bc = est.bc_chegada.pop(k, 0.0)
    proc = est.processamento.pop(k, 0.0)
    dev_plan = est.bc_saida.pop(k, 0.0)
    disp = Z0 + chegada_bc + proc
    dev = min(dev_plan, max(disp, 0.0))
    Z = disp - dev
    if chegada_bc > 0:
        C("CD", "num", c.K_BC + c.c_BC * chegada_bc)
    if dev > 0:
        C("CD", "num", c.K_BC + c.c_rBC * dev)

    # 2) transportadora
    visitas = est.visitas.pop(k, {})
    recolh = est.recolhimentos.pop(k, {})
    saldo_ant = {**{a: est.I[a] for a in rede.atms}, **{b: est.T[b] for b in rede.tesourarias}}
    quer, rec_exec, mods_exec = {}, {}, {}
    for n in rede.pontos_transp:
        v = visitas.get(n)
        if not v:
            continue
        r = min(recolh.get(n, 0.0), max(saldo_ant[n], 0.0))
        rec_exec[n] = r
        pedido = sum(v.values())
        if n in rede.cap:   # ATM externo: só cabe até a capacidade
            pedido = min(pedido, max(rede.cap[n] - (saldo_ant[n] - r), 0.0))
        quer[n] = pedido
        mods_exec[n] = v
    tot = sum(quer.values())
    fator = 1.0 if tot <= Z + 1e-9 else max(Z, 0.0) / tot
    falta_desp = max(tot - max(Z, 0.0), 0.0) if fator < 1.0 else 0.0
    entregue = {n: quer[n] * fator for n in quer}
    Z -= sum(entregue.values())
    recolhido_total = 0.0
    for n, v in mods_exec.items():
        r = rec_exec[n]
        recolhido_total += r
        pedidos_m = {m: x for m, x in v.items()}
        soma_p = sum(pedidos_m.values())
        for m, x in pedidos_m.items():
            parte = entregue[n] * (x / soma_p) if soma_p > 0 else 0.0
            rr = r if (m == "C" or "C" not in pedidos_m) else 0.0
            C(n, "conv" if m == "C" else "esp", (c.K_C if m == "C" else c.K_E) + c.kappa * (parte + rr))
        if n in rede.cap:
            est.I[n] = saldo_ant[n] - r + entregue[n]
        else:
            est.T[n] = saldo_ant[n] - r + entregue[n]
    if recolhido_total > 0:
        kp = k + c.L_P
        est.processamento[kp] = est.processamento.get(kp, 0.0) + recolhido_total
        C("CD", "num", c.c_P * recolhido_total)

    # 3) GO
    plano_go = est.go.pop(k, {})
    minutos = {b: 0.0 for b in rede.tesourarias}
    go_q, go_s, fila = {}, {}, {}
    inicio_atm = {}
    for b in rede.tesourarias:
        nvis = len(mods_exec.get(b, {}))
        if nvis:
            minutos[b] += rede.theta_cf * nvis
            C(b, "go", c.c_GO * rede.theta_cf * nvis)
        caixa = est.T[b]
        if cal.go[k]:
            ordem = [j for j in plano_go if rede.agencia_de.get(j) == b]
            for j in ordem:
                qv, sv = plano_go[j]
                if minutos[b] + rede.theta[j] > rede.Theta + 1e-9:
                    continue
                s_act = min(max(sv, 0.0), est.I[j]) if j in rede.recicladores else 0.0
                q_act = min(max(qv, 0.0), max(caixa + s_act, 0.0), max(rede.cap[j] - (est.I[j] - s_act), 0.0))
                if q_act + s_act <= 1e-6:
                    continue
                est.I[j] += q_act - s_act
                caixa += s_act - q_act
                minutos[b] += rede.theta[j]
                go_q[j], go_s[j] = q_act, s_act
                C(j, "go", c.c_GO * rede.theta[j])
            # fila do GO: precisava e não coube no limite de minutos
            for j in rede.internos[b]:
                if j in go_q:
                    continue
                need = (necessidade_go or {}).get(j)
                if need is not None and est.I[j] < need and minutos[b] + rede.theta[j] > rede.Theta + 1e-9:
                    fila[j] = True
        est.T[b] = caixa

    # 4) demanda
    for a in rede.atms:
        ini = est.I[a]
        inicio_atm[a] = ini
        g = G.get(a, 0.0) if a in rede.recicladores else 0.0
        fim = ini + g - D[a]
        u = max(-fim, 0.0)
        fim = max(fim, 0.0)
        w = max(fim - rede.cap[a], 0.0)
        fim = min(fim, rede.cap[a])
        est.I[a] = fim
        C(a, "rupt", c.p * u)
        C(a, "estouro", c.p_w * w)
        C(a, "opor", c.h * fim)
        if registrar:
            res.estados.append(dict(
                ponto=a, k=k, saldo_anterior=saldo_ant[a], saldo_ini=ini, saldo_fim=fim,
                chegada=entregue.get(a, 0.0) + go_q.get(a, 0.0), recolhimento=rec_exec.get(a, 0.0) + go_s.get(a, 0.0),
                abastecimento_go=go_q.get(a, 0.0), saque=D[a], deposito=g, nao_atendido=u, excesso=w,
                fila_go=bool(fila.get(a, False)), visita_C="C" in mods_exec.get(a, {}),
                visita_E="E" in mods_exec.get(a, {}), minutos_go=0.0, guiche=0.0))
    for b in rede.tesourarias:
        f = F.get(b, 0.0)
        tfim = est.T[b] + f
        uT = max(-tfim, 0.0)
        tfim = max(tfim, 0.0)
        o = max(tfim - rede.teto[b], 0.0)
        est.T[b] = tfim
        C(b, "rupt", c.p_T * uT)
        C(b, "estouro", c.p_o * o)
        C(b, "opor", c.h * tfim)
        if registrar:
            sai = sum(go_q.get(j, 0.0) - go_s.get(j, 0.0) for j in rede.internos[b])
            res.estados.append(dict(
                ponto=b, k=k, saldo_anterior=saldo_ant[b],
                saldo_ini=saldo_ant[b] - rec_exec.get(b, 0.0) + entregue.get(b, 0.0),
                saldo_fim=tfim, chegada=entregue.get(b, 0.0), recolhimento=rec_exec.get(b, 0.0),
                abastecimento_go=sai, saque=0.0, deposito=0.0, nao_atendido=uT, excesso=o,
                fila_go=any(fila.get(j, False) for j in rede.internos[b]),
                visita_C="C" in mods_exec.get(b, {}), visita_E="E" in mods_exec.get(b, {}),
                minutos_go=minutos[b], guiche=f))
    oCD = max(Z - rede.Zmax, 0.0)
    est.Z = Z
    C("CD", "estouro", c.p_o * oCD)
    C("CD", "opor", c.h * max(Z, 0.0))
    if registrar:
        res.estados.append(dict(
            ponto="CD", k=k, saldo_anterior=Z0, saldo_ini=Z0 + chegada_bc + proc - dev, saldo_fim=Z,
            chegada=chegada_bc, recolhimento=proc, abastecimento_go=0.0, saque=0.0, deposito=0.0,
            nao_atendido=falta_desp, excesso=oCD, fila_go=False, visita_C=False, visita_E=False,
            minutos_go=0.0, guiche=0.0, entrada_custodia=chegada_bc, recolhimento_processado=proc,
            despacho=sum(entregue.values()), devolucao=dev, falta_despachar=falta_desp))
        for (ponto, comp), v in custo.items():
            res.custos.append(dict(ponto=ponto, k=k, componente=comp, valor=v))
        # restrições ativas observadas (realizado)
        for b in rede.tesourarias:
            if cal.go[k]:
                folga_go = rede.Theta - minutos[b]
                ativa = folga_go < min(rede.theta[j] for j in rede.internos[b]) - 1e-9
                res.ativas.append(dict(ponto=b, k=k, restricao="R5", ativa=bool(ativa), folga=folga_go))
            res.ativas.append(dict(ponto=b, k=k, restricao="R7", ativa=bool(est.T[b] > rede.teto[b] + 1e-9),
                                   folga=rede.teto[b] - est.T[b]))
        res.ativas.append(dict(ponto="CD", k=k, restricao="R12", ativa=bool(oCD > 0), folga=rede.Zmax - Z))
    est.k = k
    return res


# ---------------------------------------------------------------------------
# Projeção com demanda mediana (plano das regras)
# ---------------------------------------------------------------------------
def executar_dia_projetado(est: EstadoRede, k: int, politica, E, cfg: Config, cal: CalendarioOperacional,
                           rede: RedePiloto) -> None:
    D = {a: max(E.soma([(a, k, 1.0)], 0.5), 0.0) for a in rede.atms}
    G = {a: max(E.soma([(f"G:{a}", k, 1.0)], 0.5), 0.0) for a in rede.recicladores}
    F = {b: E.soma([(f"F:{b}", k, 1.0)], 0.5) for b in rede.tesourarias}
    executar_dia(est, k, D, G, F, cfg, cal, rede, registrar=False)


# ---------------------------------------------------------------------------
# Laço do backtest
# ---------------------------------------------------------------------------
@dataclass
class ResultadoPolitica:
    nome: str
    estados: pd.DataFrame
    custos: pd.DataFrame
    decisoes: pd.DataFrame
    execucoes: pd.DataFrame
    modelo: pd.DataFrame


def estado_inicial(rede: RedePiloto, k: int) -> EstadoRede:
    return EstadoRede(k=k, I={a: float(rede.saldo_inicial[a]) for a in rede.atms},
                      T={b: float(rede.saldo_inicial[b]) for b in rede.tesourarias},
                      Z=float(rede.saldo_inicial["CD"]))


def _registros_decisao(nome: str, dec: Decisoes, cfg: Config, rede: RedePiloto) -> list[dict]:
    L = {"C": cfg.custos.L_C, "E": cfg.custos.L_E}
    k = dec.k0
    out = []
    for n, mods in dec.pedidos.items():
        for m, x in mods.items():
            out.append(dict(politica=nome, ponto=n, k_decisao=k, k_execucao=k + L[m], tipo=m, valor=x,
                            minutos=0.0, estagio=1, prob=1.0))
    for n, r in dec.recolher.items():
        out.append(dict(politica=nome, ponto=n, k_decisao=k, k_execucao=k + 1, tipo="REC", valor=r,
                        minutos=0.0, estagio=1, prob=1.0))
    for (n, ke), r in dec.recolher_futuro.items():
        out.append(dict(politica=nome, ponto=n, k_decisao=k, k_execucao=ke, tipo="REC", valor=r,
                        minutos=0.0, estagio=1, prob=1.0))
    for j, (qv, sv) in dec.go.items():
        out.append(dict(politica=nome, ponto=j, k_decisao=k, k_execucao=k + 1, tipo="GO", valor=qv,
                        minutos=rede.theta[j], estagio=1, prob=1.0))
        if sv > 0:
            out.append(dict(politica=nome, ponto=j, k_decisao=k, k_execucao=k + 1, tipo="RGO", valor=sv,
                            minutos=0.0, estagio=1, prob=1.0))
    if dec.bc_retirada > 0:
        out.append(dict(politica=nome, ponto="CD", k_decisao=k, k_execucao=None, tipo="BC", valor=dec.bc_retirada,
                        minutos=0.0, estagio=1, prob=1.0))
    if dec.bc_devolucao > 0:
        out.append(dict(politica=nome, ponto="CD", k_decisao=k, k_execucao=None, tipo="DEV",
                        valor=dec.bc_devolucao, minutos=0.0, estagio=1, prob=1.0))
    for d in dec.plano:
        if d.estagio == 2:
            out.append(dict(politica=nome, ponto=d.ponto, k_decisao=d.k_decisao, k_execucao=d.k_execucao,
                            tipo=d.tipo, valor=d.valor, minutos=d.minutos, estagio=2, prob=d.prob))
    return out


def rodar_politica(nome: str, politica, cfg: Config, rede: RedePiloto, cal: CalendarioOperacional,
                   demanda_real: dict, contexto_noite, k_ini: int, k_fim: int,
                   caminho_ckpt: str | None = None, log=print, a_cada: int = 7) -> ResultadoPolitica:
    """Roda a política nos dias k_ini..k_fim (inclusive), decidindo nas noites
    k_ini−1..k_fim−1. ``contexto_noite(k, estado)`` monta o ContextoNoite.

    Checkpoint por noite (pickle): se a sessão cair, retoma da última noite feita."""
    reg = {"estados": [], "custos": [], "decisoes": [], "execucoes": [], "modelo": []}
    est = estado_inicial(rede, k_ini - 1)
    k_next = k_ini - 1
    if caminho_ckpt and os.path.exists(caminho_ckpt):
        with open(caminho_ckpt, "rb") as fh:
            salvo = pickle.load(fh)
        if salvo.get("assinatura") == (nome, k_ini, k_fim):
            est, reg, k_next = salvo["estado"], salvo["reg"], salvo["k_next"]
            log(f"[{nome}] retomando da noite {cal.data(k_next).date()}")
    t0 = time.time()
    D_real, G_real, F_real, nec_go = demanda_real["D"], demanda_real["G"], demanda_real["F"], demanda_real.get("necessidade_go")
    for k in range(k_next, k_fim):
        ctx = contexto_noite(k, est)
        t1 = time.time()
        dec = politica.decidir(ctx)
        info_milp = dec.info.get("milp", {})
        if dec.info.get("fallback"):
            # MILP sem solução nesta noite: usa a regra P0 e registra
            dec = contexto_noite.fallback.decidir(ctx)
        reg["execucoes"].append(dict(
            politica=nome, k=k, segundos=time.time() - t1, status=info_milp.get("status", "regra"),
            gap=info_milp.get("gap"), objetivo=info_milp.get("objetivo"), variaveis=info_milp.get("variaveis"),
            binarias=info_milp.get("binarias"), restricoes=info_milp.get("restricoes"),
            cenarios=info_milp.get("cenarios"), partida_quente=info_milp.get("partida_quente"),
            fallback=bool(dec.info.get("fallback", False)) or info_milp.get("erro") is not None))
        for dd in dec.info.get("duais", []) or []:
            reg["modelo"].append(dict(politica=nome, k=k + 1, fonte="dual_lp", **dd))
        reg["decisoes"] += _registros_decisao(nome, dec, cfg, rede)
        aplicar_decisoes(est, dec, cfg, cal, rede)
        # executa o dia seguinte com a demanda real
        kd = k + 1
        D = {a: float(D_real[a][kd]) for a in rede.atms}
        G = {a: float(G_real[a][kd]) for a in G_real} if G_real else {}
        F = {b: float(F_real[b][kd]) for b in F_real} if F_real else {}
        nec = nec_go(kd) if nec_go else None
        rd = executar_dia(est, kd, D, G, F, cfg, cal, rede, necessidade_go=nec, registrar=True)
        reg["estados"] += [dict(politica=nome, **e) for e in rd.estados]
        reg["custos"] += [dict(politica=nome, **e) for e in rd.custos]
        reg["modelo"] += [dict(politica=nome, fonte="realizado", dual=None, **e) for e in rd.ativas]
        if caminho_ckpt and ((k - k_next + 1) % a_cada == 0 or k == k_fim - 1):
            os.makedirs(os.path.dirname(caminho_ckpt) or ".", exist_ok=True)
            tmp = caminho_ckpt + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump({"assinatura": (nome, k_ini, k_fim), "estado": est, "reg": reg, "k_next": k + 1}, fh)
            os.replace(tmp, caminho_ckpt)
        if (k - k_next + 1) % a_cada == 0:
            log(f"[{nome}] {cal.data(kd).date()} · {time.time() - t0:.0f} s")
    log(f"[{nome}] concluída em {time.time() - t0:.0f} s")

    def df(chave, datas=("k",)):
        d = pd.DataFrame(reg[chave])
        for col in datas:
            if col in d.columns:
                d[col.replace("k_", "data_") if col != "k" else "data"] = d[col].map(
                    lambda v: cal.data(int(v)) if v is not None and not pd.isna(v) else pd.NaT)
        return d

    dec_df = df("decisoes", ("k_decisao", "k_execucao"))
    if len(dec_df):
        # retirada/devolução: data de execução = chegada da remessa
        m = dec_df["k_execucao"].isna()
        dec_df.loc[m, "k_execucao"] = dec_df.loc[m, "k_decisao"].map(lambda kk: cal.chegada_bc(int(kk)))
        dec_df.loc[m, "data_execucao"] = dec_df.loc[m, "k_execucao"].map(lambda kk: cal.data(int(kk)))
    return ResultadoPolitica(nome, df("estados"), df("custos"), dec_df, df("execucoes"), df("modelo"))


__all__ = ["aplicar_decisoes", "executar_dia", "executar_dia_projetado", "rodar_politica", "ResultadoPolitica",
           "estado_inicial", "COMPONENTES"]

