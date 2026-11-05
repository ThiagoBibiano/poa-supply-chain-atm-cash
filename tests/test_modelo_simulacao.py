import numpy as np

from numerario.estado import Decisoes, EstadoRede
from numerario.modelo import InstanciaMILP, ModeloNumerario
from numerario.simulacao import aplicar_decisoes, executar_dia


def _estado(rede, k, fator=1.0):
    return EstadoRede(k=k, I={a: rede.saldo_inicial[a] * fator for a in rede.atms},
                      T={b: rede.saldo_inicial[b] for b in rede.tesourarias}, Z=rede.saldo_inicial["CD"])


def _instancia(dados, rede, k0, S, est, rng, ruido=0.2):
    base = dados.saques[rede.atms].values.T[:, k0 + 1:k0 + 11]
    D = np.stack([base * np.exp(rng.normal(0, ruido, base.shape)) for _ in range(S)])
    EM = {b: [rede.Rb[b]] * 10 for b in rede.tesourarias}
    return InstanciaMILP(k0=k0, H=10, estado=est, D=D, probs=np.full(S, 1 / S), EM=EM, EMCD=[rede.Rc] * 10)


def test_milp_deterministico_respeita_go_e_capacidades(base):
    cfg, dados, rede, cal = base
    cfg = cfg.copia(**{"solver.tempo_limite_s": 20, "solver.calcular_duais": True})
    k0 = cal.k("1998-03-12")           # quinta à noite: decide sexta
    est = _estado(rede, k0, fator=0.4)  # ATMs baixos forçam abastecimento
    sol = ModeloNumerario(cfg, rede, cal).resolver(_instancia(dados, rede, k0, 1, est, np.random.default_rng(0), 0))
    assert sol.stats["ok"]
    d = sol.decisoes
    assert d.go, "com saldos baixos o GO precisa abastecer"
    for b in rede.tesourarias:
        minutos = sum(rede.theta[j] for j in rede.internos[b] if j in d.go)
        minutos += rede.theta_cf * ("E" in d.pedidos.get(b, {}))
        assert minutos <= rede.Theta + 1e-6                                  # R5
    for j, (q, s) in d.go.items():
        assert est.I[j] + q - s <= rede.cap[j] + 1e-6                        # R7b
    # R4: o GO só usa o que está no cofre
    for b in rede.tesourarias:
        q_tot = sum(d.go.get(j, (0, 0))[0] for j in rede.internos[b])
        assert q_tot <= est.T[b] + d.pedidos.get(b, {}).get("E", 0.0) - d.recolher.get(b, 0.0) + 1e-6
    # dual de R5 com as binárias fixadas é zero por construção (só binárias na linha)
    r5 = [x for x in sol.duais if x["restricao"] == "R5"]
    assert all(abs(x["dual"]) < 1e-9 for x in r5)


def test_milp_estocastico_compartilha_o_primeiro_estagio(base):
    cfg, dados, rede, cal = base
    cfg = cfg.copia(**{"solver.tempo_limite_s": 30})
    k0 = cal.k("1998-03-09")
    est = _estado(rede, k0)
    mod = ModeloNumerario(cfg, rede, cal)
    inst = _instancia(dados, rede, k0, 5, est, np.random.default_rng(1))
    det = mod.resolver(_instancia(dados, rede, k0, 1, est, np.random.default_rng(1), 0))
    sol = mod.resolver(inst, inicial=det.papeis)
    assert sol.stats["ok"] and sol.stats["cenarios"] == 5
    # no plano, o 1º estágio aparece com probabilidade 1 (mesma decisão em todos os cenários)
    primeiro = [x for x in sol.decisoes.plano if x.estagio == 1 and x.tipo in ("C", "E", "GO")]
    assert all(abs(x.prob - 1.0) < 1e-6 for x in primeiro)
    # especiais só como recurso: nenhuma pedida a partir de τo = 2
    assert not [x for x in sol.decisoes.plano if x.tipo == "E" and x.k_execucao - k0 > 2]


def test_simulador_fecha_balancos_e_raciona_a_central(base):
    cfg, dados, rede, cal = base
    k = cal.k("1998-03-10")
    est = _estado(rede, k - 1)
    est.Z = 50.0                                    # central quase vazia
    dec = Decisoes(k0=k - 1)
    for kk in rede.externos:
        dec.pedidos[kk] = {"E": 100.0}              # especiais para amanhã
    aplicar_decisoes(est, dec, cfg, cal, rede)
    D = {a: float(dados.saques[a].values[k]) for a in rede.atms}
    res = executar_dia(est, k, D, {}, {}, cfg, cal, rede)
    cd = [e for e in res.estados if e["ponto"] == "CD"][0]
    assert cd["despacho"] <= 50.0 + 1e-9 and cd["falta_despachar"] > 0      # rateio
    for e in res.estados:
        if e["ponto"] in rede.atms:
            assert abs(e["saldo_ini"] + e["deposito"] - e["saque"] + e["nao_atendido"] - e["excesso"] - e["saldo_fim"]) < 1e-9
            assert e["saldo_fim"] <= rede.cap[e["ponto"]] + 1e-9
    custos = {(c["ponto"], c["componente"]): c["valor"] for c in res.custos}
    assert custos[("E1", "esp")] >= cfg.custos.K_E          # a especial foi cobrada
    assert ("CD", "num") not in custos                       # sem custódia nem recolhimento no dia
