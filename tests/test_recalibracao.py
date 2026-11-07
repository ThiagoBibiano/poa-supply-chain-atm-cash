import numpy as np
import pandas as pd
from scipy import stats

from numerario.config import Config
from numerario.previsao import ConjuntoPrevisoes, ajustar_recalibracao

NIV = Config().previsao.niveis


def _conjunto_estreito(n_origens=400, H=3, semente=0):
    """Demanda real ~ N(100, 20); a previsão acerta a cauda de baixo e tem a de cima
    fina demais (σ = 12 acima da mediana) — o padrão observado no Chronos-2."""
    rng = np.random.default_rng(semente)
    series = ["a", "b", "G:a"]
    T = n_origens + H + 1
    y = rng.normal(100, 20, size=(T, len(series)))
    z = stats.norm.ppf(NIV)
    q = 100 + np.where(z < 0, 20 * z, 12 * z)                       # (Q,)
    por = {k: np.broadcast_to(q, (len(series), H, len(NIV))).astype(np.float32) for k in range(n_origens)}
    conj = ConjuntoPrevisoes("teste", series, list(NIV), H, por)
    return conj, pd.DataFrame(y, columns=series)


def _cobertura(conj, ctx, origens):
    vals = ctx[conj.series].values
    acertos = np.zeros(len(NIV)); n = 0
    for k in origens:
        for h in range(1, conj.H + 1):
            yy = vals[k + h]
            acertos += (yy[:, None] <= conj.quantis(k)[:, h - 1, :]).sum(axis=0)
            n += len(yy)
    return acertos / n


def test_quantis_sempre_float64():
    """O Chronos-2 devolve float32; a soma de centenas de σ em float32 errava ~1e-4 e
    derrubava a conferência do ganho de centralização com ρ = 1."""
    conj, _ = _conjunto_estreito(n_origens=3)
    assert all(v.dtype == np.float64 for v in conj.por_origem.values())
    rng = np.random.default_rng(0)
    z = 1.6448536269514722
    grupos = [rng.uniform(5, 25, size=n) for n in [27] * 4 + [3] * 8]

    def em(s, r):
        return z * np.sqrt((1 - r) * (s ** 2).sum() + r * s.sum() ** 2)
    g1 = sum(em(g, 1.0) for g in grupos) - em(np.concatenate(grupos), 1.0)
    assert abs(g1) <= 1e-9 * sum(em(g, 1.0) for g in grupos)


def test_recalibracao_corrige_a_cauda_fora_da_amostra():
    conj, ctx = _conjunto_estreito()
    ajuste, avaliacao = list(range(0, 200)), list(range(203, 400))
    rec = ajustar_recalibracao(conj, ctx, ajuste, ajuste[-1] + conj.H)
    novo = rec.aplicar(conj, np.array([True, True, True]))
    niv = np.asarray(NIV)
    antes, depois = _cobertura(conj, ctx, avaliacao), _cobertura(novo, ctx, avaliacao)
    alta = niv >= 0.90
    assert np.all(antes[alta] < niv[alta] - 0.02)                       # cauda de cima fina antes
    assert np.all(np.abs(depois[alta] - niv[alta]) < 0.025)             # e calibrada depois
    baixa = niv <= 0.10
    assert np.all(np.abs(depois[baixa] - niv[baixa]) < 0.025)           # a de baixo continua boa
    # grupos com mapas próprios, quantis monotônicos e não negativos
    assert set(rec.mapa) == {"saque", "G"}
    for q in novo.por_origem.values():
        assert np.all(np.diff(q, axis=-1) >= 0) and np.all(q >= 0)
    assert rec.resumo()["q95"] > 0.95


def test_recalibracao_nao_muda_a_chave_do_cache_das_previsoes():
    """Ligar a recalibração não pode refazer o Chronos-2 (a chave das previsões brutas ignora a opção)."""
    cfg = Config()
    ligada = cfg.copia(**{"previsao.recalibrar_quantis": True})
    chave = lambda c: c.copia(**{"previsao.recalibrar_quantis": False}).hash("previsao")  # noqa: E731
    assert chave(ligada) == chave(cfg) == cfg.hash("previsao")
    # já os cenários e o backtest dependem dela
    assert ligada.hash("previsao", "cenarios") != cfg.hash("previsao", "cenarios")
