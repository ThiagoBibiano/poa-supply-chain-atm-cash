import numpy as np
from scipy import stats

from numerario.cenarios import (Dependencia, avaliar_quantil, estimar_dependencia, gerar_trajetorias, pit,
                                reduzir_cenarios)
from numerario.config import NIVEIS_QUANTIS, z_alpha

NIV = np.array(NIVEIS_QUANTIS)


def _quantis_normais(mu, sd, shape):
    return np.broadcast_to(mu + sd * stats.norm.ppf(NIV), shape + (len(NIV),)).copy()


def test_funcao_quantil_reproduz_nos_e_e_monotona():
    q = _quantis_normais(100.0, 10.0, (1, 1))
    u = NIV.reshape(-1, 1, 1)
    v = avaliar_quantil(q, NIV, u)
    assert np.allclose(v[:, 0, 0], q[0, 0], atol=1e-8)
    uu = np.linspace(0.001, 0.999, 200).reshape(-1, 1, 1)
    vv = avaliar_quantil(q, NIV, uu)[:, 0, 0]
    assert np.all(np.diff(vv) >= -1e-9)


def test_pit_inverte_a_funcao_quantil():
    q = _quantis_normais(50.0, 5.0, (3, 2))
    y = np.array([[45.0, 50.0], [55.0, 60.0], [40.0, 52.0]])
    u = pit(q, NIV, y)
    back = avaliar_quantil(q, NIV, u[None])[0]
    assert np.allclose(back, y, atol=1e-6)


def test_trajetorias_respeitam_marginais_e_piso():
    rng = np.random.default_rng(0)
    q = _quantis_normais(20.0, 15.0, (2, 4))
    traj = gerar_trajetorias(q, NIV, Dependencia.independente(2, 4), 20000, rng, piso=np.array([True, False]))
    assert traj[:, 0].min() >= 0.0                 # saque não fica negativo
    assert traj[:, 1].min() < 0.0                  # série com sinal (guichê) pode
    assert abs(np.median(traj[:, 1, 0]) - 20.0) < 0.5


def test_dependencia_estimada_recupera_correlacao():
    rng = np.random.default_rng(1)
    rho = 0.6
    C = np.array([[1, rho], [rho, 1]])
    Z = rng.multivariate_normal([0, 0], C, size=400)
    arr = np.stack([Z, Z], axis=-1)                # (origens, séries, H=2)
    dep = estimar_dependencia(arr, shrinkage=0.0)
    assert abs(dep.C_series[0, 1] - rho) < 0.1


def test_reducao_preserva_as_duas_caudas():
    rng = np.random.default_rng(2)
    traj = rng.lognormal(3, 0.3, size=(1000, 5, 4))
    cen, prob, info = reduzir_cenarios(traj, 30, rng, frac_cauda=0.05, por_cauda=3, dias_cauda=3)
    assert cen.shape == (30, 5, 4) and abs(prob.sum() - 1) < 1e-12
    chave = traj[:, :, :3].sum(axis=(1, 2))
    ck = cen[:, :, :3].sum(axis=(1, 2))
    baixos = [ck[i] for i, g in enumerate(info["grupos"]) if g == "cauda_baixa"]
    altos = [ck[i] for i, g in enumerate(info["grupos"]) if g == "cauda_alta"]
    assert max(baixos) <= np.quantile(chave, 0.05) + 1e-9
    assert min(altos) >= np.quantile(chave, 0.95) - 1e-9
    assert abs(sum(p for p, g in zip(prob, info["grupos"]) if g == "cauda_alta") - 0.05) < 0.002


def test_aproximacao_do_estoque_minimo_bate_com_normal():
    """Soma de n normais com correlação ρ: Q_α − Q_0,5 = z_α √((1−ρ)Σσ² + ρ(Σσ)²)."""
    rng = np.random.default_rng(3)
    n, sig, rho, a = 6, 10.0, 0.3, 0.95
    C = np.full((n, n), rho) + (1 - rho) * np.eye(n)
    X = rng.multivariate_normal(np.zeros(n), C * sig ** 2, size=200000).sum(axis=1)
    emp = np.quantile(X, a) - np.quantile(X, 0.5)
    formula = z_alpha(a) * np.sqrt((1 - rho) * n * sig ** 2 + rho * (n * sig) ** 2)
    assert abs(emp - formula) / formula < 0.02
