"""Geração de cenários a partir dos quantis marginais (seção 6 da proposta).

1. Função quantil por (série, dia): interpolação monotônica entre os 21 quantis,
   linear na escala normal (z = Φ⁻¹(τ)), com extrapolação de cauda além de 0,01
   e 0,99; abaixo de zero a demanda é truncada (saque negativo não existe).
2. Dependência: PITs da janela de calibração, z = Φ⁻¹(u), correlação entre ATMs
   e entre horizontes com encolhimento (estrutura de Kronecker:
   corr((a,h),(a',h')) = C_série[a,a'] · C_horizonte[h,h']). Amostra-se
   Z ~ N(0, Σ), U = Φ(Z) e d = Q̂(U) — cópula gaussiana (ou t, como checagem).
3. Redução: ~1.000 trajetórias → 30 cenários por k-medoides, preservando as
   duas caudas: os extremos de demanda alta e baixa no período de proteção
   entram como grupos próprios, com a sua probabilidade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .config import Config


# ---------------------------------------------------------------------------
# Função quantil e PIT
# ---------------------------------------------------------------------------
def _z_nos(niveis: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.asarray(niveis, dtype=float))


def avaliar_quantil(q: np.ndarray, niveis, u: np.ndarray, piso: np.ndarray | None = None) -> np.ndarray:
    """Q̂(u) para quantis ``q`` (..., Q) e probabilidades ``u`` (N, ...).

    Interpolação linear em z = Φ⁻¹(τ) entre os nós; fora deles, extrapolação
    linear com a inclinação do último segmento (cauda aproximadamente normal).
    ``piso``: máscara (...) das séries truncadas em zero."""
    zk = _z_nos(niveis)
    zu = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    i = np.clip(np.searchsorted(zk, zu), 1, len(zk) - 1)          # (N, ...)
    qb = np.broadcast_to(q, zu.shape + (q.shape[-1],))
    q0 = np.take_along_axis(qb, (i - 1)[..., None], axis=-1)[..., 0]
    q1 = np.take_along_axis(qb, i[..., None], axis=-1)[..., 0]
    z0, z1 = zk[i - 1], zk[i]
    val = q0 + (zu - z0) * (q1 - q0) / (z1 - z0)
    if piso is not None:
        val = np.where(np.broadcast_to(piso, val.shape), np.maximum(val, 0.0), val)
    return val


def pit(q: np.ndarray, niveis, y: np.ndarray) -> np.ndarray:
    """u = F̂(y) para quantis ``q`` (..., Q) e observações ``y`` (...)."""
    zk = _z_nos(niveis)
    qs = np.sort(q, axis=-1)
    out = np.full(y.shape, np.nan)
    flat_q = qs.reshape(-1, qs.shape[-1])
    flat_y = y.reshape(-1)
    res = out.reshape(-1)
    for idx in range(flat_y.shape[0]):
        yy = flat_y[idx]
        if np.isnan(yy):
            continue
        qq = flat_q[idx]
        if np.any(np.isnan(qq)):
            continue
        # evita nós repetidos
        qq = qq + np.arange(len(qq)) * 1e-9
        j = int(np.clip(np.searchsorted(qq, yy), 1, len(qq) - 1))
        z = zk[j - 1] + (yy - qq[j - 1]) * (zk[j] - zk[j - 1]) / (qq[j] - qq[j - 1])
        res[idx] = stats.norm.cdf(z)
    return res.reshape(y.shape)


# ---------------------------------------------------------------------------
# Dependência (estimada na calibração)
# ---------------------------------------------------------------------------
@dataclass
class Dependencia:
    C_series: np.ndarray        # (n, n)
    C_h: np.ndarray             # (H, H)
    rho_medio: float            # correlação média entre pares (série, dia) — usada nas aproximações
    n_origens: int
    shrinkage: float

    @classmethod
    def independente(cls, n: int, H: int) -> "Dependencia":
        return cls(np.eye(n), np.eye(H), 0.0, 0, 1.0)

    def chol(self) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.cholesky(_psd(self.C_series)), np.linalg.cholesky(_psd(self.C_h))

    def corr_completa(self) -> np.ndarray:
        return np.kron(self.C_series, self.C_h)


def _psd(C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.maximum(w, eps)
    C2 = (V * w) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


def _corr_pares(X: np.ndarray) -> np.ndarray:
    """Correlação com pares completos; X: (amostras, variáveis)."""
    n = X.shape[1]
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            if m.sum() >= 8:
                a, b = X[m, i], X[m, j]
                if a.std() > 0 and b.std() > 0:
                    C[i, j] = C[j, i] = float(np.corrcoef(a, b)[0, 1])
    return C


def estimar_dependencia(Z: np.ndarray, shrinkage: float = 0.2, H_rho: int = 3) -> Dependencia:
    """``Z``: (origens, séries, H) com z = Φ⁻¹(PIT) (NaN onde não há real).

    C_série: média, sobre h, da correlação entre séries no mesmo passo.
    C_h: média, sobre as séries, da correlação entre passos da mesma série.
    Ambas encolhidas em direção à identidade e projetadas em PSD."""
    no, n, H = Z.shape
    Cs = np.zeros((n, n)); cnt = 0
    for h in range(H):
        Cs += _corr_pares(Z[:, :, h]); cnt += 1
    Cs /= max(cnt, 1)
    Ch = np.zeros((H, H)); cnt = 0
    for s in range(n):
        Ch += _corr_pares(Z[:, s, :]); cnt += 1
    Ch /= max(cnt, 1)
    Cs = _psd((1 - shrinkage) * Cs + shrinkage * np.eye(n))
    Ch = _psd((1 - shrinkage) * Ch + shrinkage * np.eye(H))
    Hr = min(H_rho, H)
    full = np.kron(Cs, Ch[:Hr, :Hr])
    m = full.shape[0]
    rho = float((full.sum() - np.trace(full)) / (m * (m - 1))) if m > 1 else 0.0
    return Dependencia(Cs, Ch, rho, no, shrinkage)


# ---------------------------------------------------------------------------
# Trajetórias e redução
# ---------------------------------------------------------------------------
def gerar_trajetorias(q: np.ndarray, niveis, dep: Dependencia, n: int, rng: np.random.Generator,
                      copula: str = "gaussiana", nu: float = 5.0,
                      piso: np.ndarray | None = None) -> np.ndarray:
    """q: (séries, H, Q) → trajetórias (n, séries, H)."""
    ns, H, _ = q.shape
    Ls, Lh = dep.chol()
    eps = rng.standard_normal((n, ns, H))
    Z = np.einsum("ij,njk,lk->nil", Ls, eps, Lh)
    if copula == "t":
        W = rng.chisquare(nu, size=(n, 1, 1))
        U = stats.t.cdf(Z / np.sqrt(W / nu), df=nu)
    else:
        U = stats.norm.cdf(Z)
    pisos = None if piso is None else piso[:, None]
    return avaliar_quantil(q, niveis, U, pisos)


def _kmedoides(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """K-medoides simples (iteração de Voronoi). Devolve (índices dos medoides, rótulos)."""
    n = X.shape[0]
    if k >= n:
        return np.arange(n), np.arange(n)
    sq = (X ** 2).sum(axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0))
    # inicialização k-means++ determinística pela semente
    med = [int(rng.integers(n))]
    for _ in range(1, k):
        dmin = D[:, med].min(axis=1)
        p = dmin ** 2
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1 / n)
        med.append(int(rng.choice(n, p=p)))
    med = np.array(med)
    for _ in range(iters):
        rot = np.argmin(D[:, med], axis=1)
        novo = med.copy()
        for c in range(k):
            membros = np.where(rot == c)[0]
            if len(membros):
                novo[c] = membros[np.argmin(D[np.ix_(membros, membros)].sum(axis=1))]
        if np.array_equal(novo, med):
            break
        med = novo
    rot = np.argmin(D[:, med], axis=1)
    return med, rot


def reduzir_cenarios(traj: np.ndarray, S: int, rng: np.random.Generator, frac_cauda: float = 0.05,
                     por_cauda: int = 3, dias_cauda: int = 3, chave_series: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """traj: (N, séries, H) → (cenários (S, séries, H), probabilidades (S,), info).

    As trajetórias são ordenadas pela demanda total da região nos primeiros
    ``dias_cauda`` dias; os ``frac_cauda`` de baixo e de cima viram dois grupos
    com ``por_cauda`` representantes cada, e o miolo recebe o restante."""
    N = traj.shape[0]
    sel = slice(None) if chave_series is None else chave_series
    chave = traj[:, sel, :dias_cauda].sum(axis=(1, 2))
    ordem = np.argsort(chave, kind="stable")
    n_c = max(por_cauda, int(round(frac_cauda * N)))
    if S < 2 * por_cauda + 1 or N < 2 * n_c + S:
        por_cauda, n_c = 0, 0
    baixo, cima, miolo = ordem[:n_c], ordem[N - n_c:] if n_c else ordem[:0], ordem[n_c:N - n_c]
    X = traj.reshape(N, -1)
    sd = X.std(axis=0); sd[sd == 0] = 1.0
    Xs = X / sd
    cen, prob, grupo = [], [], []

    def _grupo(ids, k, nome):
        if len(ids) == 0 or k == 0:
            return
        med, rot = _kmedoides(Xs[ids], k, rng)
        for c, m in enumerate(med):
            cen.append(traj[ids[m]])
            prob.append((rot == c).sum() / N)
            grupo.append(nome)

    _grupo(baixo, por_cauda, "cauda_baixa")
    _grupo(miolo, S - 2 * por_cauda, "miolo")
    _grupo(cima, por_cauda, "cauda_alta")
    prob = np.asarray(prob)
    prob = prob / prob.sum()
    info = {"grupos": grupo, "n_cauda": int(n_c), "chave_limites": [float(chave[ordem[0]]), float(chave[ordem[-1]])]}
    return np.stack(cen), prob, info


@dataclass
class CenariosOrigem:
    """Tudo o que uma noite de decisão precisa sobre a demanda futura."""

    origem_k: int
    series: list[str]
    quantis: np.ndarray        # (séries, H, Q)
    trajetorias: np.ndarray    # (N, séries, H) — para os estoques de segurança
    cenarios: np.ndarray       # (S, séries, H) — para o MILP estocástico
    probs: np.ndarray          # (S,)
    mediana: np.ndarray        # (séries, H) — cenário único da P1
    info: dict


def gerar_cenarios_origem(cfg: Config, origem_k: int, series: list[str], q: np.ndarray, niveis,
                          dep: Dependencia, piso: np.ndarray, chave_series: np.ndarray) -> CenariosOrigem:
    c = cfg.cenarios
    rng = np.random.default_rng(c.semente * 100_003 + origem_k)
    traj = gerar_trajetorias(q, niveis, dep, c.n_trajetorias, rng, c.copula, c.nu_t, piso)
    cen, prob, info = reduzir_cenarios(traj, c.n_cenarios, rng, c.frac_cauda, c.cenarios_por_cauda,
                                       c.dias_cauda, chave_series)
    i50 = int(np.argmin(np.abs(np.asarray(niveis) - 0.5)))
    return CenariosOrigem(origem_k, series, q, traj, cen, prob, q[:, :, i50], info)


__all__ = ["avaliar_quantil", "pit", "Dependencia", "estimar_dependencia", "gerar_trajetorias",
           "reduzir_cenarios", "CenariosOrigem", "gerar_cenarios_origem"]
