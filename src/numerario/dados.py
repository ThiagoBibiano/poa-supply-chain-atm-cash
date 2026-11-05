"""Dados: NN5 (saques diários de ATMs ingleses, 1996–1998).

Fonte principal: registro 4656110 do Zenodo, "NN5 Daily Dataset (with Missing
Values)", 111 séries × 791 dias a partir de 18/03/1996 (Godahewa et al.,
Monash Time Series Forecasting Archive). O leitor ``.tsf`` oficial da Monash
importa ``distutils`` (fora do Python 3.12), então este módulo traz um leitor
próprio.

O que é real e o que é sintético (proposta, seção 7):

| Componente                 | Situação | Tratamento                                        |
|----------------------------|----------|---------------------------------------------------|
| Saques dos ATMs            | real     | 20 séries sorteadas com semente fixa              |
| Zeros e lacunas            | real     | indisponibilidade (demanda censurada): imputar    |
| Depósitos em recicladores  | não há   | sintético a partir de outras séries do NN5 (Exp. B) |
| Fluxo de guichê            | não há   | sintético a partir de outras séries do NN5 (Exp. B) |
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

URL_ZENODO = ("https://zenodo.org/records/4656110/files/"
              "nn5_daily_dataset_with_missing_values.zip?download=1")
MD5_ZENODO = "2529489455b2d9f730fbb3c65d39df88"
# Espelho sem faltantes (só se o Zenodo falhar; perde a marcação de lacunas)
URL_HF_SEM_FALTANTES = ("https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/"
                        "forecasting/monash/nn5_daily_dataset_without_missing_values.tsf")


# ---------------------------------------------------------------------------
# Leitura do .tsf
# ---------------------------------------------------------------------------
def ler_tsf(texto: str) -> pd.DataFrame:
    """Lê um arquivo .tsf da Monash (séries de mesmo comprimento) e devolve um
    DataFrame datas × séries, com NaN nos valores '?'."""
    atributos: list[str] = []
    dados_iniciados = False
    series: dict[str, pd.Series] = {}
    freq = "D"
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        if not dados_iniciados:
            low = linha.lower()
            if low.startswith("@attribute"):
                atributos.append(linha.split()[1])
            elif low.startswith("@frequency"):
                f = linha.split()[1].lower()
                freq = {"daily": "D", "weekly": "W"}.get(f, "D")
            elif low.startswith("@data"):
                dados_iniciados = True
            continue
        partes = linha.split(":")
        valores_txt = partes[-1]
        meta = partes[:-1]
        nome = meta[0]
        inicio = None
        for att, val in zip(atributos[1:], meta[1:]):
            if "timestamp" in att.lower() or "date" in att.lower():
                inicio = pd.to_datetime(val.replace(" 00-00-00", ""), format="%Y-%m-%d")
        if inicio is None:
            inicio = pd.Timestamp("1996-03-18")
        vals = np.array([np.nan if v.strip() == "?" else float(v) for v in valores_txt.split(",")])
        idx = pd.date_range(inicio, periods=len(vals), freq=freq)
        series[nome] = pd.Series(vals, index=idx)
    df = pd.DataFrame(series)
    df.index.name = "data"
    return df


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def baixar_nn5(cfg: Config, pasta_raw: str) -> tuple[pd.DataFrame, dict]:
    """Baixa (ou lê do disco) o NN5 e devolve (DataFrame, metadados da origem).

    Idempotente: se o .zip já estiver em ``pasta_raw``, não baixa de novo.
    """
    os.makedirs(pasta_raw, exist_ok=True)
    fonte = cfg.caminhos.dados_fonte
    meta: dict = {"fonte": fonte}
    if fonte == "sintetico":
        df = gerar_nn5_sintetico(semente=cfg.rede.semente_series)
        meta["aviso"] = "DADOS SINTÉTICOS no formato do NN5 (só para testes)"
        return df, meta
    if fonte == "arquivo":
        caminho = cfg.caminhos.arquivo_nn5
        if not caminho or not os.path.exists(caminho):
            raise FileNotFoundError(f"arquivo do NN5 não encontrado: {caminho}")
        b = open(caminho, "rb").read()
        meta.update(arquivo=caminho, md5=_md5(b))
        return _ler_bytes(b, caminho), meta
    if fonte == "zenodo":
        destino = os.path.join(pasta_raw, "nn5_daily_dataset_with_missing_values.zip")
        if not os.path.exists(destino):
            req = urllib.request.Request(URL_ZENODO, headers={"User-Agent": "poa-numerario/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            open(destino, "wb").write(b)
        b = open(destino, "rb").read()
        md5 = _md5(b)
        meta.update(arquivo=destino, md5=md5, md5_esperado=MD5_ZENODO, md5_ok=(md5 == MD5_ZENODO))
        if md5 != MD5_ZENODO:
            meta["aviso"] = "md5 diferente do registrado no Zenodo: conferir o arquivo"
        return _ler_bytes(b, destino), meta
    if fonte == "huggingface":
        destino = os.path.join(pasta_raw, "nn5_daily_dataset_without_missing_values.tsf")
        if not os.path.exists(destino):
            req = urllib.request.Request(URL_HF_SEM_FALTANTES, headers={"User-Agent": "poa-numerario/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                open(destino, "wb").write(r.read())
        b = open(destino, "rb").read()
        meta.update(arquivo=destino, md5=_md5(b),
                    aviso="versão SEM faltantes (lacunas já preenchidas pela Monash)")
        return _ler_bytes(b, destino), meta
    raise ValueError(f"dados_fonte desconhecida: {fonte}")


def _ler_bytes(b: bytes, nome: str) -> pd.DataFrame:
    if nome.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            tsf = [n for n in z.namelist() if n.endswith(".tsf")][0]
            texto = z.read(tsf).decode("utf-8", errors="replace")
    else:
        texto = b.decode("utf-8", errors="replace")
    return ler_tsf(texto)


# ---------------------------------------------------------------------------
# Dados sintéticos no formato do NN5 (testes locais, sem internet)
# ---------------------------------------------------------------------------
def gerar_nn5_sintetico(n_series: int = 111, n_dias: int = 791, inicio: str = "1996-03-18",
                        semente: int = 0, frac_faltantes: float = 0.01) -> pd.DataFrame:
    """Séries com sazonalidade semanal e anual, choque comum, feriados e lacunas.
    Servem apenas para testar o pipeline quando o NN5 não está acessível."""
    from .calendario import feriados_ew

    rng = np.random.default_rng(semente)
    datas = pd.date_range(inicio, periods=n_dias, freq="D")
    dow = datas.dayofweek.values
    perfil = np.array([0.95, 0.9, 0.95, 1.05, 1.45, 1.2, 0.6])  # seg..dom
    anual = 1 + 0.12 * np.sin(2 * np.pi * (datas.dayofyear.values - 80) / 365.25)
    natal = np.where((datas.month == 12) & (datas.day >= 15) & (datas.day <= 24), 1.35, 1.0)
    fer = datas.isin(feriados_ew())
    vespera = np.roll(fer, -1)
    comum = np.zeros(n_dias)
    for i in range(1, n_dias):  # choque comum AR(1)
        comum[i] = 0.6 * comum[i - 1] + rng.normal(0, 0.10)
    dados = {}
    for s in range(n_series):
        nivel = rng.uniform(12, 40)
        amp = rng.uniform(0.7, 1.3)
        prop = np.zeros(n_dias)
        for i in range(1, n_dias):
            prop[i] = 0.4 * prop[i - 1] + rng.normal(0, 0.18)
        p = perfil[dow] ** amp
        mu = nivel * p * anual * natal * np.where(fer, 0.55, 1.0) * np.where(vespera, 1.3, 1.0)
        y = mu * np.exp(comum + prop - 0.02)
        y = np.round(y, 2)
        mask = rng.random(n_dias) < frac_faltantes
        y[mask] = np.nan
        dados[f"T{s + 1}"] = y
    df = pd.DataFrame(dados, index=datas)
    df.index.name = "data"
    return df


# ---------------------------------------------------------------------------
# Imputação causal e seleção das séries
# ---------------------------------------------------------------------------
def imputar_causal(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Mediana do mesmo dia da semana nas 4 semanas anteriores a cada dia
    faltante (só passado). Zeros contam como indisponibilidade (demanda
    censurada). Devolve (série imputada, flag de indisponível)."""
    v = serie.values.astype(float).copy()
    indisp = np.isnan(v) | (v <= 0)
    out = v.copy()
    for i in np.where(indisp)[0]:
        viz = [out[i - 7 * w] for w in range(1, 5) if i - 7 * w >= 0 and not np.isnan(out[i - 7 * w])]
        if viz:
            out[i] = float(np.median(viz))
        else:
            prev = out[max(0, i - 7):i]
            prev = prev[~np.isnan(prev)]
            out[i] = float(np.mean(prev)) if len(prev) else np.nan
    # sobra no começo da série (sem passado): preenche para trás
    s = pd.Series(out, index=serie.index).bfill().ffill()
    return s, pd.Series(indisp, index=serie.index)


@dataclass
class DadosProjeto:
    """Tudo o que o resto do pacote precisa sobre a demanda."""

    bruto: pd.DataFrame                 # NN5 inteiro (datas × 111), NaN nos faltantes
    saques: pd.DataFrame                # ATMs do projeto (datas × ids), R$ mil, imputado
    indisponivel: pd.DataFrame          # flags de indisponibilidade
    saques_bruto: pd.DataFrame          # mesmas séries em R$ mil, com NaN (contexto do Chronos)
    mapa_series: pd.DataFrame           # id do ATM → série do NN5, tipo, agência
    escala: float                       # unidade NN5 → R$ mil
    depositos: pd.DataFrame | None = None   # Exp. B (datas × ids de recicladores)
    guiche: pd.DataFrame | None = None      # Exp. B (datas × ids de tesouraria)
    meta: dict = field(default_factory=dict)

    def hash(self) -> str:
        h = hashlib.sha256(pd.util.hash_pandas_object(self.saques, index=True).values.tobytes())
        return h.hexdigest()[:12]


def selecionar_series(bruto: pd.DataFrame, cfg: Config) -> list[str]:
    """Sorteia as séries do projeto com semente fixa, entre as que têm poucas
    lacunas/zeros na janela calibração + aquecimento + teste."""
    r = cfg.rede
    n = r.n_agencias * r.atms_internos_por_agencia + r.n_atms_externos
    jan = bruto.loc[cfg.datas.calibracao_inicio:cfg.datas.teste_fim]
    frac = (jan.isna() | (jan <= 0)).mean()
    elegiveis = [c for c in bruto.columns if frac[c] <= r.max_faltantes_frac]
    if len(elegiveis) < n:  # poucas séries limpas: completa com as de menos lacunas
        elegiveis = list(frac.sort_values(kind="stable").index[:n])
    rng = np.random.default_rng(r.semente_series)
    return list(rng.choice(elegiveis, size=n, replace=False))


def preparar_dados(bruto: pd.DataFrame, cfg: Config, meta_fonte: dict | None = None) -> DadosProjeto:
    """Seleciona, imputa, escala para R$ mil e monta os sintéticos do Exp. B."""
    r = cfg.rede
    escolhidas = selecionar_series(bruto, cfg)
    ids, tipos, agencias, series = [], [], [], []
    i = 0
    for b in range(1, r.n_agencias + 1):
        for j in range(1, r.atms_internos_por_agencia + 1):
            ids.append(f"A{b}.{j}"); tipos.append("int"); agencias.append(b); series.append(escolhidas[i]); i += 1
    for k in range(1, r.n_atms_externos + 1):
        ids.append(f"E{k}"); tipos.append("ext"); agencias.append(0); series.append(escolhidas[i]); i += 1
    mapa = pd.DataFrame({"ponto": ids, "tipo": tipos, "agencia": agencias, "serie_nn5": series})

    imput, indisp = {}, {}
    for pid, s in zip(ids, series):
        si, fi = imputar_causal(bruto[s])
        imput[pid], indisp[pid] = si, fi
    saques_u = pd.DataFrame(imput)
    hist = saques_u.loc[:cfg.datas.calibracao_fim]
    escala = r.saque_medio_atm_rs_mil / float(hist.values.mean())
    saques = saques_u * escala
    saques_bruto = pd.DataFrame({pid: bruto[s] * escala for pid, s in zip(ids, series)})
    saques_bruto = saques_bruto.where(saques_bruto > 0)

    dep = gui = None
    if cfg.experimento.nome.upper() == "B":
        dep, gui = _sinteticos_exp_b(bruto, escolhidas, saques, cfg)

    meta = dict(meta_fonte or {})
    meta.update(series=escolhidas, escala_rs_mil_por_unidade=escala,
                frac_indisponivel=float(pd.DataFrame(indisp).loc[cfg.datas.aquecimento_inicio:cfg.datas.teste_fim].values.mean()))
    return DadosProjeto(bruto=bruto, saques=saques, indisponivel=pd.DataFrame(indisp),
                        saques_bruto=saques_bruto, mapa_series=mapa, escala=escala,
                        depositos=dep, guiche=gui, meta=meta)


def _sinteticos_exp_b(bruto: pd.DataFrame, usadas: list[str], saques: pd.DataFrame,
                      cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp. B: depósitos G_a = α_a × outra série do NN5 (recicladores) e fluxo
    líquido de guichê F_b = μ_b + σ·z, com z de outra série, só em dia útil."""
    from .calendario import feriados_ew

    ex = cfg.experimento
    rng = np.random.default_rng(cfg.rede.semente_series + 1)
    livres = [c for c in bruto.columns if c not in set(usadas)]
    rng.shuffle(livres)
    atms = list(saques.columns)
    n_rec = int(round(ex.frac_recicladores * len(atms)))
    rec = [a for a in atms if a in set(rng.choice(atms, size=n_rec, replace=False))]
    dep = {}
    for a in rec:
        s = livres.pop()
        si, _ = imputar_causal(bruto[s])
        base = si / si.loc[:cfg.datas.calibracao_fim].mean()
        dep[a] = base * ex.razao_deposito_saque * saques[a].loc[:cfg.datas.calibracao_fim].mean()
    gui = {}
    datas = saques.index
    util = (datas.dayofweek < 5) & ~datas.isin(feriados_ew())
    for b in range(1, cfg.rede.n_agencias + 1):
        s = livres.pop()
        si, _ = imputar_causal(bruto[s])
        z = (np.log(si) - np.log(si).rolling(28, min_periods=7).mean()).fillna(0.0)
        z = z / (z.std() or 1.0)
        mu = ex.guiche_medio_rs_mil[(b - 1) % len(ex.guiche_medio_rs_mil)]
        gui[f"T{b}"] = pd.Series(np.where(util, mu + ex.guiche_desvio_rs_mil * z.values, 0.0), index=datas)
    return pd.DataFrame(dep), pd.DataFrame(gui)


__all__ = ["ler_tsf", "baixar_nn5", "gerar_nn5_sintetico", "imputar_causal",
           "selecionar_series", "preparar_dados", "DadosProjeto", "URL_ZENODO", "MD5_ZENODO"]
