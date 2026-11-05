import numpy as np
import pandas as pd

from numerario.calendario import conferir_feriados
from numerario.dados import imputar_causal, ler_tsf


def test_feriados_batem_com_pacote_holidays():
    dif = conferir_feriados()
    assert dif == [] or dif == ["pacote holidays indisponível: conferência não feita"]


def test_periodo_protecao_agencia_quinta_cobre_fim_de_semana(base):
    cfg, _, _, cal = base
    qui = cal.k("1998-03-26")          # quinta-feira comum
    P = cal.periodo_protecao_agencia(qui)
    assert [cal.data(x).day_name() for x in P] == ["Friday", "Saturday", "Sunday"]
    seg = cal.k("1998-03-23")          # segunda: dois dias úteis (L_C = 2)
    assert len(cal.periodo_protecao_agencia(seg)) == 2


def test_periodo_protecao_central_e_chegada_da_custodia(base):
    _, _, _, cal = base
    qui = cal.k("1998-03-26")
    assert [cal.data(x).day_name() for x in cal.periodo_protecao_central(qui)] == ["Friday", "Saturday", "Sunday"]
    sex = cal.k("1998-03-27")
    assert cal.data(cal.chegada_bc(sex)).day_name() == "Monday"   # L_BC = 1 dia útil
    sab = cal.k("1998-03-28")
    assert not cal.pode_pedir_bc(sab)                               # R13: custódia fechada


def test_feriado_bloqueia_go_e_custodia(base):
    _, _, _, cal = base
    sexta_santa = cal.k("1998-04-10")
    assert not cal.go[sexta_santa] and not cal.bc[sexta_santa] and not cal.transp[sexta_santa]


def test_ler_tsf():
    txt = """@relation teste
@attribute series_name string
@attribute start_timestamp date
@frequency daily
@missing true
@data
T1:1996-03-18 00-00-00:1.5,2,?,4
T2:1996-03-18 00-00-00:3,?,5,6
"""
    df = ler_tsf(txt)
    assert df.shape == (4, 2)
    assert df.index[0] == pd.Timestamp("1996-03-18")
    assert np.isnan(df.loc["1996-03-20", "T1"]) and df.loc["1996-03-21", "T2"] == 6


def test_imputacao_causal_usa_so_o_passado():
    idx = pd.date_range("1998-01-01", periods=35, freq="D")
    s = pd.Series(np.arange(35, dtype=float) + 10, index=idx)
    s.iloc[30] = np.nan
    s.iloc[31] = 0.0                     # zero = indisponível
    imp, flag = imputar_causal(s)
    assert flag.iloc[30] and flag.iloc[31]
    assert imp.iloc[30] == np.median([s.iloc[30 - 7 * w] for w in range(1, 5)])
