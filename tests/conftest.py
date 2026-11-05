import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from numerario.calendario import CalendarioOperacional  # noqa: E402
from numerario.config import Config  # noqa: E402
from numerario.dados import gerar_nn5_sintetico, preparar_dados  # noqa: E402
from numerario.rede import RedePiloto  # noqa: E402


@pytest.fixture(scope="session")
def base():
    cfg = Config()
    cfg.caminhos.dados_fonte = "sintetico"
    bruto = gerar_nn5_sintetico(semente=3)
    dados = preparar_dados(bruto, cfg)
    rede = RedePiloto.construir(cfg, dados)
    cal = CalendarioOperacional.construir(cfg, bruto.index[0], pd.Timestamp(cfg.datas.teste_fim) + pd.Timedelta(days=40))
    return cfg, dados, rede, cal
