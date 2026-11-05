import json

import numpy as np
import pandas as pd

from numerario.artefatos import escrever_painel


def test_json_do_painel_e_estrito(tmp_path):
    """O painel lê os dados com JSON.parse, que recusa NaN: tudo que não é finito vira null."""
    tpl = tmp_path / "template.html"
    tpl.write_text('<script type="application/json" id="dados">__DADOS_DO_NOTEBOOK__</script>', encoding="utf-8")
    dados = {"gap": float("nan"), "serie": [np.float64("inf"), np.int64(3), np.nan],
             "data": pd.Timestamp("1998-03-23"), "texto": "</script>", np.int64(7): {"ok": np.bool_(True)}}
    html = open(escrever_painel(dados, str(tpl), str(tmp_path / "painel.html")), encoding="utf-8").read()
    bloco = html.split('id="dados">', 1)[1].rsplit("</script>", 1)[0]
    assert "NaN" not in bloco and "Infinity" not in bloco
    assert json.loads(bloco) == {"gap": None, "serie": [None, 3, None], "data": "1998-03-23 00:00:00",
                                 "texto": "</script>", "7": {"ok": True}}
