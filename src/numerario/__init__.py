"""numerario — previsão e otimização da cadeia de suprimento de numerário em ATMs.

Projeto de Pesquisa Operacional Aplicada (UFABC, 2026.3): MILP estocástico
multiescalão (custódia do BC → tesouraria central → agências → ATMs) com
cenários dos quantis do Chronos-2, avaliado por simulação em horizonte rolante
no NN5. O notebook do Colab orquestra; o pacote calcula; o painel HTML lê.
"""

__version__ = "0.1.0"

from .config import Config  # noqa: E402

__all__ = ["Config", "__version__"]
