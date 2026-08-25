"""Modelos de dominio compartilhados pela biblioteca de midia."""

from dataclasses import dataclass


class FundoInvalidoError(RuntimeError):
    """Indica que um video de fundo nao pode ser usado pelo pipeline."""


@dataclass
class MediaItem:
    """Item de midia localmente disponivel para o pipeline."""

    caminho: str
    categoria: str
    tipo: str