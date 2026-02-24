import abc
from typing import Optional

from pydantic import BaseModel
from reportlab.lib.styles import ParagraphStyle


class PDFSection(abc.ABC):
    """
    Adversarial attacks section builder
    """

    def __init__(
            self,
            corpus_width: Optional[float] = None,
            title_style: Optional[ParagraphStyle] = None,
            subtitle_style: Optional[ParagraphStyle] = None,
            description_style: Optional[ParagraphStyle] = None
    ):
        self.corpus_width = corpus_width if corpus_width else 500
        self.title_style = title_style
        self.subtitle_style = subtitle_style
        self.description_style = description_style

    @abc.abstractmethod
    def build(self, data: BaseModel, description: Optional[str] = None) -> list:
        """
        This function has the role to build a section for each component
        """
        pass

    def _format_value(self, value):
        if value is None:
            return 'N/A'
        if isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
