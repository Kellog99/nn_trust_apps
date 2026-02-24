from typing import Optional

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from pdf_sections.pdf_section import PDFSection
from pdf_sections.utils import ModelInfo


class ModelInfoSection(PDFSection):
    """
    This class has the role to construct the info section of the document.
    """

    def __init__(
            self,
            corpus_width: Optional[float] = None,
            title_style: Optional[ParagraphStyle] = None,
            subtitle_style: Optional[ParagraphStyle] = None,
            description_style: Optional[ParagraphStyle] = None,
    ):
        super().__init__(
            corpus_width=corpus_width,
            title_style=title_style,
            subtitle_style=subtitle_style,
            description_style=description_style
        )

        self.table_style = TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F5')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#000000')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ])

    def build(
            self,
            data: ModelInfo | dict,
            description: Optional[str] = None
    ):
        if isinstance(data, dict):
            data: ModelInfo = ModelInfo.model_validate(data)

        elements = []
        model_name: str = getattr(data, "name", getattr(data, "id"))

        # Executive summary
        elements.append(
            Paragraph(
                text="Infographics",
                style=self.title_style
            )
        )

        elements.append(
            Paragraph(
                text=f"""
                This report provides a comprehensive analysis of the adversarial robustness 
                of the {model_name} model. The analysis includes multiple attack and evaluates the model's resilience against adversarial perturbations.
                """,
                style=self.description_style
            )
        )
        elements.append(Spacer(1, 20))

        elements.append(
            Paragraph(
                text="Model Information",
                style=self.subtitle_style
            )
        )

        # Model info table
        table_data = []
        for key, fieldInfo in data.model_fields.items():
            if getattr(data, key):
                title = getattr(fieldInfo, "title") if getattr(fieldInfo, "title") else key
                table_data.append([title, str(getattr(data, key, "N/A"))])

        elements.append(
            Table(
                data=table_data,
                colWidths=[self.corpus_width / 3, self.corpus_width / 3 * 2],
                style=self.table_style
            )
        )
        elements.append(Spacer(1, 50))

        return elements
