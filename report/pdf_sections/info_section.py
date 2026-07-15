from typing import Optional

from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from models import ModelInfo
from report.pdf_sections.pdf_section import PDFSection


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

        self.sub_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F5')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#333333')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#000000')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ])

    def build(
            self,
            data: ModelInfo | dict,
            description: Optional[str] = None
    ):
        if isinstance(data, dict):
            data: ModelInfo = ModelInfo.model_validate(data)

        elements = []
        model_name: str = data.name or data.id

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

        desc_style = ParagraphStyle(
            name='FieldDescription',
            fontName='Helvetica',  # Change to your template's font if needed
            fontSize=8,  # Smaller font size
            leading=10,
            textColor=colors.HexColor('#666666')  # Soft gray color
        )

        title_style = ParagraphStyle(
            name='FieldTitle',
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            textColor=colors.HexColor('#000000')
        )

        # Model info table
        table_data = []
        for key, fieldInfo in data.model_fields.items():
            if hasattr(data, key) and getattr(data, key) is not None:
                raw_val = getattr(data, key)

                # --- Upgrade 1: Title + Description ---
                title_text = fieldInfo.title or key
                if getattr(fieldInfo, 'description', None):
                    # We use Paragraphs to allow mixed styling and auto-wrapping in the cell
                    title_cell = [
                        Paragraph(title_text, title_style),
                        Paragraph(fieldInfo.description, desc_style)
                    ]
                else:
                    title_cell = Paragraph(title_text, title_style)

                # --- Upgrade 2: Dict to Nested Table ---
                if isinstance(raw_val, BaseModel):
                    raw_val: dict = raw_val.model_dump()

                if isinstance(raw_val, dict):
                    # Recursively build a table for the dictionary
                    sub_table_data = []
                    for sub_k, sub_v in raw_val.items():
                        sub_table_data.append([str(sub_k), str(sub_v)])

                    # Calculate column widths proportional to the available space in the right column
                    right_col_width = self.corpus_width / 3 * 2 - 20
                    val_cell = Table(
                        data=sub_table_data,
                        colWidths=[right_col_width / 3, right_col_width / 3 * 2],
                        style=self.sub_table_style
                    )
                else:
                    val_cell = str(raw_val)

                table_data.append([title_cell, val_cell])

        elements.append(
            Table(
                data=table_data,
                colWidths=[self.corpus_width / 3, self.corpus_width / 3 * 2],
                style=self.table_style
            )
        )
        elements.append(Spacer(1, 50))

        return elements
