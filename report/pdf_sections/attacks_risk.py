from typing import Optional

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from pdf_sections.pdf_section import PDFSection
from pdf_sections.utils import ReportAttacksProps


class AttackRisk(PDFSection):
    """
    Adversarial attacks section builder
    """

    def __init__(
            self,
            corpus_width: Optional[float] = None,
            title_style: Optional[ParagraphStyle] = None,
            subtitle_style: Optional[ParagraphStyle] = None,
            description_style: Optional[ParagraphStyle] = None,
            metrics_to_show: Optional[list[str]] = None,
    ):
        super().__init__(
            corpus_width=corpus_width,
            title_style=title_style,
            subtitle_style=subtitle_style,
            description_style=description_style
        )
        for metric in metrics_to_show:
            if metric not in ReportAttacksProps.model_fields.keys():
                raise ValueError(f"The metrics, {metric}, is not a proper value of ReportAttacksProps.")

        # The risk must be a metric that is shown
        # Moreover, it has to be positioned at the end
        metrics_to_show = [metric for metric in metrics_to_show if metric != "risk"]
        metrics_to_show.append("risk")

        self.metrics_to_show = metrics_to_show

        self.table_style = TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#5C5C5C")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),

            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),

            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F5F5F5')]),

            # General styling
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])

    def build(
            self,
            data: dict[str, ReportAttacksProps | dict],
            descriptions: Optional[str] = None
    ):
        elements = []
        elements.append(
            Paragraph(
                text="Attack Summary",
                style=self.title_style
            )
        )
        if descriptions:
            elements.append(
                Paragraph(
                    text=descriptions,
                    style=self.description_style
                )
            )
            elements.append(Spacer(1, 20))

        # Prepare summary data
        headers = ['Attack'] + [metric.upper() for metric in self.metrics_to_show]
        table_data = [headers]
        for attack_name, attack_data in data.items():
            # if attack_name != 'reference':
            row = [attack_name.upper()]
            for metric in self.metrics_to_show:
                row.append(self._format_value(getattr(attack_data, metric, "N/A")))
            table_data.append(row)

        elements.append(
            Table(
                table_data,
                colWidths=[self.corpus_width / len(headers) for _ in headers],
                style=self.table_style
            )
        )

        # Keep title and table together
        elements.append(Spacer(1, 20))

        return elements
