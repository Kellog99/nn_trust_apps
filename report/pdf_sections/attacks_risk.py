from typing import Optional, get_origin, Union, get_args

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from models.reports import AttackMetricsProps, ReportAttackProps
from report.pdf_sections.pdf_section import PDFSection


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
            metrics: list[str] | None = ["accuracy", "precision", "f1score", "misclassification", "robustness"]
    ):
        super().__init__(
            corpus_width=corpus_width,
            title_style=title_style,
            subtitle_style=subtitle_style,
            description_style=description_style
        )
        if metrics is None:
            metrics = []
            for field_name, field_info in AttackMetricsProps.model_fields.items():
                annotation = field_info.annotation

                # Check if it's directly a float
                if annotation is float or annotation is int:
                    metrics.append(field_name)
                    continue

                # Check if it's an Optional (Union[float, None])
                if get_origin(annotation) is Union:
                    args = get_args(annotation)
                    if float in args or int in args:
                        metrics.append(field_name)
        # These represent all the possible metrics that can be shown
        self.metrics = metrics

        self.table_style = TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#5C5C5C")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),

            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),

            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F5F5F5')]),

            # General styling
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ])

    def build(
            self,
            data: dict[str, ReportAttackProps | dict],
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
        headers = ['Attack'] + self.metrics
        table_data = [headers]
        for _, attack_data in data.items():
            if isinstance(attack_data, dict):
                attack_data: ReportAttackProps = ReportAttackProps.model_validate((attack_data))

            attack_name = attack_data.name
            # if attack_name != 'reference':
            if attack_name.lower().endswith("attack"):
                attack_name = attack_name.removesuffix("attack")
            if attack_name.endswith("_"):
                attack_name = attack_name.removesuffix("_")

            row = [Paragraph(
                text=attack_name,
                style=ParagraphStyle(
                    name='CustomTitle',
                    fontSize=8,
                    textColor=colors.HexColor('#000000'),
                    alignment=TA_LEFT,
                    fontName='Helvetica-Bold'
                ))]
            for metric in self.metrics:
                row.append(self._format_value(getattr(attack_data.metrics, metric)))
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
