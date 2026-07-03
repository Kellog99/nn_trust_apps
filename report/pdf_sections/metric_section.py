from typing import Optional

import matplotlib.pyplot as plt
from reportlab.lib.colors import Color
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image

from report.pdf_sections.pdf_section import PDFSection
from report.pdf_sections.utils import ReportMetricsProps, CorporateColors


class MetricsSection(PDFSection):
    """
    This section has the role to shows 2 things:
        1) The metrics that have been computed during the benchmark
        2) The absolute position of that metrics compared with other models on the same metric
    """

    def __init__(
            self,
            corpus_width: Optional[float] = None,
            title_style: Optional[ParagraphStyle] = None,
            subtitle_style: Optional[ParagraphStyle] = None,
            description_style: Optional[ParagraphStyle] = None,
            benchmark: Optional[dict] = None,
            excluded_metrics: Optional[list[str]] = None,
            table_height: Optional[float] = None,
            header_color: Optional[Color] = None,
    ):
        super().__init__(
            corpus_width=corpus_width,
            title_style=title_style,
            subtitle_style=subtitle_style,
            description_style=description_style
        )

        self.benchmark = benchmark
        self.excluded_metrics = excluded_metrics

        # Style settings
        self.table_height = table_height
        self.header_color = header_color if header_color else CorporateColors.TABLE_HEADER

        self.table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.header_color),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.TABLE_GRID),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ])

    def build(self,
              data: ReportMetricsProps | dict,
              description: Optional[str] = None
              ):
        if isinstance(data, dict):
            data: ReportMetricsProps = ReportMetricsProps.model_validate(data)

        elements = []

        elements.append(
            Paragraph(
                text="Model Performance",
                style=self.title_style
            )
        )
        if description:
            elements.append(
                Paragraph(
                    text=description,
                    style=self.description_style
                )
            )
            elements.append(Spacer(1, 20))
        elements.append(
            Paragraph(
                text="Metrics Information",
                style=self.subtitle_style
            )
        )
        # Metrics table
        table_data = [["Metric", "Value", "Position"]]
        for param, value in data.model_dump().items():
            if self.excluded_metrics and param not in self.excluded_metrics:
                value = self._format_value(value)
                pos = "1/1"
                if self.benchmark:
                    # counting the models that are better than the targeted model
                    better_models: int = sum([value < bench for bench in self.benchmark[param]])
                    # the " + 1 " is because the benchmark does not include this model
                    pos = f"{better_models + 1}/{len(self.benchmark) + 1}"
                table_data.append([param.replace('_', ' ').title(), value, pos])

        # adding the metrics table to the list
        elements.append(
            Table(
                data=table_data,
                colWidths=self.corpus_width / 3,
                style=self.table_style
            )
        )

        # Confusion Matrix
        if hasattr(data, "confusion_matrix"):
            # Create the figure and axis
            cm = getattr(data, "confusion_matrix")

            # The matrix has to be normalized
            if sum(cm[0]) != 1:
                total = sum(cm[0])
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        cm[i][j] = cm[i][j] / total

            # Configure the labels and ticks
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                cm,
                vmax=1,
                vmin=0,
                interpolation='nearest',
                cmap=plt.cm.Blues
            )
            # Add a colorbar
            ax.figure.colorbar(im, ax=ax)

            ax.set(title='Confusion Matrix',
                   ylabel='True Label',
                   xlabel='Predicted Label')

            fig.tight_layout()
            plt.savefig('./confusion_matrix_matplotlib.png')
            img = Image(
                filename='./confusion_matrix_matplotlib.png',
                width=400,
                height=300,
                kind='proportional'
            )
            img.hAlign = 'CENTER'

            ############ adding the element to the story #########
            elements.append(Spacer(1, 10))
            elements.append(
                Paragraph(
                    text="Confusion Matrix",
                    style=self.subtitle_style
                )
            )
            elements.append(
                Paragraph(
                    text=
                    """
                    This represents the confusion matrix that the model generates on the tested dataset.
                    This result is not influenced by any attack.
                    """,
                    style=self.description_style
                )
            )
            elements.append(img)

        elements.append(Spacer(1, 20))

        return elements
