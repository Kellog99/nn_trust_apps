import os
from pathlib import Path
from typing import Optional

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, Image

from models.model import ParametersProps
from models.reports import AttackMetricsProps, ReportAttackProps
from report.corporate_colors import CorporateColors
from report.pdf_sections.pdf_section import PDFSection


class AttackSection(PDFSection):
    """
    This Section provides all the information about a single attack
    """

    def __init__(
            self,
            corpus_width: Optional[float] = None,
            title_style: Optional[ParagraphStyle] = None,
            subtitle_style: Optional[ParagraphStyle] = None,
            description_style: Optional[ParagraphStyle] = None,
            excluded_metrics: Optional[list[str]] = None,
            excluded_parameters: Optional[list[str]] = None,
            available_suffix: list[str] = ["png", "jpg", "jpeg", "JPEG"],
    ):
        super().__init__(
            corpus_width=corpus_width,
            title_style=title_style,
            subtitle_style=subtitle_style,
            description_style=description_style
        )
        # The parameters don't have to be displayed in the metrics' table
        if not excluded_metrics:
            excluded_metrics = ["parameters", "name"]

        else:
            if "parameters" not in excluded_metrics:
                excluded_metrics.append("parameters")
            if "name" not in excluded_metrics:
                excluded_metrics.append("name")

        self.excluded_metrics = excluded_metrics

        # List of the excluded parameters to show in the report
        if not excluded_parameters:
            excluded_parameters = []
        self.excluded_parameters = excluded_parameters

        self.available_suffix = available_suffix

        self.metric_table_style = TableStyle([
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 0), (-1, 0), CorporateColors.TABLE_HEADER),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.TABLE_GRID),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])

        self.parameters_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), CorporateColors.TABLE_HEADER),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.TABLE_GRID),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ])

    def build_table_metrics(self, data: AttackMetricsProps) -> Table:
        """
        This function has to construct the table that collects all the metrics of the attacks
        """

        # Metrics table
        table_data = [["Metric", "Result"]]
        for metric, metric_value in data.model_dump().items():
            if self.excluded_metrics and metric not in self.excluded_metrics:
                label = metric.replace('_', ' ').title()
                value = self._format_value(metric_value)
                table_data.append([label, value])

        return Table(
            data=table_data,
            colWidths=[self.corpus_width / 3, self.corpus_width / 3 * 2],
            style=self.metric_table_style
        )

    def build_table_parameters(self, parameters: dict) -> list:
        """
        This function has to construct the table for the attack's parameters
        """
        out: list[Paragraph | Table | Spacer] = [
            Paragraph(
                text="Parameters List",
                style=self.subtitle_style
            )
        ]

        # It can happen that no parameters have been saved
        if parameters:
            out.append(
                Paragraph(
                    text="This table shows all the parameters that have been used for executing this attack.",
                    style=self.description_style
                )
            )
            out.append(Spacer(1, 10))
            # Parameters table
            header = ["Parameter", "Value"]
            table_data: list[list] = [header]

            for param, param_value in parameters.items():
                if param not in self.excluded_metrics:
                    label = param.replace('_', ' ')
                    value = self._format_value(param_value)
                    table_data.append([label, value])

            out.append(
                Table(
                    data=table_data,
                    colWidths=[self.corpus_width / len(header) for _ in range(len(header))],
                    rowHeights=15,
                    style=self.parameters_table_style
                )
            )
        else:
            out.append(
                Paragraph(
                    text="No parameters were saved for this attack.",
                    style=self.description_style
                )
            )
        return out

    def build_example(
            self,
            path_root: Optional[str | Path] = None
    ) -> list:
        """
        This function has to construct the table for the attack's parameters
        """
        if isinstance(path_root, str):
            path_root: Path = Path(path_root)

        out: list[Paragraph | Table | Spacer] = [
            Paragraph(
                text="Examples",
                style=self.subtitle_style
            )
        ]

        # It can happen that no parameters have been saved
        if path_root and path_root.exists():
            out.append(
                Paragraph(
                    text="Here below it is possible to show some example of adversarial input and their perturbation.",
                    style=self.description_style
                )
            )
            out.append(Spacer(1, 10))

            images_dict: dict[str, dict] = {}
            for image in os.listdir(path_root):
                split = image.rsplit('.', 1)
                suffix = split[-1]

                if suffix in self.available_suffix:
                    # Extract the base name (index) without extension
                    name_without_ext = image.rsplit('.', 1)[0]

                    # [index, type of image]
                    image_type = name_without_ext.rsplit("_", maxsplit=1)
                    if image_type[0] not in images_dict:
                        images_dict[image_type[0]] = {}

                    # I add the image if only has the following suffix
                    if image_type[1] in ["original", "pert", "adv"]:
                        images_dict[image_type[0]][image_type[1]] = str(path_root / image)

            image_table = [["Original", "Perturbation", "Adversarial"]]
            for base_index in sorted(images_dict.keys()):
                img_group: dict[str, str] = images_dict[base_index]
                image_row: list[Image | Paragraph] = []
                for image_type in ["original", "pert", "adv"]:
                    if image_type in img_group.keys():
                        image_row.append(
                            Image(
                                filename=img_group[image_type],
                                width=150,
                                height=150,
                                kind="proportional"
                            )
                        )
                    else:
                        image_row.append(
                            Paragraph(
                                text="N/A",
                                style=self.description_style
                            )
                        )
                image_table.append(image_row)

            table = Table(
                data=image_table,
                colWidths=[self.corpus_width / 3 for _ in range(3)],
                style=TableStyle([
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.TABLE_GRID),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 0), (-1, 0), CorporateColors.TABLE_HEADER),
                ])
            )

            out.append(table)
            out.append(Spacer(1, 20))



        else:
            out.append(
                Paragraph(
                    text="No example to show.",
                    style=self.description_style
                )
            )
        return out

    def build(
            self,
            data: ReportAttackProps | dict,
            description: Optional[str] = None,
            path_root: Optional[str | Path] = None,
    ) -> list:
        """
        It builds the section with all the attack's detail.

        Args:
            data: (dict) it contains the information about each attack.
            description (str): description to add after the title section.
            path_root (str | Path): it represents the path where the examples are stored.
        """
        elements = []
        if isinstance(data, dict):
            data: ReportAttackProps = ReportAttackProps.model_validate(data)

        name: str = data.name
        metrics: AttackMetricsProps = data.metrics
        parameters: list[ParametersProps] = data.parameters

        elements.append(
            Paragraph(
                text=f"<b>{name.upper()}</b> Attack Details",
                style=self.subtitle_style
            )
        )
        elements.append(Spacer(1, 20))

        elements.append(self.build_table_metrics(data=metrics))
        elements.append(Spacer(1, 20))

        elements.extend(self.build_table_parameters(parameters=parameters))
        elements.append(Spacer(1, 20))

        elements.extend(self.build_example(path_root=path_root / name if path_root else None))
        elements.append(Spacer(1, 20))

        return elements
