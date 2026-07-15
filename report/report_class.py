from pathlib import Path
from typing import Optional

from reportlab.platypus import SimpleDocTemplate, PageBreak, Paragraph, Spacer

from models import ModelReportProps
from report.pdf_sections import (
    AttackSection,
    AttackRisk,
    HeaderFooter,
    ModelInfoSection,
    MetricsSection,
    Title
)
from report.report_style import AdversarialReportStyle


class AdversarialReportGenerator:
    """
    Main report generator class
    """

    def __init__(
            self,
            benchmark: Optional[dict] = None,
            excluded_metrics: Optional[list[str]] = None,
            style: AdversarialReportStyle = AdversarialReportStyle(),
    ):
        """
        Initialize report generator
        """
        self.style: AdversarialReportStyle = style

        self.header = HeaderFooter()

        # this variable represent the width of that part of the pdf
        # where it is possible to write
        corpus_width: float = style.pagesize[0] - style.leftMargin - style.rightMargin

        self.title = Title(
            pdf_title_style=style.pdf_title_style,
            title_style=style.section_title_style,
            description_style=style.section_description_style
        )

        self.info_section = ModelInfoSection(
            corpus_width=corpus_width,
            title_style=style.section_title_style,
            subtitle_style=style.section_subtitle_style,
            description_style=style.section_description_style
        )

        # Global metrics section
        self.global_metrics_section = MetricsSection(
            corpus_width=corpus_width,
            benchmark=benchmark,
            title_style=style.section_title_style,
            subtitle_style=style.section_subtitle_style,
            description_style=style.section_description_style
        )

        # Attacks resume section
        self.atk_table = AttackRisk(
            corpus_width=corpus_width,
            title_style=style.section_title_style,
            subtitle_style=style.section_subtitle_style,
            description_style=style.section_description_style
        )
        # single attack section
        self.atk_section = AttackSection(
            corpus_width=corpus_width,
            excluded_metrics=excluded_metrics,
            title_style=style.section_title_style,
            subtitle_style=style.section_subtitle_style,
            description_style=style.section_description_style
        )

    def generate(
            self,
            data: ModelReportProps | dict,
            output_path: Optional[str | Path] = None,
            output_file_name: Optional[str] = None,
            header_logo_path: Optional[str | Path] = None
    ):
        """
        Generate PDF report from data that could be in a proper format or in a dict format

        Args:
            data: data associated with a model's security assessment
            output_path: Output PDF file path
            output_file_name: name of the PDF file
            header_logo_path: path to the logo
        """

        ############################# FILE PATH VALIDATION #############################
        if data is None:
            raise ValueError("It is necessary to pass the data for the report.")

        if isinstance(data, dict):
            try:
                data: ModelReportProps = ModelReportProps.model_validate(data)
            except:
                raise ValueError("The data file provided is not in the format of ModelReportProps.")
        ################################################################################

        ############## Handling the saving path ##############
        # 1. Resolve and normalize the output path immediately
        output_path: Path = Path(output_path or "./tmp/").expanduser()
        # 2. Get the file name or default it (with fallback)
        file_name: str = data.info.name or data.info.id or "model_adversarial_report.pdf"
        file_name = file_name.replace(" ", "_").lower()
        # Adding the proper extension to the file
        if not file_name.endswith(".pdf"):
            file_name = file_name + ".pdf"

        # 3. Combine them, ensure the extension is .pdf, and build the parent directory
        output_file = output_path / file_name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        ################# Document Creation #################
        doc = SimpleDocTemplate(
            str(output_path / output_file_name),
            pagesize=self.style.pagesize,
            rightMargin=self.style.rightMargin,
            leftMargin=self.style.leftMargin,
            topMargin=self.style.topMargin,
            bottomMargin=self.style.bottomMargin,
        )

        #############################################

        # Build content
        story = []
        ##################### Sections #####################
        # 1) Title page
        story.extend(
            self.title.build(
                data=data.info,
                description="ssssssssssssss"
            )
        )

        # 2) Model information
        story.extend(
            self.info_section.build(
                data=data.info,
            )
        )

        # 3) Global metrics
        story.extend(
            self.global_metrics_section.build(
                data=data.metrics,
                description="Here below, it is shown a table containing all the metrics that have been computed during the benchmark and their placement among a benchmark if it exists."
            )
        )

        # 4) Attack table summary
        story.extend(
            self.atk_table.build(
                data=data.attacks,
                descriptions="""
                This table contains all the attacks that have been tested and a subset of all the metrics that have been computed for a global comparison.
                """
            )
        )
        story.append(PageBreak())

        # 5) Attacks analysis
        # If the path_root is none then no example are shown
        if data.info.repository is not None and isinstance(data.info.repository, str):
            path_root: Path = Path(data.info.repository) / "examples"

        story.append(
            Paragraph(
                text="Adversarial Attacks Analysis",
                style=self.style.section_title_style
            )
        )
        story.append(
            Paragraph(
                text="Here below are shown all the specifics about all the attacks that have been performed and their parameters.",
                style=self.style.section_description_style
            )
        )
        story.append(Spacer(1, 10))

        for atk_name, atk_info in data.attacks.items():
            story.extend(self.atk_section.build(
                data=atk_info
            ))

        ####################################################

        # Build PDF
        if header_logo_path:
            ################### header logo path validation ###################
            if isinstance(header_logo_path, str):
                header_logo_path = Path(header_logo_path).expanduser()
            if not header_logo_path.exists():
                raise ValueError("The path to the logo does not exists.")
            if not header_logo_path.is_file():
                raise ValueError("The logo is not a file.")
            ###################################################################

            doc.build(
                story,
                onFirstPage=HeaderFooter(logo_path=header_logo_path),
                onLaterPages=HeaderFooter(logo_path=header_logo_path)
            )
        else:
            doc.build(story)
