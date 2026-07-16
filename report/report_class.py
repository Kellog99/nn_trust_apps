from io import BytesIO
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

    def pdf_to_bytesio(self, pdf_path: str | Path) -> BytesIO:
        """
        Load a PDF from disk into a BytesIO object.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            BytesIO containing the PDF data.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the path is not a PDF file.
        """
        if isinstance(pdf_path, str):
            pdf_path: Path = Path(pdf_path).expanduser().resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not pdf_path.is_file():
            raise ValueError(f"{pdf_path} is not a file.")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"{pdf_path} is not a PDF file.")

        with pdf_path.open("rb") as f:
            return BytesIO(f.read())

    def build_story(
            self,
            data: ModelReportProps,
            output_folder: str | Path = "./"
    ) -> list:
        """
        It builds the story of the PDF: the pages, the content and the plot
        """
        # Build content
        story = []
        ##################### Sections #####################
        # 1) Title page
        story.extend(
            self.title.build(
                data=data.info,
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
                description="Here below, it is shown a table containing all the metrics that have been computed during the benchmark and their placement among a benchmark if it exists.",
                output_folder=output_folder
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
        return story

    def generate(
            self,
            data: ModelReportProps | dict,
            output_path: Optional[str | Path] = None,
            header_logo_path: Optional[str | Path] = None
    ):
        """
        Generate PDF report from data that could be in a proper format or in a dict format

        Args:
            data: data associated with a model's security assessment
            output_path: Output PDF file path
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

        ################# Document Creation #################
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.style.pagesize,
            rightMargin=self.style.rightMargin,
            leftMargin=self.style.leftMargin,
            topMargin=self.style.topMargin,
            bottomMargin=self.style.bottomMargin,
        )

        story = self.build_story(
            data=data,
            output_folder=output_path.parent
        )

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
            header_footer = HeaderFooter(logo_path=header_logo_path)
        else:
            header_footer = HeaderFooter()
        doc.build(
            story,
            onFirstPage=header_footer,
            onLaterPages=header_footer,
        )

        #############################################
