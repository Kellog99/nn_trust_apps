from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate

from report.pdf_sections import (
    AttackSection,
    AttackRisk,
    HeaderFooter,
    ModelInfoSection,
    MetricsSection,
    Title,
    ModelReportProps
)


class AdversarialReportStyle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pagesize: tuple[float, float] = A4
    rightMargin: float = 40
    leftMargin: float = 40
    topMargin: float = 80
    bottomMargin: float = 60

    pdf_title_style: ParagraphStyle = Field(
        default_factory=lambda: ParagraphStyle(
            name='CustomTitle',
            fontSize=28,
            textColor=colors.HexColor('#000000'),
            spaceAfter=30,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ),
        description="This represent the title's style of the whole PDF, i.e. `Model Trustworthy`."
    )

    section_title_style: ParagraphStyle = Field(
        default_factory=lambda: ParagraphStyle(
            name='CustomHeading1',
            fontSize=18,
            textColor=colors.HexColor('#CC0000'),
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderRadius=5,
            borderColor=colors.HexColor('#CC0000'),
            borderPadding=5,
            spaceAfter=20,
            leftIndent=0,
            leading=22,
        ),
        description="This represent the style of the section's title."
    )

    section_subtitle_style: ParagraphStyle = Field(
        default_factory=lambda: ParagraphStyle(
            name='MetricLabel',
            fontSize=12,
            leading=18,
            textColor=colors.HexColor('#000000'),
            fontName='Helvetica-Bold'
        ),
        description="This represents the style associated with sub title"
    )

    section_description_style: ParagraphStyle = Field(
        default_factory=lambda: ParagraphStyle(
            name='CustomHeading2',
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            fontName='Helvetica'
        ),
        description="This represents the style of the description of each section."
    )

    metric_value: ParagraphStyle = Field(
        default_factory=lambda: ParagraphStyle(
            name='MetricValue',
            fontSize=11,
            textColor=colors.HexColor('#000000'),
            fontName='Helvetica'
        )
    )


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
            excluded_metrics=excluded_metrics,
            title_style=style.section_title_style,
            subtitle_style=style.section_subtitle_style,
            description_style=style.section_description_style
        )

        # Attacks resume section
        self.atk_table = AttackRisk(
            corpus_width=corpus_width,
            title_style=style.section_title_style,
            subtitle_style=style.section_subtitle_style,
            description_style=style.section_description_style,
            metrics_to_show=["robustness", "accuracy", "f1score", "precision"]
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
            file_path: path to the data file
            output_path: Output PDF file path
            output_file_name: name of the PDF file
            header_logo_path: path to the logo
        """

        ############################# FILE PATH VALIDATION #############################
        if data is None:
            raise ValueError("Either file_path or data must be provided.")

        if isinstance(data, dict):
            try:
                data: ModelReportProps = ModelReportProps.model_validate(data)
            except:
                raise ValueError("The data file provided is not in the format of ModelReportProps.")
        ################################################################################

        ############## Handling the saving path ##############
        if not output_path:
            output_path = Path("report/")
        if isinstance(output_path, str):
            output_path = Path(output_path).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_file_name:
            output_file_name: str | None = data.info.name
            if not output_file_name:
                output_file_name = "model_adversarial_report.pdf"
            else:
                output_file_name = output_file_name.replace(" ", "_").lower()
                if not output_file_name.endswith(".pdf"):
                    output_file_name += ".pdf"

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
        story.extend(self.title.build(data=data.info))

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
                descriptions="This table represents the list of all the attacks that have been performed and the main metrics that have been computed."
            )
        )

        # 5) Attacks analysis
        story.extend(
            self.atk_section.build(
                data=data.attacks,
                description="Here below are shown all the specifics about all the attacks that have been performed and their parameters.",
                path_root=data.info.repository
            )
        )

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

        print(f"Report generated: {output_path}")
