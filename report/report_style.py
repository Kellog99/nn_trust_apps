from pydantic import BaseModel, Field, ConfigDict
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle


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
            fontSize=20,
            textColor=colors.HexColor('#000000'),
            spaceAfter=25,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ),
        description="This represent the title's style of the whole PDF, i.e. `Model Trustworthy`."
    )

    section_title_style: ParagraphStyle = Field(
        default_factory=lambda: ParagraphStyle(
            name='CustomHeading1',
            fontSize=12,
            textColor=colors.HexColor('#CC0000'),
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderRadius=6,
            borderColor=colors.HexColor('#CC0000'),
            borderPadding=6,
            spaceAfter=10,
            leftIndent=0,
            leading=14,
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
