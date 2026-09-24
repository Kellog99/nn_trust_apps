from pydantic import BaseModel, ConfigDict, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from report.corporate_colors import CorporateColors as C


def text_style(name, size, color=C.TEXT, bold=False, **kwargs):
    return ParagraphStyle(name, fontName='Helvetica-Bold' if bold else 'Helvetica',
                          fontSize=size, leading=size * 1.35, textColor=color,
                          splitLongWords=True, **kwargs)


class AdversarialReportStyle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pagesize: tuple[float, float] = A4
    rightMargin: float = 32
    leftMargin: float = 32
    topMargin: float = 32
    bottomMargin: float = 44
    pdf_title_style: ParagraphStyle = Field(default_factory=lambda: text_style(
        'ReportTitle', 20, bold=True, spaceAfter=4, keepWithNext=True))
    section_title_style: ParagraphStyle = Field(default_factory=lambda: text_style(
        'SectionTitle', 16, bold=True, spaceBefore=12, spaceAfter=14, keepWithNext=True))
    section_subtitle_style: ParagraphStyle = Field(default_factory=lambda: text_style(
        'Subtitle', 11, bold=True, spaceAfter=8, keepWithNext=True))
    section_description_style: ParagraphStyle = Field(default_factory=lambda: text_style(
        'Description', 9, C.SECONDARY, spaceAfter=10))
    metric_value: ParagraphStyle = Field(default_factory=lambda: text_style('MetricValue', 17, bold=True))
