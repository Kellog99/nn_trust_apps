from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from report.pdf_sections.dashboard import Dashboard
from report.pdf_sections.header import DEFAULT_HEADER_LOGO, HeaderFooter
from report.report_style import AdversarialReportStyle

if TYPE_CHECKING:
    from models import ModelReportProps


class AdversarialReportGenerator:
    """Frontend-aligned PDF, including individual attack details by default.

    Benchmarks accept backend BenchmarkModelProps objects/dictionaries or the
    legacy metric-to-scores mapping. No network access is performed during render.
    """
    def __init__(self, benchmark=None, excluded_metrics=None, style=None,
                 include_attack_details=True):
        self.style = style or AdversarialReportStyle()
        self.include_attack_details = include_attack_details
        self.dashboard = Dashboard(self.style, benchmark, excluded_metrics)

    def pdf_to_bytesio(self, pdf_path: str | Path) -> BytesIO:
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'PDF not found: {path}')
        if not path.is_file() or path.suffix.lower() != '.pdf':
            raise ValueError(f'{path} is not a PDF file.')
        return BytesIO(path.read_bytes())

    def build_story(self, data: 'ModelReportProps | dict', output_folder='./', examples_root=None):
        return self.dashboard.build(data, self.include_attack_details, examples_root)

    def generate(self, data: 'ModelReportProps | dict', output_path=None, header_logo_path=None,
                 examples_root=None):
        if data is None:
            raise ValueError('It is necessary to pass the data for the report.')
        # Preserve JSON ordering and frontend fields not declared by backend models.
        if isinstance(data, dict):
            if not all(isinstance(data.get(k), dict) for k in ('info', 'metrics', 'attacks')):
                raise ValueError('Report must contain info, metrics and attacks objects.')
        path = Path(output_path or 'model_report.pdf').expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        header_logo_path = Path(header_logo_path or DEFAULT_HEADER_LOGO).expanduser()
        if not header_logo_path.is_file():
            raise ValueError('The logo must be an existing file.')
        style = self.style
        doc = BaseDocTemplate(str(path), pagesize=style.pagesize,
                              leftMargin=style.leftMargin, rightMargin=style.rightMargin,
                              topMargin=max(style.topMargin, HeaderFooter.HEADER_HEIGHT),
                              bottomMargin=style.bottomMargin,
                              title='Security Report', author='')
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates(PageTemplate(id='dashboard', frames=[frame],
                                          onPage=HeaderFooter(header_logo_path)))
        doc.build(self.build_story(data, path.parent, examples_root))
