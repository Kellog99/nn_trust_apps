from pathlib import Path

from reportlab.lib.utils import ImageReader

from report.corporate_colors import CorporateColors as C

DEFAULT_HEADER_LOGO = Path(__file__).resolve().parents[1] / 'images' / 'Logo_Leonardo.png'


class HeaderFooter:
    """Paint every page using the document's actual size."""
    HEADER_HEIGHT = 68

    def __init__(self, logo_path=None):
        self.logo = ImageReader(str(logo_path or DEFAULT_HEADER_LOGO))

    def __call__(self, canvas_obj, doc):
        canvas_obj.saveState()
        width, height = doc.pagesize
        canvas_obj.setFillColor(C.BACKGROUND)
        canvas_obj.rect(0, 0, width, height, fill=1, stroke=0)
        canvas_obj.setStrokeColor(C.BORDER)
        canvas_obj.setLineWidth(.5)
        canvas_obj.line(doc.leftMargin, 32, width - doc.rightMargin, 32)
        canvas_obj.setFillColor(C.MUTED)
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.drawString(doc.leftMargin, 19, 'Security Report')
        canvas_obj.drawRightString(width - doc.rightMargin, 19, f'Page {doc.page}')
        logo_width, logo_height = self.logo.getSize()
        scale = min(140 / logo_width, 28 / logo_height, doc.width / logo_width)
        logo_width, logo_height = logo_width * scale, logo_height * scale
        canvas_obj.drawImage(self.logo, doc.leftMargin, height - 20 - logo_height,
                             width=logo_width, height=logo_height, mask='auto')
        canvas_obj.restoreState()
