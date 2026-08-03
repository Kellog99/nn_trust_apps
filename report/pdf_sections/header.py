from datetime import datetime

from reportlab.lib.pagesizes import A4

from report.corporate_colors import CorporateColors


class HeaderFooter:
    """Page header and footer handler"""

    def __init__(self, logo_path=None):
        self.logo_path = logo_path

    def __call__(self, canvas_obj, doc):
        canvas_obj.saveState()

        # Header
        if self.logo_path:
            try:
                canvas_obj.drawImage(
                    self.logo_path,
                    40, A4[1] - 80,
                    width=100, height=100,
                    preserveAspectRatio=True,
                    mask='auto'
                )
            except:
                pass  # Skip if logo not found

        # Red header line
        canvas_obj.setStrokeColor(CorporateColors.RED)
        canvas_obj.setLineWidth(2)
        canvas_obj.line(40, A4[1] - 60, A4[0] - 40, A4[1] - 60)

        # Footer
        canvas_obj.setStrokeColor(CorporateColors.LIGHT_GRAY)
        canvas_obj.setLineWidth(1)
        canvas_obj.line(40, 40, A4[0] - 40, 40)

        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(CorporateColors.GRAY)
        canvas_obj.drawString(40, 25, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas_obj.drawRightString(A4[0] - 40, 25, f"Page {doc.page}")

        canvas_obj.restoreState()
