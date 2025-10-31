"""
Corporate Adversarial Attack Report Generator
A modular ReportLab-based PDF report generator for model robustness analysis
"""

import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class CorporateColors:
    """Corporate color palette"""
    BLACK = colors.HexColor('#000000')
    DARK_GRAY = colors.HexColor('#333333')
    GRAY = colors.HexColor('#666666')
    LIGHT_GRAY = colors.HexColor('#CCCCCC')
    VERY_LIGHT_GRAY = colors.HexColor('#F5F5F5')
    RED = colors.HexColor('#CC0000')
    WHITE = colors.HexColor('#FFFFFF')


class ReportStyles:
    """Centralized style definitions"""
    
    @staticmethod
    def get_styles():
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=28,
            textColor=CorporateColors.BLACK,
            spaceAfter=30,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 1
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=CorporateColors.RED,
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=CorporateColors.RED,
            borderPadding=5,
            leftIndent=0
        ))
        
        # Heading 2
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=CorporateColors.DARK_GRAY,
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            textColor=CorporateColors.DARK_GRAY,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Metric label
        styles.add(ParagraphStyle(
            name='MetricLabel',
            fontSize=9,
            textColor=CorporateColors.GRAY,
            fontName='Helvetica-Bold'
        ))
        
        # Metric value
        styles.add(ParagraphStyle(
            name='MetricValue',
            fontSize=11,
            textColor=CorporateColors.BLACK,
            fontName='Helvetica'
        ))
        
        return styles


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


class ModelInfoSection:
    """Model information section builder"""
    
    @staticmethod
    def build(data, styles):
        elements = []
        info = data.get('info', {})
        
        title = Paragraph("Model Information", styles['CustomHeading1'])
        
        # Model info table
        table_data = [
            ['Model Name', info.get('name', 'N/A')],
            ['Parameters', f"{info.get('parameters', 0):,}"],
            ['Classes', str(info.get('classes', 'N/A'))],
            ['Input Dimensions', str(info.get('dimensionality', 'N/A'))]
        ]
        
        table = Table(table_data, colWidths=[150, 350])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), CorporateColors.VERY_LIGHT_GRAY),
            ('TEXTCOLOR', (0, 0), (0, -1), CorporateColors.DARK_GRAY),
            ('TEXTCOLOR', (1, 0), (1, -1), CorporateColors.BLACK),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        # Keep title and table together
        elements.append(KeepTogether([title, table]))
        elements.append(Spacer(1, 20))
        
        return elements


class MetricsSection:
    """Global metrics section builder"""
    
    @staticmethod
    def build(data, styles):
        elements = []
        metrics = data.get('metrics', {})
        
        if not metrics:
            return elements
        
        title = Paragraph("Global Metrics", styles['CustomHeading1'])
        
        # Metrics table
        table_data = []
        for key, value in metrics.items():
            formatted_value = MetricsSection._format_metric(value)
            table_data.append([key.replace('_', ' ').title(), formatted_value])
        
        if table_data:
            table = Table(table_data, colWidths=[150, 350])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), CorporateColors.VERY_LIGHT_GRAY),
                ('TEXTCOLOR', (0, 0), (0, -1), CorporateColors.DARK_GRAY),
                ('TEXTCOLOR', (1, 0), (1, -1), CorporateColors.BLACK),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            # Keep title and table together
            elements.append(KeepTogether([title, table]))
            elements.append(Spacer(1, 20))
        
        return elements
    
    @staticmethod
    def _format_metric(value):
        if isinstance(value, float):
            if value < 0.01:
                return f"{value:.2e}"
            return f"{value:.4f}"
        elif isinstance(value, int):
            return f"{value:,}"
        return str(value)


class AttacksSection:
    """Adversarial attacks section builder"""
    
    @staticmethod
    def build(data, styles):
        elements = []
        attacks = data.get('attacks', {})
        
        if not attacks:
            return elements
        
        elements.append(Paragraph("Adversarial Attacks Analysis", styles['CustomHeading1']))
        
        # Summary table
        elements.extend(AttacksSection._build_summary_table(attacks, styles))
        
        # Detailed attack reports
        for attack_name, attack_data in attacks.items():
            if attack_name != 'reference':
                elements.extend(AttacksSection._build_attack_detail(
                    attack_name, attack_data, styles
                ))
        
        return elements
    
    @staticmethod
    def _build_summary_table(attacks, styles):
        elements = []
        title = Paragraph("Attack Summary", styles['CustomHeading2'])
        
        # Prepare summary data
        headers = ['Attack', 'Robustness', 'Accuracy', 'SSIM', 'Misclass.']
        table_data = [headers]
        
        for attack_name, attack_data in attacks.items():
            if attack_name != 'reference':
                row = [
                    attack_name.upper(),
                    AttacksSection._format_value(attack_data.get('robustness')),
                    AttacksSection._format_value(attack_data.get('accuracy')),
                    AttacksSection._format_value(attack_data.get('ssim')),
                    AttacksSection._format_value(attack_data.get('misclassification'))
                ]
                table_data.append(row)
        
        table = Table(table_data, colWidths=[80, 100, 80, 80, 80])
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), CorporateColors.RED),
            ('TEXTCOLOR', (0, 0), (-1, 0), CorporateColors.WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), CorporateColors.WHITE),
            ('TEXTCOLOR', (0, 1), (-1, -1), CorporateColors.DARK_GRAY),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [CorporateColors.WHITE, CorporateColors.VERY_LIGHT_GRAY]),
            
            # General styling
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        # Keep title and table together
        elements.append(KeepTogether([title, table]))
        elements.append(Spacer(1, 20))
        
        return elements
    
    @staticmethod
    def _build_attack_detail(attack_name, attack_data, styles):
        elements = []
        
        title = Paragraph(f"<b>{attack_name.upper()}</b> Attack Details", 
                         styles['CustomHeading2'])
        
        # Metrics table
        table_data = []
        metric_order = [
            'countsamples', 'accuracy', 'precision', 'robustness',
            'misclassification', 'f1score', 'expectedcalibrationerror',
            'meansquarecontingency', 'ssim'
        ]
        
        for metric in metric_order:
            if metric in attack_data:
                label = metric.replace('_', ' ').title()
                value = AttacksSection._format_value(attack_data[metric])
                table_data.append([label, value])
        
        if table_data:
            table = Table(table_data, colWidths=[200, 250])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), CorporateColors.VERY_LIGHT_GRAY),
                ('TEXTCOLOR', (0, 0), (0, -1), CorporateColors.DARK_GRAY),
                ('TEXTCOLOR', (1, 0), (1, -1), CorporateColors.BLACK),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, CorporateColors.LIGHT_GRAY),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            # Keep title and table together
            elements.append(KeepTogether([title, table]))
            elements.append(Spacer(1, 15))
        
        return elements
    
    @staticmethod
    def _format_value(value):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            if value < 0.01:
                return f"{value:.2e}"
            return f"{value:.4f}"
        return str(value)


class AdversarialReportGenerator:
    """Main report generator class"""
    
    def __init__(self, logo_path=None):
        """
        Initialize report generator
        
        Args:
            logo_path: Path to corporate logo PNG file
        """
        self.logo_path = logo_path
        self.styles = ReportStyles.get_styles()
    
    def generate(self, data, output_path='adversarial_report.pdf'):
        """
        Generate PDF report from JSON data
        
        Args:
            data: Dictionary with model and attack information
            output_path: Output PDF file path
        """
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=80,
            bottomMargin=60
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._build_title_page(data))
        
        # Model information
        story.extend(ModelInfoSection.build(data, self.styles))
        
        # Global metrics
        story.extend(MetricsSection.build(data, self.styles))
        
        # Attacks analysis
        story.extend(AttacksSection.build(data, self.styles))
        
        # Build PDF
        doc.build(
            story,
            onFirstPage=HeaderFooter(self.logo_path),
            onLaterPages=HeaderFooter(self.logo_path)
        )
        
        print(f"Report generated: {output_path}")
    
    def _build_title_page(self, data):
        elements = []
        
        # Title
        title = Paragraph(
            "TITANN Model Trustworthy Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        
        # Subtitle with model name
        model_name = data.get('info', {}).get('name', 'Unknown Model')
        subtitle = Paragraph(
            f"<font color='#{CorporateColors.RED.hexval()[2:]}'>Model: {model_name}</font>",
            self.styles['CustomHeading2']
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 10))
        
        # Report metadata
        metadata_text = f"""
        <font size=10>
        <b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        <b>Generated By:</b> TITANN framwework<br/>
        </font>
        """
        elements.append(Paragraph(metadata_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 30))
        
        # Executive summary
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading2']))
        
        summary_text = f"""
        This report provides a comprehensive analysis of the adversarial robustness 
        of the {model_name} model. The analysis includes multiple attack and evaluates the model's resilience against adversarial perturbations.
        """
        elements.append(Paragraph(summary_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 20))
        
        return elements