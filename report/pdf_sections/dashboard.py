"""Print adaptation of the frontend security dashboard."""
import json
import math
import re
from datetime import datetime
from io import BytesIO
from numbers import Real
from xml.sax.saxutils import escape

from matplotlib.figure import Figure
from reportlab.graphics.shapes import Circle, Drawing, String
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    Flowable, Image, KeepInFrame, KeepTogether, LongTable, PageBreak, Paragraph, Spacer, Table, TableStyle,
)
from report.corporate_colors import CorporateColors as C
from report.report_style import text_style


def as_dict(value):
    return value.model_dump(mode='json') if hasattr(value, 'model_dump') else value


def numeric(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def label(key):
    aliases = {'num_classes': 'Classes', 'num_samples': 'Samples',
               'input_dimensionality': 'Input shape', 'model_type': 'Framework',
               'source_path': 'Source', 'repository': 'Source', 'weights': 'File size',
               'std': 'Std. dev.', 'size': 'Resize'}
    return aliases.get(key, re.sub(r'([a-z])([A-Z])', r'\1 \2', key).replace('_', ' ').title())


def format_value(value, key='', metric=False):
    if value is None or value == '':
        return 'N/A'
    if isinstance(value, (list, tuple)):
        return ' × '.join(format_value(item, metric=metric) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    if numeric(value):
        if metric:
            return f'{value:.3f}'
        if key == 'parameters':
            if value >= 1e9:
                return f'{value / 1e9:.2f}B'
            if value >= 1e6:
                return f'{value / 1e6:.2f}M'
        return f'{value:,}'
    if key == 'date':
        try:
            date = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            months = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
            return f'{date.day:02d} {months[date.month - 1]} {date.year}'
        except ValueError:
            pass
    text = str(value)
    if key in {'repository', 'source_path', 'path'}:
        text = re.sub(r'[\x00-\x1f\x7f]', '', text).replace('\\', '/')
    return text


def criticality(risk):
    percentage = (risk * 100 if 0 <= risk <= 1 else risk) if numeric(risk) else 0
    percentage = min(100, max(0, percentage))
    for threshold, name, color in [(75, 'Critical', '#B42338'), (50, 'High', '#B45309'),
                                   (25, 'Medium', '#8A6800'), (0, 'Low', '#087F5B')]:
        if percentage >= threshold:
            return name, HexColor(color)


def risk_badge(risk):
    name, color = criticality(risk)
    badge = Drawing(72, 12)
    badge.add(Circle(4, 6, 2.5, fillColor=color, strokeColor=None))
    badge.add(String(12, 3, name, fontName='Helvetica-Bold', fontSize=8, fillColor=color))
    return badge


class MetricCard(Flowable):
    """Compact card; scalar numbers always use the same fixed height."""

    def __init__(self, key, value, width, metric=True):
        super().__init__()
        self.width = width
        self.scalar = numeric(value)
        self.caption = Paragraph(escape(label(key).upper()), text_style('CardLabel', 6, C.MUTED, True))
        self.vector = isinstance(value, (list, tuple)) and all(numeric(v) for v in value)
        values = value if self.vector else [value]
        self.values = [Paragraph(escape(format_value(v, metric=metric)), text_style(
            'CardValue', 7 if self.vector or not numeric(v) else 9, bold=True)) for v in values]

    def wrap(self, availWidth, availHeight):
        if self.scalar:
            self.height = 36
            self.fitted_caption = KeepInFrame(self.width - 12, 12, [self.caption], mode='shrink')
            self.fitted_value = KeepInFrame(self.width - 12, 12, self.values, mode='shrink')
            for content in (self.fitted_caption, self.fitted_value):
                content.wrapOn(self.canv, self.width - 12, 12)
            return self.width, self.height
        _, self.caption_height = self.caption.wrap(self.width - 12, availHeight)
        self.heights = [p.wrap(self.width - 18, availHeight)[1] for p in self.values]
        self.height = max(32, 15 + self.caption_height + sum(self.heights) +
                          (6 * len(self.values) if self.vector else 0))
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        canvas.setFillColor(C.SURFACE)
        canvas.setStrokeColor(C.BORDER)
        canvas.setLineWidth(.6)
        canvas.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=1)
        if self.scalar:
            self.fitted_caption.drawOn(canvas, 6, self.height - 5 - self.fitted_caption.height)
            self.fitted_value.drawOn(canvas, 6, 5)
            return
        y = self.height - 6 - self.caption_height
        self.caption.drawOn(canvas, 6, y)
        y -= 3
        for paragraph, height in zip(self.values, self.heights):
            y -= height
            if self.vector:
                canvas.setFillColor(C.BORDER)
                canvas.roundRect(6, y - 2, self.width - 12, height + 4, 3, fill=1, stroke=0)
            paragraph.drawOn(canvas, 9 if self.vector else 6, y)
            y -= 6 if self.vector else 0


class SectionHeading(Flowable):
    """Small vector icons stay crisp in print and do not depend on icon fonts."""

    def __init__(self, title, style):
        super().__init__()
        self.title = title
        self.text = Paragraph(escape(title), style)
        self.spaceBefore = style.spaceBefore
        self.spaceAfter = style.spaceAfter
        self.keepWithNext = True

    def wrap(self, width, height):
        _, self.height = self.text.wrap(width - 32, height)
        self.width = width
        return width, self.height

    def draw(self):
        self.text.drawOn(self.canv, 32, 0)
        c = self.canv
        c.saveState()
        c.translate(0, max(0, (self.height - 20) / 2))
        c.setStrokeColor(C.CYAN)
        c.setLineWidth(1.4)
        if self.title == 'Metrics':
            c.arc(1, 1, 21, 21, 0, 180)
            c.line(2, 10, 20, 10)
            c.line(11, 10, 16, 17)
        elif self.title == 'Benchmarking':
            path = c.beginPath()
            path.moveTo(5, 21)
            path.lineTo(17, 21)
            path.lineTo(16, 12)
            path.curveTo(16, 5, 6, 5, 6, 12)
            path.close()
            c.drawPath(path)
            c.line(11, 7, 11, 1)
            c.line(6, 1, 16, 1)
            c.arc(0, 12, 9, 21, 90, 180)
            c.arc(13, 12, 22, 21, 270, 180)
        elif self.title in {'Vulnerability Assessment', 'Attack details'}:
            path = c.beginPath()
            path.moveTo(11, 22)
            path.lineTo(20, 18)
            path.lineTo(19, 9)
            path.curveTo(18, 5, 14, 2, 11, 0)
            path.curveTo(8, 2, 4, 5, 3, 9)
            path.lineTo(2, 18)
            path.close()
            c.drawPath(path)
        else:
            for x, y in [(6, 12), (0, 0), (12, 0)]:
                c.roundRect(x, y, 9, 9, 1, stroke=1, fill=0)
        c.restoreState()


class Dashboard:
    def __init__(self, style, benchmark=None, excluded_metrics=None):
        self.style = style
        self.width = style.pagesize[0] - style.leftMargin - style.rightMargin
        self.benchmark = benchmark
        self.excluded_metrics = set(excluded_metrics or [])
        self.body = text_style('Body', 9)
        self.small = text_style('Label', 7, C.MUTED, True)
        self.mono = text_style('Path', 8)
        self.mono.fontName = 'Courier'

    def paragraph(self, value, style=None):
        return Paragraph(escape(str(value)), style or self.body)

    def heading(self, title):
        return SectionHeading(title, self.style.section_title_style)

    def table(self, rows, widths, header=True):
        table = LongTable(rows, colWidths=widths, repeatRows=1 if header else 0,
                          hAlign='LEFT', splitInRow=1)
        commands = [
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, -1), C.SURFACE),
            ('LINEBELOW', (0, 0), (-1, -1), .4, C.BORDER),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 11),
        ]
        if header:
            commands.append(('BACKGROUND', (0, 0), (-1, 0), C.BACKGROUND))
        table.setStyle(TableStyle(commands))
        return table

    def info(self, info):
        cells = []
        for key, value in info.items():
            if key in {'id', 'image', 'transformation'}:
                continue
            cells.append([self.paragraph(label(key).upper(), self.small), Spacer(1, 5),
                          self.paragraph(format_value(value, key), self.mono if key in
                                                                                {'repository', 'source_path',
                                                                                 'path'} else self.body)])
        out = [self.heading('Model information')]
        if cells:
            rows = [cells[i:i + 3] + [''] * (3 - len(cells[i:i + 3])) for i in range(0, len(cells), 3)]
            grid = self.table(rows, [self.width / 3] * 3, header=False)
            grid.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), C.BACKGROUND),
                                      ('LINEABOVE', (0, 0), (-1, 0), .5, C.BORDER),
                                      ('LEFTPADDING', (0, 0), (-1, -1), 0)]))
            out.append(grid)
        preprocessing = as_dict(info.get('transformation'))
        if isinstance(preprocessing, dict):
            rows = [[self.paragraph('Input preprocessing', self.style.section_subtitle_style), '']]
            rows += [[self.paragraph(label(k)), self.paragraph(format_value(v), self.mono)]
                     for k, v in preprocessing.items()]
            box = self.table(rows, [self.width * .35, self.width * .65])
            box.setStyle(TableStyle([('SPAN', (0, 0), (-1, 0)),
                                     ('BOX', (0, 0), (-1, -1), .5, C.BORDER)]))
            out += [Spacer(1, 16), KeepTogether([box])]
        return out + [Spacer(1, 22)]

    def metrics(self, metrics, parameters=False):
        items = list(metrics) if parameters else [
            (k, v) for k, v in metrics.items() if v is not None and k != 'confusion_matrix'
            and k not in self.excluded_metrics]
        out = []
        if not items and (parameters or metrics.get('confusion_matrix') is None):
            return [self.paragraph('No metrics computed.', self.style.section_description_style)]
        per_class = [(k, v) for k, v in items if not parameters and k.lower().startswith('class')
                     and isinstance(v, (list, tuple, dict))]
        class_keys = {k for k, _ in per_class}
        items = [(k, v) for k, v in items if k not in class_keys]
        # Large arrays/structured payloads need splittable rows, not an unbreakable card.
        oversized = [(k, v) for k, v in items if len(format_value(v, metric=True)) > 240
                     or isinstance(v, (list, tuple)) and len(v) > 12]
        items = [(k, v) for k, v in items if k not in {key for key, _ in oversized}]
        width = 82
        gap = 6
        columns = max(1, int((self.width + gap) / (width + gap)))
        for start in range(0, len(items), columns):
            row = []
            for key, value in items[start:start + columns]:
                if row:
                    row.append('')
                row.append(MetricCard(key, value, width, metric=not parameters))
            row += [''] * (columns * 2 - 1 - len(row))
            widths = [width if i % 2 == 0 else gap for i in range(columns * 2 - 1)]
            grid = Table([row], colWidths=widths, hAlign='LEFT', style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0)]))
            out += [grid, Spacer(1, gap)]
        for key, value in oversized:
            if parameters:
                card = self.table([[[self.paragraph(label(key).upper(), self.small),
                                    self.paragraph(format_value(value))]]], [self.width], header=False)
                card.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), .6, C.BORDER)]))
                out.extend([card, Spacer(1, gap)])
                continue
            out.append(self.paragraph(label(key), self.style.section_subtitle_style))
            values = value if isinstance(value, (list, tuple)) else [value]
            out.append(self.table([[self.paragraph(format_value(v, metric=True))] for v in values],
                                  [self.width], header=False))
            out.append(Spacer(1, gap))
        for key, value in per_class:
            out.extend(self.class_metric(key, value))
        if not parameters and 'confusion_matrix' not in self.excluded_metrics:
            out.extend(self.confusion_matrix(metrics.get('confusion_matrix')))
        return out

    def confusion_matrix(self, matrix):
        if matrix is None:
            return []
        title = self.paragraph('Confusion Matrix', self.style.section_subtitle_style)
        if (not isinstance(matrix, (list, tuple)) or not matrix
                or any(not isinstance(row, (list, tuple)) or len(row) != len(matrix)
                       or any(not numeric(v) or v < 0 for v in row) for row in matrix)):
            return [title, self.paragraph('Confusion matrix unavailable: invalid matrix.')]
        figure = Figure(figsize=(7, 3.8), facecolor='white')
        ax = figure.subplots()
        self.style_axes(ax)
        ax.grid(False)
        plot = ax.imshow(matrix, cmap='Blues', vmin=0, interpolation='nearest')
        ax.set(xlabel='Predicted class', ylabel='True class')
        if len(matrix) <= 20:
            ax.set_xticks(range(len(matrix)))
            ax.set_yticks(range(len(matrix)))
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if len(matrix) <= 10:
            maximum = max(max(row) for row in matrix)
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    ax.text(j, i, f'{value:g}', ha='center', va='center', fontsize=8,
                            color='white' if value > maximum / 2 else '#172638')
        figure.colorbar(plot, ax=ax).set_label('Value (as reported)')
        figure.tight_layout()
        return [title, self.figure_image(figure), Spacer(1, 12)]

    @staticmethod
    def style_axes(ax):
        ax.set_facecolor('white')
        ax.tick_params(colors='#52677D', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#D5DFE8')
        ax.xaxis.label.set_color('#40546A')
        ax.yaxis.label.set_color('#40546A')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#D5DFE8', linewidth=.5)

    def figure_image(self, figure):
        stream = BytesIO()
        figure.savefig(stream, format='png', dpi=150)
        stream.seek(0)
        width, height = figure.get_size_inches()
        return Image(stream, width=self.width, height=self.width * height / width)

    def class_metric(self, key, value):
        # List positions are the zero-based class IDs returned by nn_trust.
        pairs = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
        if not pairs:
            return [self.paragraph(label(key), self.style.section_subtitle_style),
                    self.paragraph('No per-class values available.'), Spacer(1, 12)]
        finite = [v for _, v in pairs if numeric(v)]
        lower, upper = min([0] + finite), max([0] + finite)
        padding = (upper - lower) * .1 or 1
        # One chart per metric, regardless of the number of classes.
        figure = Figure(figsize=(7, 2.1), facecolor='white')
        ax = figure.subplots()
        self.style_axes(ax)
        positions = list(range(len(pairs)))
        ax.bar(positions, [v if numeric(v) else float('nan') for _, v in pairs],
               color='#0891B2', width=.7)
        for i, (_, v) in enumerate(pairs):
            if not numeric(v):
                ax.text(i, 0, 'N/A', ha='center', va='bottom', fontsize=7, color='#52677D')
        ax.set_xticks([])
        ax.set(xlabel='Class', ylabel=label(key),
               ylim=(lower - padding if lower < 0 else 0, upper + padding))
        figure.tight_layout()
        return [self.paragraph(label(key), self.style.section_subtitle_style),
                self.figure_image(figure), Spacer(1, 12)]

    def benchmark_rows(self, metric):
        benchmarks = self.benchmark or []
        # Compatibility with the original metric -> scores constructor format.
        if isinstance(benchmarks, dict):
            if isinstance(benchmarks.get(metric), (list, tuple)):
                benchmarks = [{'name': f'Benchmark {i + 1}', 'metrics': {metric: v}}
                              for i, v in enumerate(benchmarks[metric])]
            else:
                benchmarks = [dict(as_dict(v), name=as_dict(v).get('name', k))
                              for k, v in benchmarks.items() if isinstance(as_dict(v), dict)]
        rows = []
        for item in benchmarks:
            item = as_dict(item)
            value = as_dict(item.get('metrics', {})).get(metric)
            if numeric(value):
                rows.append((item.get('name') or 'N/A', item.get('param', item.get('parameters')), value))
        return sorted(rows, key=lambda row: row[2], reverse=True)

    def benchmarking(self, metrics):
        excluded = {'params', 'name', 'confusion_matrix', 'total benchmarks'}
        valid = [k for k, v in metrics.items() if k not in excluded and not k.endswith('_rank') and numeric(v)]
        out = [self.heading('Benchmarking')]
        if not valid:
            return out + [self.paragraph('No numeric metrics available for benchmarking.')]
        metric = 'accuracy' if 'accuracy' in valid else valid[0]
        out.append(self.paragraph(f'Metric selected: {label(metric)}', self.style.section_description_style))
        rows = self.benchmark_rows(metric)
        if not rows:
            return out + [self.paragraph('No benchmark models available.'), Spacer(1, 22)]
        figure = Figure(figsize=(7, 3.3), facecolor='white')
        ax = figure.subplots()
        self.style_axes(ax)
        points = [row for row in rows if numeric(row[1]) and row[1] >= 0]
        if points:
            ax.scatter([r[1] for r in points], [r[2] for r in points], color='#55A9F5', s=36, zorder=3)
        else:
            ax.text(.5, .5, 'Benchmark parameter counts unavailable', transform=ax.transAxes,
                    ha='center', color='#52677D', fontsize=9)
        ax.axhline(metrics[metric], color='#F15B68', label='Reference', linewidth=1.5)
        ax.set(xlabel='Parameters', ylabel='Value', ylim=(0, 1))
        ax.legend(facecolor='white', edgecolor='#D5DFE8', labelcolor='#172638', fontsize=8)
        figure.tight_layout()
        out += [self.figure_image(figure), Spacer(1, 12),
                self.paragraph(f'Leaderboard: {label(metric)}', self.style.section_subtitle_style)]
        table_rows = [[self.paragraph(h, self.small) for h in ('Rank', 'Model', 'Value')]]
        table_rows += [[self.paragraph(i), self.paragraph(name), self.paragraph(f'{score:.3f}')]
                       for i, (name, _, score) in enumerate(rows, 1)]
        return out + [self.table(table_rows, [self.width * .12, self.width * .65, self.width * .23]), Spacer(1, 22)]

    def vulnerabilities(self, attacks):
        count = len(attacks)
        out = [self.heading('Vulnerability Assessment'), self.paragraph(
            f'{count} {"attack" if count == 1 else "attacks"} tested · Criticality based on measured risk.',
            self.style.section_description_style)]
        rows = [[self.paragraph(h, self.small) for h in ('Attack', 'Category', 'Criticality')]]
        for key, attack in attacks.items():
            attack = as_dict(attack)
            rows.append([self.paragraph(attack.get('name') or key),
                         self.paragraph(attack.get('category') or 'Evasion'),
                         risk_badge(as_dict(attack.get('metrics', {})).get('risk'))])
        if not attacks:
            rows.append([self.paragraph('No vulnerability assessments available.'), '', ''])
        table = self.table(rows, [self.width * .55, self.width * .22, self.width * .23])
        table.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 5),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        if not attacks:
            table.setStyle(TableStyle([('SPAN', (0, 1), (-1, 1))]))
        return out + [table]

    def build(self, data, include_attack_details=True, examples_root=None):
        data = as_dict(data)
        metrics = as_dict(data['metrics'])
        attacks = data.get('attacks', {})
        story = [Paragraph('Security Report', self.style.pdf_title_style),
                 self.paragraph('Results from the benchmark evaluation.', self.style.section_description_style),
                 Spacer(1, 12)]
        story += self.info(as_dict(data['info']))
        story += [self.heading('Metrics')] + self.metrics(metrics) + [Spacer(1, 22)]
        story += self.benchmarking(metrics)
        story += self.vulnerabilities(attacks)
        if include_attack_details and attacks:
            for key, attack in attacks.items():
                story += [PageBreak(), self.heading('Attack details')]
                attack = as_dict(attack)
                story += [self.paragraph(attack.get('name') or key, self.style.section_subtitle_style)]
                story += self.metrics(as_dict(attack.get('metrics', {})))
                story.append(self.paragraph('Parameters', self.style.section_subtitle_style))
                parameters = []
                for parameter in attack.get('parameters', []):
                    parameter = as_dict(parameter)
                    parameters.append((parameter.get('name') or parameter['id'], parameter.get('value')))
                if parameters:
                    story.extend(self.metrics(parameters, parameters=True))
                else:
                    story.append(self.paragraph('No parameters were saved for this attack.'))
                story.append(Spacer(1, 22))
                from report.pdf_sections.examples import build_examples
                story.extend(build_examples(self, examples_root or data.get('repository'),
                                            key, as_dict(data['info']).get('transformation')))
        return story
