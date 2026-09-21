from copy import deepcopy
import shutil
import subprocess

import pytest
from reportlab.lib.pagesizes import A4, landscape

from report import AdversarialReportGenerator, AdversarialReportStyle
from report.pdf_sections.dashboard import criticality, format_value


@pytest.mark.parametrize('risk,expected', [
    (None, 'Low'), ('75', 'Low'), (True, 'Low'), (float('nan'), 'Low'),
    (float('inf'), 'Low'), (-5, 'Low'), (0.249, 'Low'), (.25, 'Medium'),
    (.5, 'High'), (.75, 'Critical'), (1, 'Critical'), (25, 'Medium'),
    (50, 'High'), (75, 'Critical'), (140, 'Critical'),
])
def test_risk_thresholds(risk, expected):
    assert criticality(risk)[0] == expected


def test_formatting():
    assert format_value(25300000, 'parameters') == '25.30M'
    assert format_value([.1, .25], metric=True) == '0.100 × 0.250'
    assert format_value('2026-09-21', 'date') == '21 Sep 2026'


@pytest.mark.parametrize('pagesize', [A4, landscape(A4)])
@pytest.mark.parametrize('details', [False, True])
def test_multipage_report(tmp_path, pagesize, details):
    data = {
        'info': {'id': 'hidden-id', 'name': 'Model <A> & B', 'parameters': 25300000,
                 'repository': '/models/' + 'long_path_' * 30,
                 'transformation': {'mean': [.1, .2, .3], 'std': [.2, .3, .4]}},
        'metrics': {'accuracy': .923, 'precision': .91, 'params': 25300000,
                    'vector': [.1, .2, .3], 'large_vector': list(range(100)),
                    'details': {'source': '<custom>'}, 'confusion_matrix': [[3, 1], [1, 4]]},
        'attacks': {str(i): {'name': f'Attack {i}', 'category': 'Poisoning',
                            'metrics': {'risk': i / 100},
                            'parameters': [{'id': 'eps', 'value': .03}]} for i in range(45)},
    }
    original = deepcopy(data)
    benchmark = [
        {'name': 'Other model', 'param': 1000, 'metrics': {'accuracy': .8}},
        {'name': 'Best model', 'param': 2000, 'metrics': {'accuracy': .95}},
        {'name': 'Invalid model', 'param': 3000, 'metrics': {'accuracy': [1, 2]}},
    ]
    generator = AdversarialReportGenerator(benchmark=benchmark,
        style=AdversarialReportStyle(pagesize=pagesize), include_attack_details=details)
    assert [row[0] for row in generator.dashboard.benchmark_rows('accuracy')] == ['Best model', 'Other model']
    path = tmp_path / 'report.pdf'
    generator.generate(data, path)
    assert path.read_bytes().startswith(b'%PDF')
    assert data == original
    if shutil.which('pdftotext'):
        text = subprocess.check_output(['pdftotext', str(path), '-'], text=True)
        for expected in ['Model <A> & B', 'Security Report', 'Benchmarking', 'Leaderboard:',
                         'Vulnerability Assessment', 'Poisoning', 'Attack 44']:
            assert expected in text
        assert ('Attack details' in text) == details
        assert 'hidden-id' not in text
        assert 'Confusion Matrix' in text
        if details:
            pages = [page for page in text.split('\f') if 'Attack details' in page]
            assert len(pages) == 45


def test_empty_report_and_legacy_benchmarks(tmp_path):
    generator = AdversarialReportGenerator(benchmark={'accuracy': [.9, .5, None]})
    assert [r[2] for r in generator.dashboard.benchmark_rows('accuracy')] == [.9, .5]
    generator.generate({'info': {}, 'metrics': {}, 'attacks': {}}, tmp_path / 'empty.pdf')


def test_class_charts_preserve_labels_values_and_scale(monkeypatch):
    from reportlab.platypus import Spacer
    dashboard = AdversarialReportGenerator().dashboard
    figures = []

    def capture(figure):
        figures.append(figure)
        return Spacer(1, 1)

    monkeypatch.setattr(dashboard, 'figure_image', capture)
    dashboard.metrics({'ClassAccuracy': {'cat': .8, 'dog': .4, 'bird': None}})
    axes = figures[0].axes[0]
    assert len(axes.get_xticks()) == 0
    assert [bar.get_height() for bar in axes.patches[:2]] == [.8, .4]
    assert any(text.get_text() == 'N/A' for text in axes.texts)
    assert figures[0].get_facecolor() == (1, 1, 1, 1)
    assert axes.get_facecolor() == (1, 1, 1, 1)
    figures.clear()
    dashboard.metrics({'classrobustness': list(range(30))})
    assert len(figures) == 1
    axes = figures[0].axes[0]
    assert [bar.get_height() for bar in axes.patches] == list(range(30))
    assert len(axes.get_xticks()) == 0


def test_default_report_includes_attack_details_and_class_charts(tmp_path):
    from reportlab.platypus import Image
    data = {'info': {'name': 'Class report'},
            'metrics': {'ClassAccuracy': [.9, .8] * 20},
            'attacks': {'fgsm': {'name': 'FGSM',
                                'metrics': {'ClassRobustness': [.5, .2] * 20, 'risk': .7},
                                'parameters': [{'id': 'epsilon', 'value': .031}]}}}
    generator = AdversarialReportGenerator()
    story = generator.build_story(data)
    assert sum(isinstance(item, Image) for item in story) == 2
    path = tmp_path / 'classes.pdf'
    generator.generate(data, path)
    if shutil.which('pdftotext'):
        text = subprocess.check_output(['pdftotext', str(path), '-'], text=True)
        for expected in ['Attack details', 'FGSM', 'EPSILON', '0.031',
                         'Class Accuracy', 'Class Robustness']:
            assert expected in text


def test_matrix_values_and_parameter_cards(monkeypatch):
    from reportlab.platypus import Spacer, Table
    from report.pdf_sections.dashboard import MetricCard
    dashboard = AdversarialReportGenerator().dashboard
    figures = []
    monkeypatch.setattr(dashboard, 'figure_image', lambda fig: figures.append(fig) or Spacer(1, 1))
    matrix = [[0, 0], [2, 8]]
    dashboard.metrics({'confusion_matrix': matrix})
    assert figures[0].axes[0].images[0].get_array().tolist() == matrix
    assert matrix == [[0, 0], [2, 8]]
    cards = dashboard.metrics([('epsilon', .00001), ('enabled', False), ('class_count', [1, 2])], parameters=True)
    grid = next(item for item in cards if isinstance(item, Table))
    cells = [cell for cell in grid._cellvalues[0] if isinstance(cell, MetricCard)]
    assert len(cells) == 3
    assert cells[0].values[0].getPlainText() == '1e-05'
    assert cells[1].values[0].getPlainText() == 'False'
    from io import BytesIO
    from reportlab.pdfgen.canvas import Canvas
    canvas = Canvas(BytesIO())
    for title, value in [('accuracy', .9), ('a_long_metric_name_that_wraps', 123456789.123)]:
        card = MetricCard(title, value, 82)
        assert card.wrapOn(canvas, 82, 500) == (82, 36)
        card.drawOn(canvas, 0, 0)


@pytest.mark.parametrize('storage', ['checkpoint', 'images'])
def test_saved_examples(tmp_path, storage):
    import torch
    from PIL import Image as PILImage
    from reportlab.platypus import Image
    from report.pdf_sections.examples import build_examples
    folder = tmp_path / 'fgsm'
    folder.mkdir()
    if storage == 'checkpoint':
        torch.save({'original_input': [torch.zeros(3, 8, 8)],
                    'adversarial_input': [torch.ones(3, 8, 8) * .1]}, folder / 'log.pth')
    else:
        for kind in ['original', 'adv']:
            PILImage.new('RGB', (8, 8)).save(folder / f'0_{kind}.png')
    dashboard = AdversarialReportGenerator().dashboard
    assert sum(isinstance(item, Image) for item in build_examples(dashboard, tmp_path, 'fgsm')) == 1
    data = {'info': {}, 'metrics': {}, 'repository': str(tmp_path),
            'attacks': {'fgsm': {'name': 'FGSM', 'metrics': {'confusion_matrix': [[1, 0], [0, 1]]},
                                'parameters': [{'id': 'epsilon', 'value': .1}]}}}
    AdversarialReportGenerator().generate(data, tmp_path / 'examples.pdf')
