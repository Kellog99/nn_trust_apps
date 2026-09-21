"""Render existing attack artifacts from the report repository."""
from pathlib import Path
from itertools import zip_longest
from pickle import UnpicklingError

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread
from reportlab.platypus import Spacer


def _tensor_image(value, transformation=None):
    if value is None:
        return None
    array = value.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (1, 3, 4)):
        raise ValueError('The saved input is not an image.')
    transformation = transformation or {}
    mean, std = transformation.get('mean'), transformation.get('std')
    if array.ndim == 3 and mean is not None and std is not None and len(mean) == len(std) == array.shape[-1]:
        array = array * np.asarray(std) + np.asarray(mean)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def build_examples(dashboard, repository, attack_id, transformation=None):
    out = [dashboard.paragraph('Examples', dashboard.style.section_subtitle_style)]
    if not repository:
        return out + [dashboard.paragraph('No saved examples available.')]
    root = Path(repository).expanduser().resolve()
    folder = (root / attack_id).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        return out + [dashboard.paragraph('No saved examples available.')]
    try:
        groups = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'} or not path.resolve().is_relative_to(root):
                continue
            sample, _, kind = path.stem.rpartition('_')
            if sample and kind in {'original', 'pert', 'adv'}:
                groups.setdefault(sample, {})[kind] = imread(path)
        checkpoint = folder / 'log.pth'
        if not groups and checkpoint.is_file() and checkpoint.resolve().is_relative_to(root):
            import torch
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            for i, (original, adversarial) in enumerate(zip_longest(
                    saved.get('original_input', []), saved.get('adversarial_input', []))):
                original = _tensor_image(original, transformation)
                adversarial = _tensor_image(adversarial, transformation)
                perturbation = None
                if original is not None and adversarial is not None and original.shape == adversarial.shape:
                    perturbation = np.abs(adversarial - original)
                    maximum = perturbation.max()
                    if maximum > 0:
                        perturbation = perturbation / maximum
                groups[str(i)] = {'original': original, 'adv': adversarial, 'pert': perturbation}
        if not groups:
            return out + [dashboard.paragraph('No saved examples available.')]
        for sample, group in groups.items():
            figure = Figure(figsize=(7, 2.3), facecolor='white')
            axes = figure.subplots(1, 3)
            for ax, kind, title in zip(axes, ('original', 'pert', 'adv'),
                                       ('Original', 'Perturbation', 'Adversarial')):
                value = group.get(kind)
                if value is not None:
                    ax.imshow(np.clip(value, 0, 1) if np.issubdtype(value.dtype, np.floating) else value,
                              cmap='gray')
                else:
                    ax.text(.5, .5, 'N/A', ha='center', va='center')
                ax.set_title(title, fontsize=9, color='#40546A')
                ax.axis('off')
            figure.tight_layout()
            out.extend([dashboard.paragraph(f'Sample {sample}', dashboard.style.section_subtitle_style),
                        dashboard.figure_image(figure), Spacer(1, 8)])
        if checkpoint.is_file() and not any(folder.glob('*_pert.*')):
            out.append(dashboard.paragraph('Perturbation: absolute difference, scaled for visibility.'))
        return out
    except (OSError, ValueError, TypeError, RuntimeError, AttributeError, UnpicklingError, EOFError) as exc:
        return out + [dashboard.paragraph(f'Saved examples could not be rendered ({type(exc).__name__}).')]
