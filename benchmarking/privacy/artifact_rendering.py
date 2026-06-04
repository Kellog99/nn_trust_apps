"""Rendering helpers for human-readable privacy artifacts.

Used to format and serialize reconstructed image tensors into HTML galleries 
and base64-encoded PNG payloads for display on the frontend.
"""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from typing import Any

import torch
from torchvision.transforms.functional import to_pil_image


def _coerce_image_tensor(image: Any) -> torch.Tensor:
    """Normalize one reconstruction tensor into a displayable ``[C,H,W]`` image."""
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    image = image.detach().cpu().float()

    if image.ndim == 1:
        side = int(round(float(image.numel()) ** 0.5))
        if side * side != int(image.numel()):
            raise ValueError(
                "Flat reconstruction tensors must have a perfect-square number of elements to render as images, "
                f"got {int(image.numel())}."
            )
        image = image.view(1, side, side)
    elif image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[0] in (1, 3):
            pass
        elif image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(
                "Three-dimensional reconstruction tensors must be channel-first or channel-last with 1 or 3 channels, "
                f"got shape {tuple(image.shape)}."
            )
    else:
        raise ValueError(
            "Reconstruction gallery rendering expects tensors with 1, 2, or 3 dimensions, "
            f"got {image.ndim}."
        )

    if image.shape[0] not in (1, 3):
        raise ValueError(
            f"Reconstruction gallery rendering supports 1-channel or 3-channel images, got shape {tuple(image.shape)}."
        )

    image_min = float(image.min().item())
    image_max = float(image.max().item())
    if image_min < 0.0 or image_max > 1.0:
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = torch.zeros_like(image)

    return image.clamp(0.0, 1.0)


def encode_reconstruction_png_base64(image: Any) -> str:
    """Encode one reconstruction tensor as an inline PNG data URI payload."""
    normalized_image = _coerce_image_tensor(image)
    with BytesIO() as buffer:
        to_pil_image(normalized_image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _is_renderable_image_tensor(image: Any) -> bool:
    """Return whether one reconstruction tensor can be normalized into an image."""
    try:
        _coerce_image_tensor(image)
    except (TypeError, ValueError):
        return False
    return True


def can_render_reconstruction_record(record: dict[str, Any]) -> bool:
    """Return whether one reconstruction record can be rendered as an image."""
    try:
        image = record["x_recon"]
    except (KeyError, TypeError, ValueError):
        return False
    return _is_renderable_image_tensor(image)


def build_reconstruction_gallery_html(
    reconstruction_records: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str | None = None,
) -> str:
    """Build a self-contained HTML gallery for reconstruction outputs."""
    escaped_title = escape(title)
    subtitle_markup = ""
    if subtitle:
        subtitle_markup = f"<p class=\"subtitle\">{escape(subtitle)}</p>"

    # Determine best/worst indices for badge labeling.
    confidences = [float(r.get("confidence", 0.0)) for r in reconstruction_records]
    best_idx = max(range(len(confidences)), key=lambda i: confidences[i]) if confidences else -1
    worst_idx = min(range(len(confidences)), key=lambda i: confidences[i]) if confidences else -1

    cards: list[str] = []
    for index, record in enumerate(reconstruction_records, start=1):
        encoded_image = encode_reconstruction_png_base64(record["x_recon"])
        target_class = int(record["y_target"])
        confidence = float(record["confidence"])
        rank_label = ""
        if index - 1 == best_idx:
            rank_label = ' <span class="badge best">BEST</span>'
        elif index - 1 == worst_idx:
            rank_label = ' <span class="badge worst">WORST</span>'
        cards.append(
            """
            <article class=\"card\">
              <div class=\"image-wrapper\">
                <img alt=\"Reconstruction {index}\" src=\"data:image/png;base64,{encoded_image}\" />
              </div>
              <div class=\"card-body\">
                <h2>Target class {target_class}{rank_label}</h2>
                <dl>
                  <div><dt>Rank</dt><dd>{index}</dd></div>
                  <div><dt>Confidence</dt><dd>{confidence:.6f}</dd></div>
                </dl>
              </div>
            </article>
            """.format(
                index=index,
                encoded_image=encoded_image,
                target_class=target_class,
                confidence=confidence,
                rank_label=rank_label,
            )
        )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{escaped_title}</title>
    <style>
      :root {{
        color-scheme: light;
        --page-bg: #f4f6f8;
        --card-bg: #ffffff;
        --card-border: #d8dee4;
        --text: #1f2933;
        --muted: #52606d;
        --accent: #0b69a3;
      }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--page-bg);
        color: var(--text);
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 24px 48px;
      }}
      header {{
        margin-bottom: 24px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 1.9rem;
      }}
      .subtitle {{
        margin: 0;
        color: var(--muted);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 16px;
      }}
      .card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
      }}
      .image-wrapper {{
        background: #111827;
        padding: 16px;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 192px;
      }}
      img {{
        max-width: 100%;
        max-height: 320px;
        object-fit: contain;
        image-rendering: pixelated;
        background: #ffffff;
        border-radius: 8px;
      }}
      .card-body {{
        padding: 16px;
      }}
      .card-body h2 {{
        margin: 0 0 12px;
        font-size: 1rem;
      }}
      .badge {{
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 2px 8px;
        border-radius: 4px;
        vertical-align: middle;
        margin-left: 6px;
      }}
      .badge.best {{
        background: #d1fae5;
        color: #065f46;
      }}
      .badge.worst {{
        background: #fee2e2;
        color: #991b1b;
      }}
      dl {{
        margin: 0;
        display: grid;
        gap: 8px;
      }}
      dl div {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 0.95rem;
      }}
      dt {{
        color: var(--muted);
      }}
      dd {{
        margin: 0;
        font-variant-numeric: tabular-nums;
      }}
      footer {{
        margin-top: 24px;
        color: var(--muted);
        font-size: 0.9rem;
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>{escaped_title}</h1>
        {subtitle_markup}
      </header>
      <section class=\"grid\">{''.join(cards)}</section>
      <footer>Rendered from persisted reconstruction tensors.</footer>
    </main>
  </body>
</html>
"""


__all__ = [
    "build_reconstruction_gallery_html",
    "can_render_reconstruction_record",
    "encode_reconstruction_png_base64",
]
