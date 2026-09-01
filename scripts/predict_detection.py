from pathlib import Path
import torch
from torchvision.utils import save_image, draw_bounding_boxes
from torchvision.io import write_png

log_path = Path("benchmark_output/20260901T163641/dag/log.pth")

image_out_path = log_path.parent / "comparison.png"
pred_out_path = log_path.parent / "prediction_comparison.png"

data = torch.load(log_path, map_location="cpu", weights_only=False)

x = data["original_input"][0].clamp(0, 1)
x_adv = data["adversarial_input"][0].clamp(0, 1)

y_pred = data["original_prediction"][0]
y_pred_adv = data["adversarial_prediction"][0]

save_image([x, x_adv], image_out_path, nrow=2)

def xywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ],
        dim=-1,
    )

def filter_predictions(pred, score_threshold=0.25, top_k=20, label_filter=None):
    boxes = pred["boxes"].detach().cpu()
    labels = pred["labels"].detach().cpu()

    if "scores" in pred:
        scores = pred["scores"].detach().cpu()
        keep = scores >= score_threshold
    else:
        scores = None
        keep = torch.ones(len(labels), dtype=torch.bool)

    if label_filter is not None:
        keep = keep & (labels == label_filter)

    idx = torch.where(keep)[0]

    if scores is not None and idx.numel() > top_k:
        idx = idx[scores[idx].topk(top_k).indices]
    else:
        idx = idx[:top_k]

    out = {
        "boxes": boxes[idx],
        "labels": labels[idx],
    }

    if scores is not None:
        out["scores"] = scores[idx]

    return out


def draw_predictions(image, pred):
    pred = filter_predictions(
        pred,
        score_threshold=0.25,
        top_k=15,
        label_filter=None,
    )

    image_uint8 = (image.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    _, h, w = image_uint8.shape

    boxes = pred["boxes"]

    boxes = xywh_to_xyxy(boxes)
    if boxes.numel() > 0 and boxes.max() <= 1.5:
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

    labels_tensor = pred["labels"]

    if "scores" in pred:
        scores = pred["scores"]
        labels = [
            f"{int(label)}:{float(score):.2f}"
            for label, score in zip(labels_tensor, scores)
        ]
    else:
        labels = [str(int(label)) for label in labels_tensor]

    return draw_bounding_boxes(
        image_uint8,
        boxes,
        labels=labels,
        width=2,
        colors="red",
    )

x_with_pred = draw_predictions(x, y_pred)
x_adv_with_pred = draw_predictions(x_adv, y_pred_adv)

comparison = torch.cat([x_with_pred, x_adv_with_pred], dim=2)
write_png(comparison, str(pred_out_path))

print(image_out_path)
print(pred_out_path)