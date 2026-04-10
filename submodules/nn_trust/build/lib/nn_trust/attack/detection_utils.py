import torch
import torchvision


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def x1y1wh2xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, w, h = boxes.unbind(1)
    x = x1 + w / 2
    y = y1 + h / 2
    return torch.stack([x, y, w, h], dim=1)


def xywh2x1y1wh(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(1)
    x1 = x - w / 2
    y1 = y - h / 2
    return torch.stack([x1, y1, w, h], dim=1)


def relativize_(
        boxes: torch.Tensor,
        w: float,
        h: float,
) -> torch.Tensor:
    boxes[..., torch.tensor([0, 2])] /= w
    boxes[..., torch.tensor([1, 3])] /= h
    return boxes


def relativize(
        boxes: torch.Tensor,
        w: float,
        h: float,
) -> torch.Tensor:
    return relativize_(boxes.clone(), w, h)


def absolutize_(
        boxes: torch.Tensor,
        w: float,
        h: float,
) -> torch.Tensor:
    boxes[..., torch.tensor([0, 2])] *= w
    boxes[..., torch.tensor([1, 3])] *= h
    return boxes


def absolutize(
        boxes: torch.Tensor,
        w: float,
        h: float,
) -> torch.Tensor:
    return absolutize_(boxes.clone(), w, h)


def nms(
        y: dict[str, torch.Tensor],
        iou_threshold: float,
        score_threshold: float,
) -> list[dict[str, torch.Tensor]]:
    """
    Given a object detection prediction, return a post nms output.
    """
    boxes = y["boxes"]
    scores = y["scores"]
    cls_scores = y["cls_scores"]

    preds = []

    for i in range(boxes.size(0)):  # Iterate over the batch
        # Extract boxes, scores, and cls_scores for the current batch item
        curr_boxes = boxes[i]
        curr_scores = scores[i]
        curr_cls_scores = cls_scores[i]

        # Convert to (x1, y1, x2, y2) format
        xyxy_boxes = xywh2xyxy(curr_boxes)

        # Apply NMS
        idxs = torchvision.ops.nms(xyxy_boxes, curr_scores, iou_threshold)
        curr_boxes = curr_boxes[idxs]
        curr_scores = curr_scores[idxs]
        curr_cls_scores = curr_cls_scores[idxs]

        # Filter by score
        idxs = curr_scores >= score_threshold
        curr_boxes = curr_boxes[idxs]
        curr_scores = curr_scores[idxs]
        curr_cls_scores = curr_cls_scores[idxs]

        # Get the predicted labels
        curr_labels = torch.argmax(curr_cls_scores, dim=1)

        # Append the results for the current batch item
        preds.append(
            dict(boxes=curr_boxes,
                 scores=curr_scores,
                 cls_scores=curr_cls_scores,
                 labels=curr_labels)
        )

    return preds
