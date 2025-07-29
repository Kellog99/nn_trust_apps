import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import torchvision.transforms.functional as F


class MetricsCalculator:
    """Handles calculation of image similarity metrics."""

    def __init__(self,
                 target: str = None,
                 data_range: tuple[float, float] = (-1, 1)):
        self.target = target
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range)

    def generate(self):
        gr.Markdown("### Metrics")
        desc = gr.Markdown(
            f"""
            This plot shows the changes of two confidences over the iterations:\n
            1) The blue represents the 'most probable'/target confidence.\n 
            2) The green plot represent the confidence of the original class.\n 
            It can be seen that with each iteration the confidence of the original class decreases, showing how each iteration improves the effectiveness of the attack.
            """
        )
        with gr.Row():
            with gr.Column(scale=3):
                # Confidence plot
                line_plot = gr.Plot()

            with gr.Column(scale=1):
                # Metrics
                metrics = gr.Dataframe(
                    headers=["Metric", "Value"],
                    value=[["SSIM", "None"], ["Difference", "None"], ["Time", "None"]],
                    interactive=False,
                )
        return desc, line_plot, metrics

    def update_metrics(self, original: np.ndarray, predicted: np.ndarray):
        original = self._valid(original)
        predicted = self._valid(predicted)
        max_diff = (original - predicted).abs().amax().item()
        ssim = self.ssim(original, predicted).item()
        return gr.update(
            value=[["SSIM", round(ssim, 3)], ["Difference", round(max_diff, 3)], ["Time", f"{round(self.time, 3)}s"]]
        )

    def _valid(self, img) -> torch.Tensor:
        """Ensure tensor has batch dimension."""
        img = F.to_tensor(img)
        return img.unsqueeze(0) if img.dim() == 3 else img

    def update(self, step_idx: int):
        # Clear matplotlib history
        plt.clf()
        plt.cla()
        plt.close()

        # Check if confidence data is available
        if not hasattr(self, "confidence") or not hasattr(self, "most_confidence"):
            # Return empty plot if no data
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(
                0.5,
                0.5,
                "No confidence data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Confidence", fontsize=12)
            out = gr.update(value=fig)
            plt.close(fig)
            return out
        else:
            confidence = getattr(self, "confidence")
            most_confidence = getattr(self, "most_confidence")

        N = len(confidence)

        # Validate step_idx
        step_idx = max(0, min(step_idx, N))

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the main lines
        ax.plot(list(range(1, N + 1)), most_confidence, "b-",
                linewidth=2,
                alpha=0.7,
                label=f"{self.target} confidence" if self.target is not None else "Most probable")

        ax.plot(
            list(range(1, N + 1)),
            confidence,
            "g-",
            linewidth=2,
            alpha=0.7,
            label="Original prediction's confidence",
        )

        # Add a vertical line at the slider position
        ax.axvline(x=step_idx + 1, color="red", linestyle="--", alpha=0.5)

        # Highlight the current points
        if step_idx < len(most_confidence):
            ax.scatter(step_idx + 1, most_confidence[step_idx], color="blue", s=100, zorder=5)
        if step_idx < len(confidence):
            ax.scatter(step_idx + 1, confidence[step_idx], color="green", s=100, zorder=5)

        # Customize the plot
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Confidence", fontsize=12)
        ax.set_title("Confidence Evolution", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set reasonable axis limits
        ax.set_xlim(-0.5, N + 1.5)

        plt.tight_layout()
        out = gr.update(value=fig)
        return out
