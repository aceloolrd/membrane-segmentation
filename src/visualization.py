import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def show_first_batch(data_loader: DataLoader, max_items: int = 2) -> None:
    """Show the first batch of images and masks from a DataLoader."""
    images, masks = next(iter(data_loader))
    for i in range(min(max_items, len(images))):
        image = images[i].squeeze().cpu().numpy()
        mask = masks[i].squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


def plot_metrics(metrics: dict) -> None:
    """Plot train/val curves for loss, accuracy, Jaccard index and F-beta score."""
    keys = list(metrics.keys())
    n = len(keys)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, key in zip(axes, keys):
        epochs = range(1, len(metrics[key]) + 1)
        ax.plot(epochs, [m["train"] for m in metrics[key]], label=f"Train {key}", marker="o")
        ax.plot(epochs, [m["val"] for m in metrics[key]], label=f"Val {key}", marker="o")
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.legend()

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def visualize_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 2,
) -> None:
    """Visualize image | ground truth | heatmap | binary prediction for each sample."""
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in data_loader:
            if shown >= max_batches:
                break
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            outputs = torch.sigmoid(model(images)).cpu().numpy()
            binary = (outputs > 0.5).astype(float)

            for i in range(len(images)):
                img = images[i].cpu().squeeze().numpy()
                mask = masks[i].cpu().squeeze().numpy()
                heatmap = 1 - outputs[i].squeeze()
                bin_mask = binary[i].squeeze()

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img, cmap="gray");  axes[0].set_title("Image");            axes[0].axis("off")
                axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Ground Truth");     axes[1].axis("off")
                axes[2].imshow(heatmap, cmap="viridis"); axes[2].set_title("Heatmap");    axes[2].axis("off")
                axes[3].imshow(bin_mask, cmap="gray"); axes[3].set_title("Prediction");   axes[3].axis("off")
                plt.tight_layout()
                plt.show()
            shown += 1
