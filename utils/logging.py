import io
import warnings
from typing import Dict, Optional, Sequence
import random

from loguru import logger

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.figure import Figure
from PIL import Image
from torch.functional import Tensor
from torchvision.transforms import ToPILImage, ToTensor

import joblib


def detach(x: Tensor):
    return x.detach().cpu()

def plot_to_image(fig: Figure):
    """Convert matplotlib to Pytorch image tensor.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return ToTensor()(image)

def attention_predicate_plot(raw: Tensor, attentions: Sequence[Tensor], predicate: Tensor, target: Optional[Tensor] = None, parent_fig=None, multi_class: bool = False):
    """Receive tensor of raw image tensor, list of attention tensor, predicate output tensor and plot in matplotlib figure.
    """
    batch_elim = lambda x: x[0] if x.ndim == 4 else x
    if raw.shape[0] != 1 and raw.ndim > 3:
        warnings.warn('For batch size > 1, only first sample will be plotted.', UserWarning)
    attentions = [att[:, 1:, ...] for att in attentions]
    predicate = predicate[:, 1:, ...].float()
    num_att_classes = attentions[0].shape[1]
    num_pred_classes = predicate.shape[1]
    raw_img = ToPILImage()(batch_elim(raw))
    if multi_class:
       attention_imgs = [ [ ToPILImage()(batch_elim(att[:, c, ...])) for att in  attentions ] for c in range(num_att_classes) ]
       pred_img = [ ToPILImage()(batch_elim(predicate[:, c, ...])) for c in range(num_pred_classes) ]
    else:
        attention_imgs = [ToPILImage()(batch_elim(att)) for att in attentions]
        pred_img = ToPILImage()(batch_elim(predicate))

    if parent_fig is None:
        f, ax = plt.subplots(num_att_classes if multi_class else 1, len(attentions) + (2 if target is None else 3), figsize=(30, 10))
    else:
        f, ax = parent_fig.subplots(num_att_classes if multi_class else 1, len(attentions) + (2 if target is None else 3))

    if multi_class:
        for i in range(num_pred_classes):
            ax[i, 0].set_title('Raw')
            ax[i, 0].imshow(raw_img, cmap='binary')
            ax[i, 1].set_title(f'Predicate {i}')
            ax[i, 1].imshow(pred_img[i], cmap='binary')
            if not target is None:
                target_img = ToPILImage()(batch_elim(target[:, i + 1, ...]).float())
                ax[i, 2].set_title(f'Target {i}')
                ax[i, 2].imshow(target_img, cmap='binary')
            idx = 2 if target is None else 3
            for slot in range(idx, len(attentions) + idx):
                ax[i, slot].set_title(f'Attention-{i} {slot-idx}')
                ax[i, slot].imshow(attention_imgs[i][slot-idx], cmap='viridis')
        f.tight_layout()
        return f

    else:
        ax[0].set_title('Raw')
        ax[0].imshow(raw_img, cmap='binary')
        ax[1].set_title('Predicate')
        ax[1].imshow(pred_img, cmap='binary')
        if not target is None:
            target_img = ToPILImage()(batch_elim(target.float()))
            ax[2].set_title('Target')
            ax[2].imshow(target_img, cmap='binary')

        idx = 2 if target is None else 3
        for slot in range(idx, len(attentions) + idx):
            ax[slot].set_title(f'Attention {slot-idx}')
            ax[slot].imshow(attention_imgs[slot-idx], cmap='viridis')
        f.tight_layout()
        return f

def to_wandb_semantic_segmentation(raw: Tensor, predicate: Tensor, class_labels: Dict[int, str], ground_truth: Optional[Tensor] = None):
    raw_img, predicate = ToPILImage()(detach(raw)[0]), detach(predicate)
    if not ground_truth is None:
        ground_truth = detach(ground_truth)
        ground_truth = ground_truth[0] if ground_truth.ndim > 3 else ground_truth
        mask_gt = torch.argmax(ground_truth, dim=0).numpy()
    predicate = predicate[0] if predicate.ndim > 3 else predicate
    last_key = list(class_labels.keys())[-1]
    # Shifting
    gt_class_labels = {k + last_key + 1: v for k,v in class_labels.items()}
    # Replacing mask value
    if not ground_truth is None:
        for k in list(class_labels.keys()):
            mask_gt = np.where(mask_gt == k, k + last_key + 1, mask_gt)

    mask = torch.argmax(predicate, dim=0).numpy()
    mask_dict = {
        "predictions": {
            "mask_data": mask,
            "class_labels": class_labels,
        },
    }
    if not ground_truth is None:
        mask_dict['ground_truth'] = {
            'mask_data': mask_gt,
            'class_labels': gt_class_labels,
        }
    try:
        mask_img = wandb.Image(raw_img, masks=mask_dict)
    except Exception as e:
        raise Exception(f'Mask data: {np.unique(mask)}, {mask.shape} from predicate dimension {predicate.shape}') from e
    return mask_img
