from typing import List, Literal, Tuple 

import cv2
from einops.einops import rearrange, reduce
import numpy as np
from sklearn import metrics
from skimage import filters, morphology
from torch.functional import Tensor

def detach(func):
    def wrapper(*args, **kwargs):
        args = [ arg.detach().cpu().numpy() for arg in args]
        kwargs = {k: v.detach().cpu().numpy() for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

@detach
def confusion_matrix(pred, gt, threshold=0.5):
    tn, fp, fn, tp = metrics.confusion_matrix(gt[:, 1:, :, :].flatten(), pred[:, 1:, :, :].flatten() > threshold).ravel()
    return tn, fp, fn, tp

@detach
def auc(pred, gt):
    auroc = metrics.roc_auc_score(gt[:, 1:, :, :].flatten(), pred[:, 1:, :, :].flatten())
    return auroc

def accuracy(pred, gt):
    tn, fp, fn, tp = confusion_matrix(pred, gt)
    acc = (tp + tn) / (fp + fn + tp + tn)
    return acc

def iou(pred, gt):
    _, fp, fn, tp = confusion_matrix(pred, gt)
    iou = tp / (fp + fn + tp + 1e-12)
    return iou

def dice(pred, gt):
    _, fp, fn, tp = confusion_matrix(pred, gt)
    dice = (2. * tp) / (fp + fn + (2. * tp) + 1e-12)
    return dice

def cohen_kappa(pred, gt):
    tn, fp, fn, tp = confusion_matrix(pred, gt)
    m = np.array([[tp, fp], [fn, tn]])
    n = np.sum(m)
    sum_po = 0
    sum_pe = 0
    for i in range(len(m[0])):
        sum_po += m[i][i]
        row = np.sum(m[i, :])
        col = np.sum(m[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)

def sensitivity_specificity_fpr(pred, gt):
    tn, fp, fn, tp = confusion_matrix(pred, gt)
    sen = tp / (fn + tp + 1e-12)
    spe = tn / (fp + tn + 1e-12)
    fpr = fp / (fp + tp + 1e-12)
    return sen, spe, fpr


class Evaluator:

    def __init__(
        self,
        tolerance: Literal[None, 'gt-tolerance', 'predicate-tolerance', 'bi-tolerance'],
        thresholding_mode: Literal['fixed', 'adaptive'],
        threshold_value: float = 0.5,
        tolerance_classes: List[int] = [2],
        num_classes: int = 1,
        collapsible: bool = True,
        bg_start_idx: int = 1,
        tolerance_kernel: int = (3, 3)):
        """Evaluation metric utility class. Only handle single class at a time.
        params:
        tolerance: Literal['gt-tolerance', 'predicate-tolerance', 'bi-tolerance']       Mode of tolerance.
        thresholding_mode: Literal['fixed', 'adaptive']                                 Thresholding mode.
        threshold_value: float                                                          Fixed thresholding value. Default 0.5.
        tolerance_classes: List[int]                                                    List of class that subject to tolerancing.
        num_classes: int                                                                Number of classes, disregarding background.
        collapsible: bool                                                               Collapsible evaluation.
        """
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.thresholding_mode = thresholding_mode
        self.fixed_threshold_value = threshold_value
        self.tolerance_classes = tolerance_classes
        self.collapsible = collapsible
        self.bg_start_idx = bg_start_idx
        self.tolerance_kernel = tolerance_kernel

    def register(self,
        predicate: Tensor,
        ground_truth: Tensor):
        """Register predicate and ground truth.
        """
        bg_idx = self.bg_start_idx
        try:
            self.predicate: np.ndarray = predicate.detach().cpu().numpy()
        except:
            self.predicate: np.ndarray = predicate
        try:
            self.ground_truth: np.ndarray = ground_truth.detach().cpu().numpy().astype(np.uint8)
        except:
            self.ground_truth: np.ndarray = ground_truth.astype(np.uint8)

        self.threshold_value = [ self.fixed_threshold_value if self.thresholding_mode == 'fixed' else filters.threshold_otsu(self.predicate[0, c, ...]) for c in range(bg_idx, self.num_classes + bg_idx) ]
        self.thresholded_predicate = [ (self.predicate[0, c, ...] >= val) * 1.0 for c, val in enumerate(self.threshold_value, bg_idx) ]
        self.ground_truths = [self.ground_truth[0, c, ...] for c in range(bg_idx, self.num_classes + bg_idx)]
        self.cf_mat = self.confusion_matrix()

    def tolerencing(self):
        """Post-processing for tolerated evaluation.
        """
        if self.tolerance is None:
            # No tolerance
            return self.thresholded_predicate, self.ground_truths, self.ground_truths
        if self.tolerance in ['gt-tolerance', 'bi-tolerance']:
            # Dilate ground truth by 3 pixel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.tolerance_kernel)
            self.dilated_ground_truths = [ self.ground_truths[c].copy() for c in range(self.num_classes) ]
            for c in self.tolerance_classes:
                self.dilated_ground_truths[c-1] = cv2.dilate(self.ground_truths[c-1], kernel, iterations=1)

        if self.tolerance in ['bi-tolerance', 'predicate-tolerance']:
            # Skeletonizing the output.
            for c in self.tolerance_classes:
                self.thresholded_predicate[c-1] = morphology.skeletonize(self.thresholded_predicate[c-1])

        # Collapsing mask
        if self.collapsible:
            collapsed_pred = np.max(rearrange(self.thresholded_predicate, 'c h w -> c h w'), axis=0)
            collapsed_gt = np.max(rearrange(self.ground_truths, 'c h w -> c h w'), axis=0)
            collapsed_dilated = np.max(rearrange(self.dilated_ground_truths, 'c h w -> c h w'), axis=0)
            self.thresholded_predicate.append(collapsed_pred)
            self.ground_truths.append(collapsed_gt)
            self.dilated_ground_truths.append(collapsed_dilated)

        return self.thresholded_predicate, self.ground_truths, self.dilated_ground_truths

    def confusion_matrix(self) -> List[Tuple[int, int, int, int]]:
        _confusion_matrix = []
        thresholded_predicate, ground_truths, dilated_ground_truths = self.tolerencing()
        for pred, gt, dilated in zip(thresholded_predicate, ground_truths, dilated_ground_truths):
            fp = float(np.sum(np.logical_and(pred == 1, dilated == 0)))
            fn = float(np.sum(np.logical_and(pred == 0, gt == 1)))
            tp = float(np.sum(np.logical_and(pred == 1, dilated == 1)))
            tn = float(np.sum(np.logical_and(pred == 0, gt == 0)))
            _confusion_matrix.append((tn, fp, fn, tp))
        return _confusion_matrix

    def auc(self):
        class_separated_predicate = [ self.predicate[0, c, ...] for c in range(self.bg_start_idx, self.num_classes+self.bg_start_idx) ]
        if self.collapsible:
            class_separated_predicate.append(reduce(self.predicate[0, self.bg_start_idx:, ...], 'c h w -> h w', 'sum'))
        pred, gt = class_separated_predicate, self.ground_truths
        return [metrics.roc_auc_score(gt[c].flatten().astype(np.uint8), pred[c].flatten()) for c in range(len(gt))]

    def accuracy(self):
        acc = []
        for cf_mat in self.cf_mat:
            tn, fp, fn, tp = cf_mat
            acc.append((tp + tn) / (fp + fn + tp + tn))
        return acc

    def iou(self):
        iou = []
        for cf_mat in self.cf_mat:
            _, fp, fn, tp = cf_mat
            iou.append(tp / (fp + fn + tp + 1e-12))
        return iou

    def dice(self):
        dice = []
        for cf_mat in self.cf_mat:
            _, fp, fn, tp = cf_mat
            dice.append((2. * tp) / (fp + fn + (2. * tp) + 1e-12))
        return dice

    def cohen_kappa(self):
        kappa = []
        for cf_mat in self.cf_mat:
            tn, fp, fn, tp = cf_mat
            m = np.array([[tp, fp], [fn, tn]])
            n = np.sum(m)
            sum_po = 0
            sum_pe = 0
            for i in range(len(m[0])):
                sum_po += m[i][i]
                row = np.sum(m[i, :])
                col = np.sum(m[:, i])
                sum_pe += row * col
            po = sum_po / n
            pe = sum_pe / (n * n)
        
            kappa.append((po - pe) / (1 - pe))
        return kappa

    def sensitivity_specificity_fpr(self):
        sen, spe, fpr = [], [], []
        for cf_mat in self.cf_mat:
            tn, fp, fn, tp = cf_mat
            sen.append(tp / (fn + tp + 1e-12))
            spe.append(tn / (fp + tn + 1e-12))
            fpr.append(fp / (fp + tp + 1e-12))
        return sen, spe, fpr
