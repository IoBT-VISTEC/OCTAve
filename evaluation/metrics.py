from operator import gt
from typing import Tuple, NewType, Optional, Type


SliceBegin = Optional[int]
SliceEnd = Optional[int]

from einops import rearrange
import numpy as np
import sklearn.metrics as metrics
from skimage.filters import threshold_otsu


def confusion_matrix(pred, gt):
    tn, fp, fn, tp = metrics.confusion_matrix(gt.flatten(), pred.flatten()).ravel()
    return tn, fp, fn, tp

def auc(pred, gt):
    auroc = metrics.roc_auc_score(gt.flatten(), pred.flatten())
    return auroc

def accuracy(tn, fp, fn, tp):
    acc = (tp + tn) / (fp + fn + tp + tn)
    return acc

def iou(tn, fp, fn, tp):
    iou = tp / (fp + fn + tp + 1e-12)
    return iou

def dice(tn, fp, fn, tp):
    dice = (2. * tp) / (fp + fn + (2. * tp) + 1e-12)
    return dice

def cohen_kappa(tn, fp, fn, tp):
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

def sensitivity_specificity_fpr(tn, fp, fn, tp):
    sen = tp / (fn + tp + 1e-12)
    spe = tn / (fp + tn + 1e-12)
    fpr = fp / (fp + tp + 1e-12)
    return sen, spe, fpr



class StandardEvaluator:

    def __init__(
        self, num_classes: int = 1,
        pred_class_slice: Tuple[SliceBegin, SliceEnd] = (0, None),
        gt_class_slice: Tuple[SliceBegin, SliceEnd] = (0, None),
        enable_adaptive_thresholding: bool = True,
        fixed_threshold_value: float = 0.5
        ):
        """Standard evaluation class

        params:
        num_classes: int                                        Number of the `one_hot` classes.
        pred_class_slice: Tuple[SliceBegin, SliceEnd]           Predicate array slicing.
        gt_class_slice: Tuple[SliceBegin, SliceEnd]             Groundtruth array slicing.
        enable_adaptive_thresholding: bool                      Enable Adaptive Thresholding (OTSU).
        fixed_threshold_value: float                            Fixed thresholding value in case of non-adaptive.
        ):
        """
        self.num_classes = num_classes
        self.pred_class_slice = pred_class_slice
        self.gt_class_slice = gt_class_slice
        self.adaptive_thresholding = enable_adaptive_thresholding
        self.fixed_threshold_value = fixed_threshold_value

        self.wrapped = False

    @property
    def wrapped(self):
        return self._wrapped

    @wrapped.setter
    def wrapped(self, flag: bool):
        self._wrapped = flag

    def wrap(self):
        self.wrapped = not self.wrapped

    def _rearrange(self, x):
        _shape = x.shape
        if len(_shape) > 4: raise ValueError(f'Invalid number of dimension, got {len(_shape)}')
        elif len(_shape) == 4:
            if _shape[0] > 1:
                raise ValueError(f'Expecting batch_size of 1 in evaluation for {type(self).__name__}')
            x = rearrange(x, '1 c h w -> c h w')
        elif len(_shape) == 2:
            x = rearrange(x, 'h w -> 1 h w')
        return x

    def _thresholding(self, x):
        """Threshold transformed array C x H X W"""
        _x = []
        for c in range(x.shape[0]):
            thresh = threshold_otsu(x[c])
            _x.append(x[c] >= thresh)
        return rearrange(_x, 'c h w -> c h w')

    def register(self, predicate: np.ndarray, ground_truth: np.ndarray):
        if any(map(lambda x: not type(x) is np.ndarray, [predicate, ground_truth])):
            raise TypeError('Expecting type `np.ndarray`')
        
        predicate = self._rearrange(predicate)
        ground_truth = self._rearrange(ground_truth)

        predicate = predicate[self.pred_class_slice[0]:self.pred_class_slice[1], ...]
        ground_truth = ground_truth[self.gt_class_slice[0]:self.gt_class_slice[1], ...]

        pred_num_classes = predicate.shape[0]
        gt_num_classes = ground_truth.shape[0]
        if pred_num_classes != gt_num_classes:
            raise ValueError(f'`num_classes` mismatch, with predicate {pred_num_classes} classes and ground_truth {gt_num_classes} classes.')
        self.predicate, self.ground_truth = predicate, ground_truth
        self.thresholded_predicate = self._thresholding(predicate) if self.adaptive_thresholding else self.predicate >= self.fixed_threshold_value


    def cfm(self, predicate: np.ndarray, ground_truth: np.ndarray):
        """Confusion Matrix"""
        if self.wrapped:
            self.register(predicate, ground_truth)
            self.cfm_out = confusion_matrix(self.thresholded_predicate, self.ground_truth)
            return self.cfm_out
        return confusion_matrix(predicate, ground_truth)

    def cohen_kappa(self, predicate: np.ndarray, ground_truth: np.ndarray):
        if self.wrapped:
            return cohen_kappa(*self.cfm_out)
        return cohen_kappa(*confusion_matrix(predicate, ground_truth))

    def dice(self, predicate: np.ndarray, ground_truth: np.ndarray):
        if self.wrapped:
            return dice(*self.cfm_out)
        return dice(*confusion_matrix(predicate, ground_truth))

    def roc_auc(self, predicate: np.ndarray, ground_truth: np.ndarray):
        if self.wrapped:
            return auc(self.predicate, self.ground_truth)
        return auc(predicate, ground_truth)

    def acc(self, predicate: np.ndarray, ground_truth: np.ndarray):
        if self.wrapped:
            return accuracy(*self.cfm_out)
        return accuracy(*confusion_matrix(predicate, ground_truth))

    def evaluate(self, predicate: np.ndarray, ground_truth: np.ndarray):
        """Evaluate method.
        Perform evalution giving predicate and ground_truth.
        """
        self.wrap()

        metrics = {}

        self.register(predicate=predicate, ground_truth=ground_truth)

        metrics['cfm'] = self.cfm(predicate, ground_truth)
        metrics['accuracy'] = self.acc(predicate, ground_truth)
        metrics['dice'] = self.dice(predicate, ground_truth)
        metrics['kappa'] = self.cohen_kappa(predicate, ground_truth)
        metrics['auroc'] = self.roc_auc(predicate, ground_truth)

        self.wrap()
        return metrics