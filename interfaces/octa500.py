# Dataloader from OCTA-500 Dataset
from logging import warning
import random
from pathlib import Path
from re import I
import shutil
from typing import Any, Dict, Literal, Optional, Tuple, Union

from einops import rearrange
import numpy as np
import pandas as pd
from PIL import Image
from pandas.core.frame import DataFrame
from sklearn.model_selection import KFold, train_test_split
from skimage.morphology import skeletonize
from skimage import morphology as morph
from torch.functional import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm


VESSEL: str = 'vessel'
FAZ: str =  'faz'

class UnpairOcta500(Dataset):

    __3M_PATH: str = 'OCTA_3M'
    __6M_PATH: str = 'OCTA_6M'
    GT: str = 'GroundTruth'
    PROJ: str = 'Projection Maps'
    LBLNAME: str = 'Text labels.xlsx'
    LABEL: dict = {
        'bg': 0,
        'faz': 100,
        'vessel': 255,
    }

    def __init__(
        self,
        datapath: str,
        modality: Literal['3m', '6m', 'both'],
        level: Literal['FULL', 'ILM_OPL', 'OPL_BM', None],
        ground_truth_style: Literal['grayscale', 'one-hot'],
        health_control_ratio: Union[float, None] = None,
        sample_size: int = 0) -> None:
        """OCTA500 Projection Dataset

        params:
        datapath: str                               Path to directory containing both OCTA-3M and OCTA-6M dataset.
        modality: str                               3M or 6M Modality, Accept None if multi_mode is enabled.
        multi_mode: bool                            Loading both modalities.
        health_control_ratio: Union[float, None]    Ratio of the healthy control group, None for disabling control group. Default is None.
        sample_size: int                            Size of the returning sample. 0 for entire dataset. Default is 0.
        """
        self.path = datapath
        self.gt_style = ground_truth_style
        self.healthy_ratio = health_control_ratio
        self.sample_size = sample_size
        self.dataset = self._get_dataset(modality=modality, level=level)
        self.keys = list(self.dataset['projections'].keys())

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = Path(path)
        self._3m_path = path
        self._6m_path = path

    @property
    def _3m_path(self):
        return self._3m_path_

    @_3m_path.setter
    def _3m_path(self, path):
        self._3m_path_ = Path(path).joinpath(self.__3M_PATH)
        assert self._3m_path_.is_dir()

    @property
    def _6m_path(self):
        return self._6m_path_

    @_6m_path.setter
    def _6m_path(self, path):
        self._6m_path_ = Path(path).joinpath(self.__6M_PATH)
        assert self._6m_path.is_dir()

    def _get_dataset(self, modality: Literal['3m', '6m', 'both'], level: Literal['FULL', 'ILM_OPL', 'OPL_BM', None]) -> Dict[str, Union[Dict[str, Any], Any]]:
        """Get Dataset
        """
        root = Path()
        def _get_from_path(root):
            gt_dir = root.joinpath(self.GT)
            proj_dir = root.joinpath(self.PROJ)
            label_file = root.joinpath(self.LBLNAME)
            # Load info
            info = pd.read_excel(str(label_file))
            proj_img_dict = {}
            gt_img_dict = {}
            for filename in gt_dir.iterdir():
                if filename.suffix != '.bmp': continue
                gt_img_dict[filename.stem] = str(filename)
            if not level is None:
                modal_dir = proj_dir.joinpath(f'OCTA({level})')
                for filename in modal_dir.iterdir():
                    if filename.suffix != '.bmp': continue
                    proj_img_dict[filename.stem] = str(filename)
            else:
                NotImplementedError('Multi-modalities yet to be implemented.')
            # Apply filter
            if not self.healthy_ratio is None:
                control_group = info[info['Disease'].isin(['NORMAL', 'DR'])]
                assert self.sample_size != 0, 'Sample size must be given for creating the control group.'
                healthy_group = control_group[control_group['Disease'] == 'NORMAL']
                dr_group = control_group[control_group['Disease'] == 'DR']
                n_healthy = len(healthy_group)
                n_dr = len(dr_group)
                healthy_sample = int(self.sample_size * self.healthy_ratio)
                assert healthy_sample <= n_healthy, f'Not enough healthy sample. {healthy_sample}/{n_healthy}.'
                dr_sample = self.sample_size - healthy_sample
                assert dr_sample <= n_dr, f'Not enough dr sample. {dr_sample}/{n_dr}.'
                sampled_h = list(healthy_group.sample(n=healthy_sample).ID)
                sampled_d = list(dr_group.sample(n=dr_sample).ID)
                filter_list = list(map(str, sampled_h + sampled_d))
                filter_func = lambda ele: ele[0] in filter_list
                proj_img_dict = dict(filter(filter_func, proj_img_dict.items()))
                gt_img_dict = dict(filter(filter_func, gt_img_dict.items()))
                info = control_group
            elif self.sample_size > 0:
                # Sampling
                filter_list = list(map(str, info.sample(n=self.sample_size).ID))
                filter_func = lambda ele: ele[0] in filter_list
                proj_img_dict = dict(filter(filter_func, proj_img_dict.items()))
                gt_img_dict = dict(filter(filter_func, gt_img_dict.items()))
            return {'projections': proj_img_dict, 'ground truths': gt_img_dict, 'info': info}
        if modality != 'both':
            root = {'3m': self._3m_path, '6m': self._6m_path}[modality]
        else: root = [self._3m_path, self._6m_path]
        if not isinstance(root, list):
            return _get_from_path(root)
        else:
            temp = [_get_from_path(r) for r in root]
            proj_img_dict: Dict[str, Tensor] = {**temp[0]['projections'], **temp[1]['projections']}
            gt_img_dict: Dict[str, Tensor] = {**temp[0]['ground truths'], **temp[1]['ground truths']}
            info = {'3m': temp[0]['info'], '6m': temp[1]['info']}
            info = pd.concat(info)
            return {'projections': proj_img_dict, 'ground truths': gt_img_dict, 'info': info}

    def __getitem__(self, index):
        key = self.keys[index]
        gt_path = self.dataset['ground truths'][key]
        proj_path = self.dataset['projections'][key]
        # Load to image
        proj = Image.open(proj_path).convert('RGB')
        gt = np.array(Image.open(gt_path).convert('L'))
        # Remove FAZ
        gt = np.where(gt == self.LABEL['faz'], self.LABEL['bg'], gt)
        proj, gt = ToTensor()(proj).float(), ToTensor()(gt).long()

        if self.gt_style == 'one-hot':
            gt = rearrange(F.one_hot(gt, num_classes=2), '1 h w code -> code h w')

        return {'projection': proj, 'ground truth': gt.int()}

    def __len__(self):
        return len(self.dataset['ground truths'])


def faz_scribble_syntheize(faz_img_arr):
    """FAZ Scribble Synthesizer Function.
    """
    ddisk = morph.disk(3)
    edisk = morph.disk(20)
    cvx = morph.convex_hull_image(faz_img_arr)
    faz_s = morph.dilation(morph.skeletonize(cvx), ddisk)
    bg_s = morph.dilation(morph.skeletonize(~(morph.erosion(~faz_img_arr, edisk)) ^ cvx), ddisk)
    return faz_s, bg_s


def prepare_octa_500(
    datapath: str,
    target_path: str,
    projection_level_3m: Literal['FULL', 'ILM_OPL', 'OPL_BM'],
    projection_level_6m: Literal['FULL', 'ILM_OPL', 'OPL_BM'],
    folds: int = 5,
    shuffle: bool = True,
    random_background_crop: bool = False,
    crop_portion: Literal[1, 2, 3] = 1,
    safety_override: bool = False,
    ontop_train_test_split: bool = False,
    test_ratio: float = 0.3,
    label_of_interest: Literal['vessel', 'faz'] = 'vessel',
    ):
    """Prepare the dataset
    """
    loi = label_of_interest
    dirpath = Path(datapath)
    target_dirpath = Path(target_path)
    if not safety_override:
        if target_dirpath.is_dir():
            shutil.rmtree(target_dirpath)
    target_dirpath.mkdir(parents=True, exist_ok=safety_override)
    octa3m = dirpath.joinpath('OCTA_3M')
    octa3m_target = octa3m.joinpath('GroundTruth')
    octa3m_proj = octa3m.joinpath('Projection Maps')
    octa3m_info = octa3m.joinpath('Text labels.xlsx')
    octa6m = dirpath.joinpath('OCTA_6M')
    octa6m_target = octa6m.joinpath('GroundTruth')
    octa6m_proj = octa6m.joinpath('Projection Maps')
    octa6m_info = octa6m.joinpath('Text labels.xlsx')

    def _get_from_path(gt_dir: Path, proj_dir: Path, label_file: Path):
        # Load info
        info = pd.read_excel(str(label_file))
        proj_img_dict = {}
        gt_img_dict = {}
        for filename in gt_dir.iterdir():
            if filename.suffix != '.bmp': continue
            gt_img_dict[filename.stem] = str(filename)
        if not projection_level_3m is None:
            modal_dir = proj_dir.joinpath(f'OCTA({projection_level_3m})')
            for filename in modal_dir.iterdir():
                if filename.suffix != '.bmp': continue
                proj_img_dict[filename.stem] = str(filename)
        return proj_img_dict, gt_img_dict, info

    def stratify_split(df: DataFrame, split_by: str = 'Disease') -> Tuple[DataFrame, DataFrame]:
        category = df.pop(split_by).to_frame()
        _id = df.pop('ID').to_frame()
        train, test, _, _ = train_test_split(_id, category, stratify=category, test_size=test_ratio)
        return train, test

    # K-folding
    def folding(projection: list, ground_truth: list, modal: Literal['3M', '6M'], projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        portion = []
        kf = KFold(n_splits=folds, shuffle=shuffle)
        fold_n = 0
        warned = False
        for train_idxs, val_idxs in tqdm(kf.split(projection), total=folds, desc=f'OCTA_{modal}_{loi}', unit='fold'):
            proj_save = target_dirpath.joinpath(f'{modal}_{projection_level}/train/projection/fold_{fold_n}')
            gt_save = target_dirpath.joinpath(f'{modal}_{projection_level}/train/pixel_ground_truth/fold_{fold_n}')
            weak_gt_save = target_dirpath.joinpath(f'{modal}_{projection_level}/train/weak_ground_truth/fold_{fold_n}')
            mask_gt_save = target_dirpath.joinpath(f'{modal}_{projection_level}/train/mask_ground_truth/fold_{fold_n}')
            for dir_path in [proj_save, gt_save, weak_gt_save, mask_gt_save]:
                try:
                    dir_path.mkdir(parents=True, exist_ok=False)
                except FileExistsError as e:
                    if safety_override:
                        if not warned:
                            warning(f'FileExistsError raised due to some directory is going to be override. Proceed with caution.')
                            warned = True
                    else:
                        raise e
            for train_idx in train_idxs:
                proj = Image.open(projection[train_idx]).convert('RGB')
                proj.save(proj_save.joinpath(f'{Path(projection[train_idx]).stem}.bmp'), 'bmp')
                gt = Image.open(ground_truth[train_idx]).convert('L')
                if len(portion) == 0 and random_background_crop:
                    # Calculate portions
                    h, w = np.array(gt).shape
                    h_2, w_2 = h//2, w//2
                    portion = [((0, h_2), (0, w_2)), ((h_2, h), (0, w_2)), ((0, h_2), (w_2, w)), ((h_2, h), (w_2, w))]
                # Remove FAZ
                if loi == VESSEL:
                    gt = np.where(np.array(gt) == 100, 0, gt)
                    weak_gt = (skeletonize(np.where(gt == 255, 1, 0).astype(np.uint8)) * 255).astype(np.uint8)
                elif loi == FAZ:
                    gt = np.where(np.array(gt) == 255, 0, gt) != 0
                    faz_s, bg_s = [arr.astype(np.uint8) for arr in faz_scribble_syntheize(gt)]
                    # Merge faz, bg_s
                    weak_gt = (faz_s * 255) + (bg_s * 128)
                    gt = gt.astype(np.uint8)
                    gt = np.where(gt == 0, 128, gt)
                    gt = np.where(gt == 1, 255, gt)
                mask_gt: np.ndarray = np.ones_like(weak_gt)
                if random_background_crop:
                    cropping = random.sample(portion, crop_portion)
                    for x, y in cropping:
                        sx, tx = x
                        sy, ty = y
                        mask_gt[sx:tx, sy:ty] = 0
                gt_img, weak_gt_img, mask_gt_img = list(map(Image.fromarray, [gt, weak_gt, mask_gt]))
                gt_img.save(gt_save.joinpath(f'{Path(ground_truth[train_idx]).stem}.bmp'), 'bmp')
                weak_gt_img.save(weak_gt_save.joinpath(f'{Path(ground_truth[train_idx]).stem}.bmp'), 'bmp')
                mask_gt_img.save(mask_gt_save.joinpath(f'{Path(ground_truth[train_idx]).stem}.bmp'), 'bmp')

            
            proj_save = target_dirpath.joinpath(f'{modal}_{projection_level}/val/projection/fold_{fold_n}')
            gt_save = target_dirpath.joinpath(f'{modal}_{projection_level}/val/pixel_ground_truth/fold_{fold_n}')
            weak_gt_save = target_dirpath.joinpath(f'{modal}_{projection_level}/val/weak_ground_truth/fold_{fold_n}')
            mask_gt_save = target_dirpath.joinpath(f'{modal}_{projection_level}/val/mask_ground_truth/fold_{fold_n}')
            for dir_path in [proj_save, gt_save, weak_gt_save, mask_gt_save]:
                try:
                    dir_path.mkdir(parents=True, exist_ok=False)
                except FileExistsError as e:
                    if safety_override:
                        if not warned:
                            warning(f'FileExistsError raised due to some directory is going to be override. Proceed with caution.')
                            warned = True
                    else:
                        raise e
            for val_idx in val_idxs:
                proj = Image.open(projection[val_idx]).convert('RGB')
                proj.save(proj_save.joinpath(f'{Path(projection[val_idx]).stem}.bmp'), 'bmp')
                gt = Image.open(ground_truth[val_idx]).convert('L')
                # Remove FAZ
                if loi == VESSEL:
                    gt = np.where(np.array(gt) == 100, 0, gt)
                    weak_gt = (skeletonize(np.where(gt == 255, 1, 0).astype(np.uint8)) * 255).astype(np.uint8)
                elif loi == FAZ:
                    gt = np.where(np.array(gt) == 255, 0, gt) != 0 
                    faz_s, bg_s = [arr.astype(np.uint8) for arr in faz_scribble_syntheize(gt)]
                    # Merge faz, bg_s
                    weak_gt = (faz_s * 255) + (bg_s * 128)
                    gt = gt.astype(np.uint8)
                    gt = np.where(gt == 0, 128, gt)
                    gt = np.where(gt == 1, 255, gt)
                mask_gt: np.ndarray = np.ones_like(weak_gt)
                if random_background_crop:
                    cropping = random.sample(portion, crop_portion)
                    for x, y in cropping:
                        sx, tx = x
                        sy, ty = y
                        mask_gt[sx:tx, sy:ty] = 0
                gt_img, weak_gt_img, mask_gt_img = list(map(Image.fromarray, [gt, weak_gt, mask_gt]))
                gt_img.save(gt_save.joinpath(f'{Path(ground_truth[val_idx]).stem}.bmp'), 'bmp')
                weak_gt_img.save(weak_gt_save.joinpath(f'{Path(ground_truth[val_idx]).stem}.bmp'), 'bmp')
                mask_gt_img.save(mask_gt_save.joinpath(f'{Path(ground_truth[val_idx]).stem}.bmp'), 'bmp')
            fold_n += 1

    proj_3m, gt_3m, info_3m = _get_from_path(gt_dir=octa3m_target, proj_dir=octa3m_proj, label_file=octa3m_info)
    proj_6m, gt_6m, info_6m = _get_from_path(gt_dir=octa6m_target, proj_dir=octa6m_proj, label_file=octa6m_info)

    if not ontop_train_test_split:
        # NOTE: Replace two following lines with filtering logic that return list of filtered id.
        id_3m = [ str(_id) for _id in list(info_3m.ID)]
        id_6m = [ str(_id) for _id in list(info_6m.ID)]
        proj_3m = [proj_3m[_id] for _id in id_3m]
        proj_6m = [proj_6m[_id] for _id in id_6m]
        gt_3m = [gt_3m[_id] for _id in id_3m]
        gt_6m = [gt_6m[_id] for _id in id_6m]

        folding(proj_3m, gt_3m, '3M', projection_level_3m)
        folding(proj_6m, gt_6m, '6M', projection_level_6m)
    else:
        # Split construction.
        id_3m, id_3m_test = [list(map(lambda x: str(x), list(df.ID))) for df in stratify_split(info_3m)]
        id_6m, id_6m_test = [list(map(lambda x: str(x), list(df.ID))) for df in stratify_split(info_6m)]

        proj_3m_train = [proj_3m[_id] for _id in id_3m]
        proj_6m_train = [proj_6m[_id] for _id in id_6m]
        gt_3m_train = [gt_3m[_id] for _id in id_3m]
        gt_6m_train = [gt_6m[_id] for _id in id_6m]

        folding(proj_3m_train, gt_3m_train, '3M', projection_level_3m)
        folding(proj_6m_train, gt_6m_train, '6M', projection_level_6m)

        proj_3m_test = [proj_3m[_id] for _id in id_3m_test]
        proj_6m_test = [proj_6m[_id] for _id in id_6m_test]
        gt_3m_test = [gt_3m[_id] for _id in id_3m_test]
        gt_6m_test = [gt_6m[_id] for _id in id_6m_test]

        def construct_test(projection: list, ground_truth: list, modal: Literal['3M', '6M'], projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
            proj_save = target_dirpath.joinpath(f'{modal}_{projection_level}/test/projection/')
            target_save = target_dirpath.joinpath(f'{modal}_{projection_level}/test/ground_truth/')
            proj_save.mkdir(parents=True, exist_ok=True)
            target_save.mkdir(parents=True, exist_ok=True)

            for idx in tqdm(range(len(projection)), desc='Test set'):
                proj = Image.open(projection[idx]).convert('RGB')
                proj.save(proj_save.joinpath(f'{Path(projection[idx]).stem}.bmp'), 'bmp')
                gt = Image.open(ground_truth[idx]).convert('L')
                if loi == VESSEL:
                    gt = np.where(np.array(gt) == 100, 0, gt)
                elif loi == FAZ:
                    gt = np.where(np.array(gt) == 255, 0, gt)
                    gt = np.where(np.array(gt) == 100, 255, gt)
                    gt = np.where(np.array(gt) == 0, 128, gt)
                gt = Image.fromarray(gt)
                gt.save(target_save.joinpath(f'{Path(ground_truth[idx]).stem}.bmp'), 'bmp')


        construct_test(proj_3m_test, gt_3m_test, '3M', projection_level_3m)
        construct_test(proj_6m_test, gt_6m_test, '6M', projection_level_6m)