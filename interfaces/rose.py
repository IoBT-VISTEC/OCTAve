from pathlib import Path
from typing import Literal

from einops import rearrange, repeat
from kornia import enhance
import numpy as np
from PIL import Image
import torch
from torch.functional import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor


class ROSE(Dataset):
    IMG: str = 'img'
    PATH_MAP: dict = {
        'ROSE-1/SVC': 'ROSE-1/SVC',
        'ROSE-1/DVC': 'ROSE-1/DVC',
        'ROSE-1/SVC-DVC': 'ROSE-1/SVC_DVC',
        'ROSE-2': 'ROSE-2',
    }

    def __init__(
        self,
        datapath: str,
        is_train: bool,
        rose_modal: Literal['ROSE-1/SVC', 'ROSE-1/DVC', 'ROSE-1/SVC-DVC', 'ROSE-2'],
        ground_truth_mode: Literal['thick_gt', 'thin_gt', 'gt'],
        separate_vessel_capilary: bool = False,
        ground_truth_style: Literal['grayscale', 'one-hot'] = 'one-hot',
        enable_augment: bool = False,
        vessel_capillary_gt: Literal['vessel', 'capillary', 'both'] = 'both',
        ):
        """ROSE SVC Dataset
        params:
        datapath: str                       Path(str) to directory containing ROSE-1/ROSE-2 dataset.
        is_train: bool                      Load in training or validation.
        rose_modal: str                     Data modal.
        ground_truth_mode: str              Ground truth modality. 'thick_gt' for pixel-level label, 'thin_gt' for center-line level label and
                                            'gt' for mixed label.
        separate_vessel_capilary: bool      Separate centerline ground truth of vessel apart from capillary centerline ground truth.
                                            Enabling this option alter `ground_truth_mode` parameter. If `ground_truth_mode` is 
                                            - `thick_gt` will provide pixel-level label for vessel.
                                            - `thin_gt` will provide centerline-level label for all classes.
                                            - `gt` is not supported an will raise an error.
                                            Default is False.
        ground_truth_style: str             Ground truth processing. 'grayscale' left ground truth as one channel as is,
                                            'one-hot' one-hot vectorize the image label.
        enable_augment: bool                Enable addition input augmentation.
        vessel_capillary_gt: Literal[str]   Including ground truth type, vessel-only, capillary-only or both. Default is 'both'.
        """
        super().__init__()
        self.is_train = is_train
        self.modal = rose_modal
        self.path = datapath
        self.gt_mode = ground_truth_mode
        self.gt_style = ground_truth_style
        self.sep_vessel_capill = separate_vessel_capilary
        self.enable_augment = enable_augment
        if not self.sep_vessel_capill:
            assert vessel_capillary_gt == 'both', '`separate_vessel_capillary` must be set to False if `vessel_capillary_gt` is both.'
        self.vessel_capill_gt = vessel_capillary_gt
        self.dataset = self._get_dataset()
        self.keys = list(self.dataset['projections'].keys())

    @property
    def path(self):
        return self._path

    @property
    def modal(self):
        return self._modal
    
    @modal.setter
    def modal(self, modal):
        self._modal = self.PATH_MAP[modal]

    @path.setter
    def path(self, path):
        _path = Path(path).joinpath(self.modal)
        assert _path.is_dir()
        self._path = _path

    def _get_dataset(self):
        root = self.path.joinpath('train' if self.is_train else 'test')
        def get_files(dir: Path):
            _dict = {}
            for filename in dir.iterdir():
                if not filename.suffix in ('.tif', '.png'): continue
                _dict[filename.stem] = str(filename)
            return _dict
        img = root.joinpath(self.IMG)
        img_dict = get_files(img)
        if self.sep_vessel_capill:
            thin_gt = root.joinpath('thin_gt')
            thick_gt = root.joinpath('thick_gt')
            return {'projections': img_dict, 'ground truths': {
                'thick': get_files(thick_gt),
                'thin': get_files(thin_gt)
            }}
        else:
            gt = root.joinpath(self.gt_mode)
            gt_dict = get_files(gt)
            return {'projections': img_dict, 'ground truths': gt_dict}

    @staticmethod
    def augment(x: Tensor, out_channels: int = 3):
        return enhance.equalize_clahe(x)

    def __getitem__(self, index):
        key = self.keys[index]
        img_path = self.dataset['projections'][key]
        img = Image.open(img_path).convert('RGB')
        img = ToTensor()(img)
        deg = int(torch.randint(low=-10, high=10, size=(1,)).int())
        if self.is_train:
            img = transforms.F.rotate(img, deg)
        if self.enable_augment:
            img = self.augment(img, out_channels=3)
        if self.sep_vessel_capill:
            thick_path = self.dataset['ground truths']['thick'][key]
            thin_path = self.dataset['ground truths']['thin'][key]
            thick, thin = ToTensor()(Image.open(thick_path)), ToTensor()(Image.open(thin_path))
            thick, thin = transforms.F.rotate(thick, deg).numpy(), transforms.F.rotate(thin, deg).numpy()
            if self.gt_mode == 'thin_gt':
                vessel_gt_thin: np.ndarray = np.logical_and(thick, thin)  # Get vessel centerline.
                capill_gt_thin: np.ndarray = np.logical_and(thin, np.logical_not(vessel_gt_thin))  # Subtract
            elif self.gt_mode == 'thick_gt':
                vessel_gt_thick: np.ndarray = thick # Left vessel gt as pixel level.
                capill_gt_thick: np.ndarray = np.logical_and(thin, np.logical_not(np.logical_and(thick, thin))) # Extract centerline-level label for capillary.
            else:
                raise Exception(f'Unsupport ground truth mode: {self.gt_mode} is not proper mode for separate vessel and capillary.')

            if self.gt_style == 'one-hot':
                def _process_label(vessel_gt, capill_gt, vessel_capill_gt):
                    bg: np.ndarray = np.logical_not(np.logical_or(vessel_gt, capill_gt)) # Get inverted true bg.
                    if vessel_capill_gt == 'both':
                        gt = rearrange([bg, vessel_gt, capill_gt], 'code () h w -> code h w')  # Stack
                    elif vessel_capill_gt == 'vessel':
                        gt = rearrange([bg, vessel_gt], 'code () h w -> code h w')  # Stack
                    elif vessel_capill_gt == 'capillary':
                        gt = rearrange([bg, capill_gt], 'code () h w -> code h w')  # Stack
                    return gt
                thick_gt = _process_label(vessel_gt_thick, capill_gt_thick, self.vessel_capill_gt)
                thin_gt = _process_label(vessel_gt_thin, capill_gt_thin, self.vessel_capill_gt)
            else:
                thick_gt = vessel_gt_thick + (capill_gt_thick * 2) # Multi-class mask bg=0, vessel=1, capill=2
                thin_gt = vessel_gt_thin + (capill_gt_thin * 2) # Multi-class mask bg=0, vessel=1, capill=2
            thick_gt = torch.tensor(thick_gt)
            thin_gt = torch.tensor(thin_gt)

        else:
            gt_path = self.dataset['ground truths'][key]
            gt = Image.open(gt_path).convert('L')
            gt = ToTensor()(gt).long()
            if self.is_train:
                gt =  transforms.F.rotate(gt, deg)

            # One-hot encoding
            if self.gt_style == 'one-hot':
                gt = rearrange(F.one_hot(gt, num_classes=2), '1 h w code -> code h w')

        return {'x': img, 'y': gt.int()}

    def __len__(self):
        return len(self.dataset['projections'])

