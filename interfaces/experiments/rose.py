from pathlib import Path
from typing import List, Literal, Optional, Union
from loguru import logger

import numpy as np
from skimage.morphology import skeletonize
from pytorch_lightning import LightningDataModule
from sklearn import datasets
from sklearn.model_selection import train_test_split

from interfaces.octa500 import UnpairOcta500
from interfaces.rose import ROSE
from torch.utils.data import Dataset, RandomSampler, DataLoader


class RoseOcta500(Dataset):

    def __init__(
        self,
        is_train: bool,
        rose_datapath: str,
        rose_modality: Literal['thick_gt', 'thin_gt', 'gt'],
        octa500_datapath: str,
        octa500_modality: Literal['3m', '6m', 'both'],
        octa500_level: Literal['FULL', 'ILM_OPL', 'OPL_BM'],
        ground_truth_style: Literal['grayscale', 'one-hot'] = 'one-hot',
        separate_vessel_capillary: bool = False,
        rose_input_augmentation: bool = False,
        vessel_capillary_gt: Literal['vessel', 'capillary', 'both'] = 'both',
        health_control_ratio: Union[float, None] = None,
        ) -> None:
        """Rose-Octa500 Paired Dataset
        params:
        is_train: bool                      Load in training or validation.
        rose_datapath: str                  Path(str) to directory containing ROSE-1/ROSE-2 dataset.
        rose_modality: str                  ROSE-1 SVC ground truth modality ['thick_gt', 'thin_gt', 'gt'].   
        octa500_datapath: str               Path(str) to directory containing OCTA_3M/OCTA_6M dataset.
        octa500_modality: Literal[str]      OCTA-500 Dataset modality ['3m', '6m', 'both'].
        octa500_level: Literal[str]         OCTA-500 Projection level ['FULL', 'ILM_OPL', 'OPL_BM'].
        ground_truth_style: Literal[str]    Ground truth processing ['grayscale', 'one-hot']. Default to 'one-hot'.
        separate_vessel_capillary: bool     Process ground truth such that vessel and capillary belongs to separate class. Default is False.
        rose_input_augmentation: bool       ROSE additional input augmentation.
        vessel_capillary_gt: Literal[str]   Including ground truth type, vessel-only, capillary-only or both. Default is 'both'.
        health_control_ratio: float | None  Ratio of unpair healthy sample in health control group proportional to the training sample.
        """
        super().__init__()
        self.is_train = is_train
        self.ROSE = ROSE(
                        datapath=rose_datapath, is_train=is_train, ground_truth_mode=rose_modality,
                        ground_truth_style=ground_truth_style, separate_vessel_capilary=separate_vessel_capillary,
                        enable_augment=rose_input_augmentation, vessel_capillary_gt=vessel_capillary_gt)
        if is_train:
            self.octa500 = UnpairOcta500(
                datapath=octa500_datapath,
                modality=octa500_modality,
                level=octa500_level,
                ground_truth_style=ground_truth_style,
                health_control_ratio=health_control_ratio,
                sample_size=len(self.ROSE),
                )
            self.sample_key = list(RandomSampler(self.octa500))[:len(self.ROSE)]

    def __getitem__(self, index):
        ROSE_sample = self.svc[index]
        if self.is_train:
            octa500_sample = self.octa500[self.sample_key[index]]
            x = ROSE_sample['projection']
            y_weak = ROSE_sample['ground truth']
            y = octa500_sample['ground truth']
            return {'x': x, 'y_weak': y_weak, 'y': y}
        else:
            x = ROSE_sample['projection']
            y_target = ROSE_sample['ground truth']
            return {'x': x, 'y_target': y_target}

    def __len__(self):
        return len(self.ROSE)


class RoseDatamodule(LightningDataModule):

    class Rose(Dataset):

        def __init__(
            self,
            is_train: Literal['train', 'val', 'test'],
            rose_modality: Literal['ROSE-1/SVC', 'ROSE-1/DVC', 'ROSE-1/SVC-DVC', 'ROSE-2'],
            rose_datapath: str,
            rose_label_modality: Literal['thick_gt', 'thin_gt', 'gt'],
            ground_truth_style: Literal['grayscale', 'one-hot'] = 'one-hot',
            separate_vessel_capillary: bool = False,
            rose_input_augmentation: bool = False,
            vessel_capillary_gt: Literal['vessel', 'capillary', 'both'] = 'both',
            weakly_enable: bool = False,
            train_idx_map: Optional[np.ndarray] = None,
            val_idx_map: Optional[np.ndarray] = None,
            ) -> None:
            """Rose-Octa500 Paired Dataset
            params:
            is_train: bool                      Load in training or validation.
            rose_datapath: str                  Path(str) to directory containing ROSE-1/ROSE-2 dataset.
            rose_label_modality: str                  ROSE-1 SVC ground truth modality ['thick_gt', 'thin_gt', 'gt'].   
            ground_truth_style: Literal[str]    Ground truth processing ['grayscale', 'one-hot']. Default to 'one-hot'.
            separate_vessel_capillary: bool     Process ground truth such that vessel and capillary belongs to separate class. Default is False.
            rose_input_augmentation: bool       ROSE additional input augmentation.
            vessel_capillary_gt: Literal[str]   Including ground truth type, vessel-only, capillary-only or both. Default is 'both'.
            """
            super().__init__()
            self.is_train = is_train
            self.rose_modality = rose_modality
            self.rose_label_modality = rose_label_modality
            self.weakly_enable = weakly_enable
            self.train_idx_map = train_idx_map
            self.val_idx_map = val_idx_map
            self.ROSE = ROSE(
                            datapath=rose_datapath, rose_modal=rose_modality, is_train=True if is_train in ('train', 'val') else False, ground_truth_mode=rose_label_modality,
                            ground_truth_style=ground_truth_style, separate_vessel_capilary=separate_vessel_capillary,
                            enable_augment=rose_input_augmentation, vessel_capillary_gt=vessel_capillary_gt)
            # if is_train:
            #     # self.sample_key = list(RandomSampler(self.octa500))[:len(self.ROSE)]
            if is_train == 'train' and not self.train_idx_map is None:
                self.map_idx = self.train_idx_map
            elif is_train == 'val' and not self.val_idx_map is None:
                self.map_idx = self.val_idx_map
            else:
                self.map_idx = np.arange(len(self.ROSE))
            logger.info(f'Staged: {self.map_idx}')

        def __getitem__(self, index):
            # Remapping
            new_idx = self.map_idx[index]
            ROSE_sample = self.ROSE[new_idx]
            if self.rose_label_modality == 'thin_gt':
                raise Exception('No longer supported.')
            if self.rose_label_modality in ('thick_gt', 'gt'):
                x = ROSE_sample['x']
                # Force construction of y_weak via skeletonization.
                y = ROSE_sample['y']
                if self.weakly_enable:
                    # y_weak = (skeletonize(np.where(y == 255, 1, 0).astype(np.uint8)) * 255).astype(np.uint8)
                    mask = y.numpy().astype(np.uint8)
                    mask[1] = skeletonize(mask[1], method='zhang').astype(np.uint8)
                    mask[0] = np.logical_not(mask[1]).astype(np.uint8)
                    y_weak = mask
                else:
                    y_weak = y
                return {'x': x, 'y': y, 'y_weak': y_weak}

        def __len__(self):
            return len(self.map_idx)

    def __init__(
        self,
        source_dataset_path: str,
        rose_modal: Literal['ROSE-1/SVC', 'ROSE-1/DVC', 'ROSE-1/SVC-DVC', 'ROSE-2'],
        rose_label_modality: Literal['thick_gt', 'thin_gt', 'gt'],
        ground_truth_style: Literal['grayscale', 'one-hot'] = 'one-hot',
        separate_vessel_capillary: bool = False,
        rose_input_augmentation: bool = False,
        vessel_capillary_gt: Literal['vessel', 'capillary', 'both'] = 'both',
        weakly_gt: bool = False,
    ):
        """ROSE Experimentation Datamodule.
        params:
        source_dataset_path: str                                            Path to source dataset.
        processed_dataset_path: str                                         Path to process(ed) dataset.
        train_modality: Literal['3M', '6M']                                 Modality that going to be trained.
        train_annoation_type: Literal['Full', 'Weak']                       Annotion type used in training.
        unpaired_modality: Literal['3M', '6M']                              Modality that used for an unpaired dataset.
        train_projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM']
        unpair_projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM']
        n_fold: int = 5                                                     Prepare dataset from k-fold cross validation.
        shuffle_cv: bool = True                                             Prepare k-fold cv in random shuffling manner.
        unpair_scribble_ratio: float = 0.2                                  Ratio of unpair dataset samples and training samples.
        random_bg_crop: bool = False                                        Enable random ground truth cropping. Simulating lazy clinician.
        scribble_presence_ratio: float                                      Ration of scribble label vs unsupervised in training dataset.
        crop_portions: Literal[1, 2, 3]                                     Number of quarters excluded. This parameter is unused if `randomized_bg_crop` is False.
        ontop_train_test_split: bool                                        Enable on-top train test split.
        test_ratio: float                                                   On-top test set proportion.
        """
        super().__init__()
        self.src_path = Path(source_dataset_path)
        assert self.src_path.is_dir(), 'Source dataset path is invalid.'
        self.rose_modal = rose_modal
        self.rose_label_modality = rose_label_modality
        self.ground_truth_style = ground_truth_style
        self.separate_vessel_capillary = separate_vessel_capillary
        self.rose_input_augmentation = rose_input_augmentation
        self.vessel_capillary_gt = vessel_capillary_gt
        self.weakly_gt = weakly_gt


    def setup(self, test_ratio = 0.2, **kwargs) -> None:
        # Load from each fold
        logger.info('Setup procedure is called.')
        logger.info(f'Parameters: {self.rose_modal}, {self.rose_label_modality}, {self.ground_truth_style}')

        dataset = self.Rose(
            rose_datapath=self.src_path, is_train='train', rose_modality=self.rose_modal, rose_label_modality=self.rose_label_modality
            , rose_input_augmentation=self.rose_input_augmentation, ground_truth_style=self.ground_truth_style
            , vessel_capillary_gt=self.vessel_capillary_gt, weakly_enable=self.weakly_gt
        )
        logger.info(f'Dataset Size: {len(dataset)}')
        logger.info(f'Dataset Meta: {dataset.ROSE.modal}, {dataset.ROSE.keys}')

        self.train_idx, self.val_idx = train_test_split(np.arange(len(dataset)), test_size=test_ratio)
        logger.info(f'Train Index: {self.train_idx}')
        logger.info(f'Validation Index: {self.val_idx}')
        logger.info('Setup Completed')

    def train_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        train_dataset = self.Rose(
            rose_datapath=self.src_path, is_train='train', rose_modality=self.rose_modal, rose_label_modality=self.rose_label_modality
            , rose_input_augmentation=self.rose_input_augmentation, ground_truth_style=self.ground_truth_style
            , vessel_capillary_gt=self.vessel_capillary_gt, weakly_enable=self.weakly_gt, train_idx_map=self.train_idx
        )
        try:
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        except Exception as e:
            logger.exception(e)
            logger.debug(f'DATAMODAL {train_dataset.modal}')
            raise e
    
    def val_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        # train_dataset = self.Rose(
        #     rose_datapath=self.src_path, is_train=False, rose_modality=self.rose_modal, rose_label_modality=self.rose_label_modality
        #     , rose_input_augmentation=self.rose_input_augmentation, ground_truth_style=self.ground_truth_style
        #     , vessel_capillary_gt=self.vessel_capillary_gt, weakly_enable=self.weakly_gt
        # )
        val_dataset = self.Rose(
            rose_datapath=self.src_path, is_train='val', rose_modality=self.rose_modal, rose_label_modality=self.rose_label_modality
            , rose_input_augmentation=self.rose_input_augmentation, ground_truth_style=self.ground_truth_style
            , vessel_capillary_gt=self.vessel_capillary_gt, weakly_enable=self.weakly_gt, val_idx_map=self.val_idx
        )
        try:
            return DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        except Exception as e:
            logger.exception(e)
            logger.debug(f'DATAMODAL {val_dataset.modal}')
            raise e

    def test_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        val_dataset = self.Rose(
            rose_datapath=self.src_path, is_train='test', rose_modality=self.rose_modal, rose_label_modality=self.rose_label_modality
            , rose_input_augmentation=self.rose_input_augmentation, ground_truth_style=self.ground_truth_style
            , vessel_capillary_gt=self.vessel_capillary_gt, weakly_enable=self.weakly_gt,
        )
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)