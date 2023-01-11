from pathlib import Path
import pickle
import random
from typing import List, Literal, Optional, Sequence
from loguru import logger

from skimage.morphology import skeletonize, binary_erosion
from einops import rearrange
import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as FT
from torchvision.transforms.transforms import ToTensor

from interfaces.octa500 import prepare_octa_500
from utils.tools import BalancedBatchSampler, read_subject_label, split_data


VESSEL: str = 'vessel'
FAZ: str = 'faz'


class OCTA500Datamodule(LightningDataModule):

    class OCTAData(Dataset):
        DATA_CLASS_LABEL: dict = {
            "NORMAL": [1, 0, 0, 0, 0, 0, 0],
            "DR": [0, 1, 0, 0, 0, 0, 0],
            "CNV": [0, 0, 1, 0, 0, 0, 0],
            "AMD": [0, 0, 0, 1, 0, 0, 0],
            "RVO": [0, 0, 0, 0, 1, 0, 0],
            "CSC": [0, 0, 0, 0, 0, 1, 0],
            "OTHERS": [0, 0, 0, 0, 0, 0, 1]
        }
        def __init__(
            self,
            is_train: bool,
            X: Sequence[np.ndarray],
            y_weak: Sequence[np.ndarray],
            y_full: Sequence[np.ndarray],
            x_subject_id: Sequence[np.ndarray],
            y_unpair: Optional[Sequence[np.ndarray]] = None,
            y_class: Optional[Sequence[str]] = None,
            mask_y: Optional[Sequence[np.ndarray]] = None,
            skeletonize_background: bool = False,
            unpair_augmentation: bool = False,
            scribble_presence_ratio: float = 1.0,
            class_filter: Optional[Sequence[str]] = None,
            label_of_interest: Literal['vessel', 'faz'] = 'vessel',
            rotate_aug: bool = True,
            ):
            self.is_train = is_train
            # Label Elimination
            unsupervise_labels = int(len(y_weak) * (1 - scribble_presence_ratio))
            self.y_excluded = random.sample(range(len(y_weak)), unsupervise_labels)
            self.X =  X
            self.y_weak = y_weak
            self.y_full = y_full
            self.x_subject_id = x_subject_id
            self.y_unpair = y_unpair
            self.y_class = y_class
            self.y_class_onehot = list()
            self.mask_y = mask_y
            self.skel_bg = skeletonize_background
            self.unpair_aug = unpair_augmentation
            self.class_filter = class_filter
            self.loi = label_of_interest
            self.rotate_aug = rotate_aug

            # For convenient sake
            unpair_flag = False
            mask_flag = False
            if self.y_unpair is None:
                unpair_flag = True
                self.y_unpair = [None for _ in range(len(self.X))]
            if self.mask_y is None:
                mask_flag = True
                self.mask_y = [None for _ in range(len(self.X))]

            # Unfolding class label into a one-hot encoding.
            if not self.y_class is None:
                self.y_class_onehot = np.array([ self.DATA_CLASS_LABEL[c] for c in self.y_class])
                if not self.class_filter is None:
                    # Filter the dataset with class criteria
                    _t = np.arange(len(self.class_filter))
                    mat = np.zeros((_t.size, _t.max() + 1))
                    mat[np.arange(_t.size), _t] = 1
                    mat = list(mat)
                    _t = {v: mat[k] for k, v in enumerate(self.class_filter)}
                    criteria_check = lambda class_item: class_item in self.class_filter
                    criterias = np.array(list(map(criteria_check, self.y_class)))
                    logger.info(f'Criteria Filter: {criterias.shape}')
                    self.X, self.y_weak, self.y_full, self.x_subject_id, self.y_unpair, self.y_class, self.mask_y = ( 
                        np.array(self.X)[criterias]
                        , np.array(self.y_weak)[criterias]
                        , np.array(self.y_full)[criterias]
                        , np.array(self.x_subject_id)[criterias]
                        , np.array(self.y_unpair)[criterias]
                        , np.array(self.y_class)[criterias]
                        , np.array(self.mask_y)[criterias])
                    self.X, self.y_weak, self.y_full, self.x_subject_id, self.y_unpair, self.y_class, self.mask_y = [
                        list(obj) for obj in (self.X, self.y_weak, self.y_full, self.x_subject_id, self.y_unpair, self.y_class, self.mask_y)]
                    self.y_class_onehot = np.array([_t[str(c)] for c in self.y_class])

                if not self.y_class_onehot is None:
                    if len(self.y_class_onehot) != 0:
                        assert len(self.X) == len(self.y_class_onehot), f"Sanity check failed, data point length {len(self.X)} not eqaul to the label length {len(self.y_class_onehot)}"

            # Restore
            if unpair_flag:
                self.y_unpair = None
            if mask_flag:
                self.mask_y = None

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            # Cast to Tensor
            x = ToTensor()(self.X[index])
            subject_id = self.x_subject_id[index]
            if self.loi == VESSEL:
                y_weak = ToTensor()(self.y_weak[index]).long()
                y_full = ToTensor()(self.y_full[index]).long()
            elif self.loi == FAZ:
                y_weak = rearrange(torch.tensor(self.y_weak[index]).long(), 'h w -> 1 h w')
                y_full = rearrange(torch.tensor(self.y_full[index]).long(), 'h w -> 1 h w')
            # Augment
            # Rotation augmentation
            aug_deg = np.random.uniform(low=-10, high=10)
            if self.rotate_aug:
                x, y_w, y_f = FT.rotate(x, angle=aug_deg), FT.rotate(y_weak, angle=aug_deg), FT.rotate(y_full, angle=aug_deg)
            else:
                y_w, y_f = y_weak, y_full

            y_class = dict()
            if len(self.y_class_onehot) == len(self.X):
                y_class = {'y_class': self.y_class_onehot[index]}

            # No unpair to be loaded.
            if self.y_unpair is None:
                if self.loi == VESSEL:
                    y_w = rearrange(F.one_hot(y_w, num_classes=2), '1 h w code -> code h w')
                    y_f = rearrange(F.one_hot(y_f, num_classes=2), '1 h w code -> code h w')
                    return {'x': x, 'y_weak': y_w, 'y': y_f, 'ignore_y': index in self.y_excluded, **y_class, 'subject_id': subject_id}
                elif self.loi == FAZ:
                    y_w = torch.cat([y_w == 128,  y_w == 255], dim=0).long()
                    y_f = torch.cat([y_f == 128,  y_f == 255], dim=0).long()
                    return {'x': x, 'y_weak': y_w, 'y': y_f, 'ignore_y': index in self.y_excluded, **y_class, 'subject_id': subject_id}

            # Cast unpair to Tensor
            if self.loi == VESSEL:
                y_upr = ToTensor()(random.choice(self.y_unpair)).long()
            elif self.loi == FAZ:
                y_upr = rearrange(torch.tensor(self.y_unpair[index]).long(), 'h w -> 1 h w')
            h, w = [int(i) for i in y_weak.shape[1:]]

            # To one-hot
            if self.loi == VESSEL:
                y_w = rearrange(F.one_hot(y_w, num_classes=2), '1 h w code -> code h w')
                y_f = rearrange(F.one_hot(y_f, num_classes=2), '1 h w code -> code h w')
                if self.skel_bg:
                    y_w[0] = torch.tensor(skeletonize(binary_erosion(y_w[0].numpy()))).long()
            elif self.loi == FAZ:
                y_w = torch.cat([y_w == 128,  y_w == 255], dim=0).long()
                y_f = torch.cat([y_f == 128,  y_f == 255], dim=0).long()

            # In case where the unpair size is mismatched. 
            y_upr = FT.center_crop(y_upr, [h, w])
            if self.unpair_aug:
                y_upr = FT.rotate(y_upr, angle=aug_deg)
            if self.loi == VESSEL:
                y_upr = rearrange(F.one_hot(y_upr, num_classes=2), '1 h w code -> code h w').float()
            elif self.loi == FAZ:
                y_upr = torch.cat([y_upr == 128,  y_upr == 255], dim=0).long()

            if self.mask_y is None:
                return {'x': x, 'y_weak': y_w, 'y': y_f, 'y_unpair': y_upr, 'ignore_y': index in self.y_excluded, **y_class, 'subject_id': subject_id}
            else:
                y_mask = rearrange(torch.from_numpy(self.mask_y[index]).long(), 'h w -> 1 h w')
                # Masking
                return {'x': x, 'y_weak': y_w * y_mask, 'y': y_f, 'y_unpair': y_upr, 'ignore_y': index in self.y_excluded, **y_class, 'subject_id': subject_id}

    def __init__(
        self,
        source_dataset_path: str,
        processed_dataset_path: str,
        train_modality: Literal['3M', '6M'],
        train_annoation_type: Literal['Full', 'Weak'],
        unpaired_modality: Literal['3M', '6M', None],
        train_projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM'],
        unpair_projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM'],
        n_fold: int = 5,
        shuffle_cv: bool = True,
        unpair_scribble_ratio: float = 0.2,
        random_bg_crop: bool = False,
        crop_portions: Literal[1, 2, 3] = 1,
        skeletonize_bg: bool = False,
        unpair_augmentation: bool = False,
        scribble_presence_ratio: float = 1.0,
        ontop_train_test_split: bool = True,
        test_ratio: float = 0.3,
        label_3m_path: Optional[Path] = None,
        label_6m_path: Optional[Path] = None,
        class_filter: Optional[Sequence[str]] = None,
        label_of_interest: Literal['vessel', 'faz'] = 'vessel',
    ):
        """OCTA-500 Experimentation Datamodule.
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
        self.load_path = Path(processed_dataset_path)
        self.train_modality = train_modality
        self.unpaired_modality = unpaired_modality
        self.self_unpaired = self.train_modality == self.unpaired_modality
        self.train_annotation = train_annoation_type
        self.train_projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM'] = train_projection_level
        self.unpair_projection_level: Literal['FULL', 'ILM_OPL', 'OPL_BM'] = unpair_projection_level
        self.unpair_scribble_ratio = unpair_scribble_ratio
        self.scribble_presence_ratio = scribble_presence_ratio
        self.n_fold = n_fold
        self.shuffle_cv = shuffle_cv
        self.random_bg_crop = random_bg_crop
        self.crop_portions = crop_portions
        self.skel_bg = skeletonize_bg
        self.unpair_aug = unpair_augmentation
        self.ontop_train_test_split = ontop_train_test_split
        self.test_ratio = test_ratio
        self.label_3m_path = label_3m_path
        self.label_6m_path = label_6m_path
        self.class_filter = class_filter
        self.data_class_label = None
        self._loaded_label = list()
        self._subject_label = list()
        self.loi = label_of_interest

        if not self.label_3m_path is None and not self.label_6m_path is None:
            self.load_label()

    def prepare_data(self, override: bool = False) -> None:
        prepare_octa_500(
            datapath=str(self.src_path),
            target_path=str(self.load_path),
            projection_level_3m=self.train_projection_level if self.train_modality == '3M' else self.unpair_projection_level,
            projection_level_6m=self.train_projection_level if self.train_modality == '6M' else self.unpair_projection_level,
            folds=self.n_fold,
            shuffle=self.shuffle_cv,
            random_background_crop=self.random_bg_crop,
            crop_portion = self.crop_portions,
            safety_override=override,
            ontop_train_test_split=self.ontop_train_test_split,
            test_ratio=self.test_ratio,
            label_of_interest=self.loi,
        )

    def _get_images(self, dir: Path, color_mode: str):
        """Hooking method"""
        for p in dir.iterdir():
            if p.is_file() and p.suffix == '.bmp':
                if not self.data_class_label is None:
                    label = self.data_class_label[int(p.stem)]
                    self._loaded_label.append(label)
                self._subject_label.append(p.stem)
                yield np.array(Image.open(p).convert(color_mode))

    def load_label(self):
        _3m_label = read_subject_label(self.label_3m_path)
        _6m_label = read_subject_label(self.label_6m_path)
        self.data_class_label = {**_3m_label, **_6m_label}

    def unload_label(self):
        ret = np.array(self._loaded_label)
        self._loaded_label = list()
        return ret

    def unload_subject_id(self):
        ret = np.array(self._subject_label)
        self._subject_label = list()
        return ret

    def setup(self, fold: int) -> None:
        # Load from each fold
        print(f"Datamodule is now loading data from fold {fold}.")
        self.current_fold = fold
        self.train_mode_path = self.load_path.joinpath(f'{self.train_modality}_{self.train_projection_level}')
        assert self.train_mode_path.is_dir(), f'{self.train_mode_path} is not exists.'
        if self.unpaired_modality is None:
            logger.info('Unpair disabled')
        else:
            self.unpaired_mode_path = self.load_path.joinpath(f'{self.unpaired_modality}_{self.unpair_projection_level}')
            assert self.unpaired_mode_path.is_dir(), f'{self.unpaired_mode_path} is not exists.'

    def train_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        if not self.self_unpaired and not self.unpaired_modality is None:
            raise NotImplementedError('Unmatched unpair configuration disabled.')
        else:
            train_mode_x = list(self._get_images(self.train_mode_path.joinpath(f'train/projection/fold_{self.current_fold}'), 'RGB'))
            train_mode_y_class = self.unload_label()
            train_mode_x_id = self.unload_subject_id()
            train_mode_y_weak = list(self._get_images(self.train_mode_path.joinpath(f'train/weak_ground_truth/fold_{self.current_fold}'), 'L'))
            train_mode_y_full = list(self._get_images(self.train_mode_path.joinpath(f'train/pixel_ground_truth/fold_{self.current_fold}'), 'L'))
            train_mode_y_mask = list(self._get_images(self.train_mode_path.joinpath(f'train/mask_ground_truth/fold_{self.current_fold}'), 'L'))
            self.unload_label()
            self.unload_subject_id()

            # Divide to halves, resampling data into pair-unpair.
            if self.unpaired_modality is None:
                logger.info("Loading without unpair data")
                train_unpaired_mode_y = None
            else:
                if len(train_mode_y_class) == 0:
                    train_samples= list(zip(train_mode_x, train_mode_y_weak, train_mode_y_full, train_mode_y_mask))
                else:
                    train_samples= list(zip(train_mode_x, train_mode_y_weak, train_mode_y_full, train_mode_y_mask, train_mode_y_class))
                samples = len(train_samples) // 2
                train_unpaired_mode_y = [train_samples.pop(random.randint(0, len(train_samples)-1))[2] for _ in range(samples)] # (_, _, y, _)
                if len(train_mode_y_class) == 0:
                    train_mode_x, train_mode_y_weak, train_mode_y_full, train_mode_y_mask = list(zip(*train_samples))
                else:
                    train_mode_x, train_mode_y_weak, train_mode_y_full, train_mode_y_mask, train_mode_y_class = list(zip(*train_samples))
            train_dataset = self.OCTAData(
                True, train_mode_x, y_weak=train_mode_y_weak, y_full=train_mode_y_full, x_subject_id=train_mode_x_id, y_unpair=train_unpaired_mode_y, y_class=train_mode_y_class,
                mask_y=train_mode_y_mask, skeletonize_background=self.skel_bg, unpair_augmentation=self.unpair_aug, scribble_presence_ratio=self.scribble_presence_ratio
                , class_filter=self.class_filter, label_of_interest=self.loi)
            return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        val_mode_x = list(self._get_images(self.train_mode_path.joinpath(f'val/projection/fold_{self.current_fold}'), 'RGB'))
        val_mode_y_class = self.unload_label()
        val_mode_x_id = self.unload_subject_id()
        val_mode_y = list(self._get_images(self.train_mode_path.joinpath(f'val/pixel_ground_truth/fold_{self.current_fold}'), 'L'))
        val_dataset = self.OCTAData(True, val_mode_x, val_mode_y, val_mode_y, x_subject_id=val_mode_x_id,
                                    y_class=val_mode_y_class, class_filter=self.class_filter, label_of_interest=self.loi)
        self.unload_label()
        self.unload_subject_id()
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    def test_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        test_mode_x = list(self._get_images(self.train_mode_path.joinpath(f'test/projection'), 'RGB'))
        test_mode_y_class = self.unload_label()
        test_mode_x_id = self.unload_subject_id()
        test_mode_y = list(self._get_images(self.train_mode_path.joinpath(f'test/ground_truth'), 'L'))
        test_dataset = self.OCTAData(True, test_mode_x, test_mode_y, test_mode_y, x_subject_id=test_mode_x_id, y_class=test_mode_y_class,
                                     class_filter=self.class_filter, label_of_interest=self.loi, rotate_aug=False)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class OCTA500Classification(LightningDataModule):

    path_pattern: str = 'OCTA({})'

    class OCTA500CLDataset(Dataset):

        def __init__(self, X: Sequence, y: Sequence, onehot_dict: dict, subject_id: Sequence, balance: bool = False):
            self.X = X
            self.y = list(y)
            self.y_class_onehot = [onehot_dict[_y] for _y in y]
            self.x_subject_id = subject_id

            if balance:
                unique, counts = np.unique(np.array(self.y), return_counts=True)
                min_idx = np.argmin(counts)
                min_c = counts[min_idx]
                max_idx = np.argmax(counts)
                maj_label = unique[max_idx]
                print(maj_label)
                maj_idx = list(map(lambda x: x[0], filter(lambda i: i[1] == maj_label, enumerate(self.y))))
                sampled = random.sample(maj_idx, k=min_c)
                min_idx = list(map(lambda x: x[0], filter(lambda i: i[1] != maj_label, enumerate(self.y))))
                sampled = [*sampled, *min_idx]
                filter_fn = lambda l: list(map(lambda x: x[1], filter(lambda tup: tup[0] in sampled, enumerate(l))))
                self.X, self.y, self.y_class_onehot, self.x_subject_id = \
                    filter_fn(self.X), filter_fn(self.y), filter_fn(self.y_class_onehot), filter_fn(self.x_subject_id)

        def __getitem__(self, index):
            # Cast to Tensor
            x = ToTensor()(self.X[index])
            subject_id = self.x_subject_id[index]

            # Augment
            # Rotation augmentation
            aug_deg = np.random.uniform(low=-10, high=10)
            x = FT.rotate(x, angle=aug_deg)

            y_class = {'y_class': self.y_class_onehot[index], 'class_label': self.y[index]}

            return {'x': x, **y_class, 'subject_id': subject_id}

        def __len__(self):
            return len(self.X)

    def __init__(self, datapath: str, labelpath:str, modality: Literal['OCTA_3M', 'OCTA_6M'], depth: Literal['FULL', 'ILM_OPL', 'OPL_BM'], dump_path: str):
        super().__init__()
        self.datapath = Path(datapath).joinpath(modality).joinpath('Projection Maps').joinpath(self.path_pattern.format(depth))
        assert self.datapath.is_dir()
        self.labelpath = Path(labelpath)
        assert self.labelpath.is_file()
        self.dump_root = Path(dump_path).joinpath(modality).joinpath('classification')
        self.dump_root.mkdir(parents=True, exist_ok=True)
        self.train_save = self.dump_root.joinpath('train.pkl')
        self.val_save = self.dump_root.joinpath('val.pkl')
        self.test_save = self.dump_root.joinpath('test.pkl')
        self.label_one_hot_save = self.dump_root.joinpath('label_onehot.pkl')
        self.modality = modality
        self.depth = depth

    def prepare_data(self, class_filter: list, test_ratio: float = 0.5, val_ratio: float = 0.3,  **kwargs):
        label = read_subject_label(self.labelpath)
        label = {k: v for k, v in filter(lambda it: it[1] in class_filter, label.items())}
        train, test = split_data(label=label, test_size=test_ratio)
        train, val = split_data(label=label, test_size=val_ratio)


        _t = np.arange(len(class_filter))
        mat = np.zeros((_t.size, _t.max() + 1))
        mat[np.arange(_t.size), _t] = 1
        mat = list(mat)
        _t = {v: mat[k] for k, v in enumerate(class_filter)}

        # Dump
        with self.train_save.open('wb') as f:
            pickle.dump(train, f)
        with self.val_save.open('wb') as f:
            pickle.dump(val, f)
        with self.test_save.open('wb') as f:
            pickle.dump(test, f)
        with self.label_one_hot_save.open('wb') as f:
            pickle.dump(_t, f)

    def _open_save(self, path: Path):
        with path.open('rb') as f:
            k = pickle.load(f)
        return k

    def setup(self, **kwargs):
        assert self.train_save.is_file()
        assert self.val_save.is_file()
        assert self.test_save.is_file()
        assert self.label_one_hot_save.is_file()
        self.train_label = self._open_save(self.train_save)
        self.val_label = self._open_save(self.val_save)
        self.test_label = self._open_save(self.test_save)
        self.label_one_hot = self._open_save(self.label_one_hot_save)

    def _get_images(self, path: Path, label: dict):
        idx = list(label.keys())
        pathmap = dict()
        # idx -> path map
        for p in path.iterdir():
            if p.suffix == '.bmp' and int(p.stem) in idx:
                pathmap[int(p.stem)] = p
        for _idx in idx:
            yield np.array(Image.open(pathmap[_idx]).convert('RGB'))

    def train_dataloader(self, batch_size: int = 12, balance: bool = True):
        X = list(self._get_images(self.datapath, label=self.train_label))
        x_subject_id = list(self.train_label.keys())
        y = self.train_label.values()
        onehot_map = self.label_one_hot
        dataset = self.OCTA500CLDataset(X=X, y=y, onehot_dict=onehot_map, subject_id=x_subject_id, balance=balance)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self, batch_size: int = 1):
        X = list(self._get_images(self.datapath, label=self.val_label))
        x_subject_id = list(self.val_label.keys())
        y = self.val_label.values()
        onehot_map = self.label_one_hot
        dataset = self.OCTA500CLDataset(X=X, y=y, onehot_dict=onehot_map, subject_id=x_subject_id)
        return DataLoader(dataset, batch_size=batch_size)

    def test_dataloader(self, batch_size: int = 1):
        X = list(self._get_images(self.datapath, label=self.test_label))
        x_subject_id = list(self.test_label.keys())
        y = self.test_label.values()
        onehot_map = self.label_one_hot
        dataset = self.OCTA500CLDataset(X=X, y=y, onehot_dict=onehot_map, subject_id=x_subject_id)
        return DataLoader(dataset, batch_size=batch_size)