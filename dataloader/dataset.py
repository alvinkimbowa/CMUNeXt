import os
import cv2
import json
from glob import glob
from torch.utils.data import Dataset
import numpy as np

class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)
        sample = {"image": image, "label": label, "case": case}
        return sample


class CMUNeXt_nnUNetDataset(Dataset):
    def __init__(self, dataset_name, split, input_channels=3, num_classes=1, fold=None, split_type='train', transform=None, eval=False):
        self.transform = transform
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.eval = eval
        
        nnunet_raw = os.environ['nnUNet_raw']
        nnunet_preprocessed = os.environ['nnUNet_preprocessed']
        self.img_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/images{split}')
        self.label_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/labels{split}')
        
        with open(os.path.join(f'{nnunet_raw}/{dataset_name}/dataset.json'), 'r') as f:
            dataset_info = json.load(f)
        self.img_ext = dataset_info['file_ending']
        
        if eval:
            img_ids = glob(f'{self.img_dir}/*{self.img_ext}')
            self.img_ids = [os.path.basename(img_id).rsplit('.', 1)[0].replace('_0000', '') for img_id in img_ids]
        else:
            with open(os.path.join(f'{nnunet_preprocessed}/{dataset_name}/splits_final.json'), 'r') as f:
                splits = json.load(f)
            
            if fold == 'all':
                img_ids = []
                for split_dict in splits:
                    img_ids.extend(split_dict[split_type])
                img_ids = list(set(img_ids))
            else:
                img_ids = splits[int(fold)][split_type]
            
            self.img_ids = img_ids
        
        print("Found %d images" % len(self.img_ids))
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_filename = f'{self.img_dir}/{img_id}_0000{self.img_ext}'
        label_filename = f'{self.label_dir}/{img_id}{self.img_ext}'
        
        # Determine if other channels exist
        other_chs = [ch for ch in glob(f'{self.img_dir}/{img_id}_*{self.img_ext}') if ch != img_filename]
        other_chs = sorted(other_chs)

        # Load channels expected by the model.
        # Keep deterministic order: _0000, _0001, ...
        all_channels = [img_filename] + other_chs
        if len(all_channels) < self.input_channels:
            raise ValueError(
                f"Sample {img_id} has {len(all_channels)} channels, "
                f"but input_channels={self.input_channels}."
            )
        selected_channels = all_channels[:self.input_channels]
        imgs = [cv2.imread(ch, cv2.IMREAD_GRAYSCALE)[..., None] for ch in selected_channels]
        img = np.concatenate(imgs, axis=-1)
        
        if os.path.exists(label_filename):
            mask = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)
            if self.num_classes > 1:
                masks = []
                for i in range(self.num_classes):
                    masks.append((mask == (i + 1)).astype('float32')[..., None])
                mask = np.dstack(masks)
            else:
                mask = mask[..., None]
        elif self.eval:
            if self.num_classes > 1:
                mask = np.zeros((img.shape[0], img.shape[1], self.num_classes), dtype='float32')
            else:
                mask = np.zeros(img.shape[:2])[..., None]
        else:
            raise ValueError(f"Label file not found: {label_filename}")
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)
        
        sample = {"image": img, "label": mask, "case": img_id}
        return sample
