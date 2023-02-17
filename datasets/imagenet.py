import torch
import torch.utils.data as data
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image

import os


def load_imagesets_file(file_path):
    # with open(file_path, 'r') as f:
    return pd.read_csv(file_path, sep=' ', header=None)


def load_solution_file(file_path):
    # with open(file_path, 'r') as f:
    return pd.read_csv(file_path, sep=',', header=0)


def pil_loader(img_path):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageNet1KDataset(data.Dataset):
    # Kaggle CLS-LOC ImageNet1K
    def __init__(self, root, subset, transform):
        self.num_classes = 1000
        self.data_root = root
        assert subset == 'train' or subset == 'val', "Wrong subset!"
        self.data_root = os.path.join(root, 'ILSVRC', 'Data', 'CLS-LOC', subset)
        self.loader = pil_loader
        self.transform = transform
        solution_file = None
        solution_map = {}
        loc_synset_mapping_file = pd.read_csv(os.path.join(root, 'LOC_synset_mapping.txt'), sep=' ',
                                              names=[str(i) for i in range(45)], header=None)
        self.idx_to_id = loc_synset_mapping_file.iloc[:, 0].values.tolist()
        self.id_to_idx = {id: idx for idx, id in enumerate(self.idx_to_id)}

        if subset == "train":
            imagesets_file = load_imagesets_file(
                os.path.join(root, 'ILSVRC', 'ImageSets', 'CLS-LOC', 'train_cls.txt'))
        else:  # val
            imagesets_file = load_imagesets_file(
                os.path.join(root, 'ILSVRC', 'ImageSets', 'CLS-LOC', 'val.txt'))
            solution_file = load_solution_file(os.path.join(root, 'LOC_val_solution.csv'))
            for img_id, pred_str in zip(solution_file.iloc[:, 0], solution_file.iloc[:, 1]):
                label_id = pred_str.split(' ')[0]
                solution_map[img_id] = self.id_to_idx[label_id]

        self.data = []
        if subset == 'train':
            for p in imagesets_file.iloc[:, 0]:
                self.data.append({
                    'img_path': os.path.join(self.data_root, p + '.JPEG'),
                    'label': self.id_to_idx[p.split('/')[0]]
                })
        else:  # val
            for p in imagesets_file.iloc[:, 0]:
                self.data.append({
                    'img_path': os.path.join(self.data_root, p + '.JPEG'),
                    'label': solution_map[p]
                })


    def __getitem__(self, index):
        data_item = self.data[index]
        sample = self.loader(data_item['img_path'])
        if self.transform is not None:
            sample = self.transform(sample)
        target = data_item['label']
        return sample, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    root = ''
    ds = ImageNet1KDataset(root, 'train', transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
    kwargs = {"num_workers": 4,
              "pin_memory": True,
              "batch_size": 1,
              "prefetch_factor": 10,
              "persistent_workers": True}

    train_loader = torch.utils.data.DataLoader(ds, shuffle=True, drop_last=True, **kwargs)
    for img, label in train_loader:
        print(img, label)

    print(len(ds))
