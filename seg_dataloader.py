import torch
import random
import numpy as np
import os.path as osp

from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image


class COCOSegDataset(Dataset):
    def __init__(self, root_dir, input_height, input_width):
        
        json_path = osp.join(root_dir, "jsonfile", "gt.json")
        image_dir = osp.join(root_dir, "image")
    
        self.coco      = COCO(json_path)
        self.image_dir = image_dir
        self.ids       = list(self.coco.imgs.keys())

        self.input_height = input_height
        self.input_width  = input_width
        
        
    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        img_id   = self.ids[index]
        ann_ids  = self.coco.getAnnIds(imgIds=img_id)
        anns     = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        
        path = img_info['file_name']
        image = Image.open(f'{self.image_dir}/{path}').convert('RGB')

        masks = []
        for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        
        masks = np.array(masks)
        masks = np.max(masks, axis=0)
        
        aug_image, aug_gt = self.resize(image=image, gt=masks, size=(self.input_width, self.input_height))
        aug_image, aug_gt = self.rotate_image(image=aug_image, gt=aug_gt, angle=45)
        aug_image         = np.array(aug_image, dtype= np.float32) / 255.
        aug_gt            = np.array(aug_gt,    dtype= np.float32)
        aug_image, aug_gt = self.flip(image=aug_image, gt=aug_gt)
            
        sample = {'image': aug_image, 'gt': aug_gt}
        
        preprocessing_transforms = transforms.Compose([ToTensor()])
        aug_image, aug_gt = preprocessing_transforms(sample)
        
        return aug_image, aug_gt


    def resize(self, image, gt, size):
        
        resized_image = image.resize(size, Image.BICUBIC)
        
        gt  = Image.fromarray(gt)
        resized_gt    = gt.resize(size,    Image.NEAREST)
        
        return resized_image, resized_gt
        
        
    def rotate_image(self, image, gt, angle):
        angle = random.uniform(-angle, angle)
        image = F.rotate(image, angle)
        gt = F.rotate(gt, angle)
        
        return image, gt


    def flip(self, image, gt):
        hflip = random.random()
        vflip = random.random()
        if hflip > 0.5:
            image = (image[:, ::-1, :]).copy()
            gt = (gt[:, ::-1]).copy()
        
        if vflip > 0.5:
            image = (image[::-1, :, :]).copy()
            gt = (gt[::-1, :]).copy()

        return image, gt
        

class ToTensor(object):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = np.array(image, dtype= np.float32)
        gt    = np.array(gt,    dtype= np.int32)
        
        if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

            return image, gt