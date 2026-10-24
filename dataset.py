import os
import numpy as np
import torch
from torch.utils import data
import SimpleITK as sitk

class RegDataset(data.Dataset):
    def __init__(self, root_dir, split, res=96, aug=True, transform=None):
        super(RegDataset, self).__init__()
        self.res = res
        self.aug = aug
        self.split = split
        self.transform = transform
        self.USimages = sorted([os.path.join(root_dir,'US_image_resample',x) for x in os.listdir(os.path.join(root_dir,'US_image_resample'))])
        self.USlabels = [file.replace('US_image_resample','US_label_resample') for file in self.USimages]
        self.MRimages = [file.replace('US_image_resample','MR_image_resample') for file in self.USimages]
        self.MRlabels = [file.replace('US_image_resample','MR_label_resample') for file in self.USimages]
        print('Number of {0} images: ::{1} NIFTIs'.format(split, self.__len__()))
        
    def augment(self, image, label, res):
        if np.ndim(label) == 4:
            label = label[0]

        a, b, c = np.where(label == 255)
        am, bm, cm = label.shape
        pada = 0
        padb = 0
        padc = 0
        if am < res:
            a1 = 0
            a2 = am
            pada = res - am
        else:
            a1 = (np.max(a) + np.min(a)) // 2 - res//2
            a2 = (np.max(a) + np.min(a)) // 2 + res//2
            if self.aug:
                bia = np.random.randint(-15,15)
                a1 += bia
                a2 += bia
            if a1 < 0:
                a1 = 0
                a2 = res
            elif a2 > am:
                a1 = am-res
                a2 = am
        if bm < res:
            b1 = 0
            b2 = bm
            padb = res - bm
        else:
            b1 = (np.max(b) + np.min(b)) // 2 - res//2
            b2 = (np.max(b) + np.min(b)) // 2 + res//2
            if self.aug:
                bia = np.random.randint(-15,15)
                b1 += bia
                b2 += bia
            if b1 < 0:
                b1 = 0
                b2 = res
            elif b2 > bm:
                b1 = bm-res
                b2 = bm
        if cm < res:
            c1 = 0
            c2 = cm
            padc = res - cm
        else:
            c1 = (np.max(c) + np.min(c)) // 2 - res//2
            c2 = (np.max(c) + np.min(c)) // 2 + res//2
            if self.aug:
                bia = np.random.randint(-15,15)
                c1 += bia
                c2 += bia
            if c1 < 0:
                c1 = 0
                c2 = res
            elif c2 > cm:
                c1 = cm-res
                c2 = cm
        image = image[a1:a2,b1:b2,c1:c2]
        label = label[a1:a2,b1:b2,c1:c2]
        label[label <= 0] = 0
        label[label > 0] = 1
        if self.aug:
            pada_bia = int(pada*np.random.rand())
            padb_bia = int(padb*np.random.rand())
            padc_bia = int(padc*np.random.rand())
        else:
            pada_bia = pada//2
            padb_bia = padb//2
            padc_bia = padc//2
        image = np.pad(image,((pada_bia,pada-pada_bia),(padb_bia,padb-padb_bia),(padc_bia,padc-padc_bia)))
        label = np.pad(label,((pada_bia,pada-pada_bia),(padb_bia,padb-padb_bia),(padc_bia,padc-padc_bia)))
        return image, label
        
    def pre_process(self, inputs, is_image=True):
        if is_image:
            max_value = np.max(inputs)
            min_value = np.min(inputs)
            clip_min = min_value + 0.005 * (max_value-min_value)
            clip_max = max_value - 0.005 * (max_value-min_value)
            inputs = np.clip(inputs, clip_min, clip_max)
            
            mean = np.mean(inputs)
            std = np.std(inputs)
            inputs = (inputs - mean)/std
        
        out = torch.tensor(inputs)
        out = torch.unsqueeze(out, dim=0)
        if self.split == 'test':
            out = torch.unsqueeze(out, dim=0)
            
        return out.float()

    def __getitem__(self, index):
        # load the NIFTI images
        USimage = sitk.GetArrayFromImage(sitk.ReadImage(self.USimages[index]))
        USlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.USlabels[index]))
        MRimage = sitk.GetArrayFromImage(sitk.ReadImage(self.MRimages[index]))
        MRlabel = sitk.GetArrayFromImage(sitk.ReadImage(self.MRlabels[index]))
        
        USimage, USlabel = self.augment(USimage, USlabel, self.res)
        MRimage, MRlabel = self.augment(MRimage, MRlabel, self.res)

        # USimage = USimage[::2,::2,::2]
        # USlabel = USlabel[::2,::2,::2]
        # MRimage = MRimage[::2,::2,::2]
        # MRlabel = MRlabel[::2,::2,::2]
        
        USimage = self.pre_process(USimage, is_image=True)
        USlabel = self.pre_process(USlabel, is_image=False)
        MRimage = self.pre_process(MRimage, is_image=True)
        MRlabel = self.pre_process(MRlabel, is_image=False)

        return MRimage, USimage, MRlabel, USlabel
        
    def __len__(self):
        return len(self.USimages)


train_data = RegDataset('../../u-RegPro/train', split = 'train', aug=True)
valid_data = RegDataset('../../u-RegPro/val', split = 'valid', aug=False)
test_data = RegDataset('../../u-RegPro/val', split = 'test', aug=False)


if __name__ == "__main__":
    for i in range(50):
        a,b,c,d = test_data[i]
        sitk.WriteImage(sitk.GetImageFromArray(a), './Test' + test_data.MRimages[i][14:])
        sitk.WriteImage(sitk.GetImageFromArray(b), './Test' + test_data.USimages[i][14:])
        sitk.WriteImage(sitk.GetImageFromArray(c), './Test' + test_data.MRlabels[i][14:] + '.gz')
        sitk.WriteImage(sitk.GetImageFromArray(d), './Test' + test_data.USlabels[i][14:] + '.gz')


