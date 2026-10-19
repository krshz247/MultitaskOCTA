import torch
import os
import random
import numpy as np
import cv2

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


def mean_and_std(paths):
    print('Calculating mean and std of training set for data normalization.')
    m_list, s_list = [], []
    for img_filename in tqdm(paths):
        img = cv2.imread(img_filename)
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    m = m[0][::-1][0]/255
    s = s[0][::-1][0]/255
    print(m)
    print(s)

    return m, s
class DatasetImageClassification(Dataset):

    def __init__(self, img_names, input_size, mean, std, clahe):

        self.img_names = img_names
        self.size = input_size
        self.mean = mean
        self.std = std
        self.clahe = clahe

    def __len__(self):

        return len(self.img_names)

    def __getitem__(self, idx):

        img_file_name = self.img_names[idx]
        image = load_image(img_file_name, self.mean, self.std, self.clahe, self.size)
        
        cls = load_class(img_file_name)

        data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
        image = data_transforms(image)

        # mask , contour, dist = masks

        return image, cls

class DatasetImageFazMaskContourDist(Dataset):

    def __init__(self, img_names, gt_names, input_size, distance_type, mean, std, clahe):

        self.img_names = img_names
        self.gt_names = gt_names
        self.size = input_size
        self.distance_type = distance_type
        self.mean = mean
        self.std = std
        self.clahe = clahe

    def __len__(self):

        return len(self.img_names)

    def __getitem__(self, idx):

        img_file_name = self.img_names[idx]
        gt_file_name = self.gt_names[idx]
        image = load_image(img_file_name, self.mean, self.std, self.clahe, self.size)

        mask = load_mask(gt_file_name, self.size)[0, :, :]
        contour = load_contourheat(gt_file_name, self.size)[0, :, :]
        dist = load_distance(gt_file_name, self.distance_type, self.size)[0, :, :]

        image = (image - self.mean) / self.std
        masks = [mask , contour, dist]

        for i, mask in enumerate(masks):
            masks[i] = torch.from_numpy(np.expand_dims(mask, 0)).float()
        
        cls = load_class(img_file_name)

        data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
        image = data_transforms(image)

        mask , contour, dist = masks

        return image, mask, contour, dist, cls



class DatasetImageMaskContourDist(Dataset):

    def __init__(self, img_names, gt_names, input_size, distance_type, mean, std, clahe, augment=False):

        self.img_names = img_names
        self.gt_names = gt_names
        self.size = input_size
        self.distance_type = distance_type
        self.mean = mean
        self.std = std
        self.clahe = clahe
        self.augment = augment

    def __len__(self):

        return len(self.img_names)

    def __getitem__(self, idx):

        img_file_name = self.img_names[idx]
        gt_file_name = self.gt_names[idx]
        image = load_image(img_file_name, self.mean, self.std, self.clahe, self.size)

        mask = load_mask(gt_file_name, self.size)
        contour = load_contourheat(gt_file_name, self.size)
        dist = load_distance(gt_file_name, self.distance_type, self.size)

        image = (image - self.mean) / self.std
        if self.augment:
            image, mask, contour, dist = random_rot_flip(image, mask, contour, dist)
        
        cls = load_class(img_file_name)

        data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
        image = data_transforms(image)

        return image, mask, contour, dist, cls


def random_rot_flip(image, *maps):
    """Random 90-degree rotation plus horizontal/vertical flips (thesis section 3.3.3), applied identically
    to the image (H, W) and to every (C, H, W) target map."""
    k = random.randint(0, 3)
    flip_h, flip_v = random.random() < 0.5, random.random() < 0.5

    def apply(a, spatial_axes):
        a = np.rot90(a, k, axes=spatial_axes)
        if flip_h:
            a = np.flip(a, axis=spatial_axes[1])
        if flip_v:
            a = np.flip(a, axis=spatial_axes[0])
        return np.ascontiguousarray(a)

    return (apply(image, (0, 1)),) + tuple(apply(m, (1, 2)) for m in maps)


def load_image(path, mean, std, clahe, size):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img / img.max()
    # print(type(img))
    # if clahe:
    #     img = clahe_equalized(img)
    #     img = np.array(img ,dtype=np.float32)

    return img


def load_mask(path, size):

    # mask = cv2.imread(path.replace("image", "mask").replace("bmp", "bmp"), 0)
    vessel_path = path.replace("gt_full", "gt_vessel")
    faz_path = path.replace("gt_full", "gt_faz")
    vessel_mask = cv2.resize(cv2.imread(vessel_path, cv2.IMREAD_GRAYSCALE), size, interpolation = cv2.INTER_NEAREST)
    faz_mask = cv2.resize(cv2.imread(faz_path, cv2.IMREAD_GRAYSCALE), size, interpolation = cv2.INTER_NEAREST)
    vessel_mask = vessel_mask / vessel_mask.max()
    faz_mask = faz_mask / faz_mask.max()
    background_mask = np.ones(size) - (faz_mask + vessel_mask)

    combined_mask = np.stack([faz_mask, vessel_mask, background_mask], axis=0)
    combined_mask = combined_mask

    # mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # mask[mask == 255] = 1

    return combined_mask


def load_contourheat(path, size):

    # path = path.replace("mask", "contour_img").replace("bmp", "mat")
    
    # contour = cv2.imread(path)
    # contour = contour.mean(2)
    # contour[contour == 255] = 1
    # contour = io.loadmat(path)["contour"]

    # contour = cv2.imread(path.replace("mask", "contour_img").replace("bmp", "png"), cv2.IMREAD_GRAYSCALE)

    faz_contour = cv2.imread(path.replace("gt_full", "gt_faz").replace("mask", "contour_img").replace("bmp", "png"), cv2.IMREAD_GRAYSCALE)
    faz_contour = cv2.resize(faz_contour, size, interpolation = cv2.INTER_NEAREST)

    vessel_contour = cv2.imread(path.replace("gt_full", "gt_vessel").replace("mask", "contour_img").replace("bmp", "png"), cv2.IMREAD_GRAYSCALE)
    vessel_contour = cv2.resize(vessel_contour, size, interpolation = cv2.INTER_NEAREST)

    combined_mask = np.stack([faz_contour, vessel_contour], axis=0)
    combined_mask = combined_mask / combined_mask.max()

    return combined_mask


def load_class(path):

    cls0 = [1, 0, 0]
    cls1 = [0, 1, 0]
    cls2 = [0, 0, 1]
    if 'N' in os.path.basename(path):
        cls = cls0
    if 'D' in os.path.basename(path):
        cls = cls1
    if 'M' in os.path.basename(path):
        cls = cls2

    return torch.from_numpy(np.expand_dims(cls, 0)).long()


def load_distance(path, distance_type, size):
    """Signed-distance ground truth of the FAZ and vessel masks: array (2, H, W) scaled by its maximum.
    Only 'dist_signed11' (written by Preprocessing/build_processed_octa500.py) is supported; the other .mat based
    variants of the original BSDA-Net pipeline are not produced any more."""
    if distance_type != "dist_signed11":
        raise ValueError(f"unsupported distance_type '{distance_type}' (only 'dist_signed11')")

    faz_dist = cv2.imread(path.replace("gt_full", "gt_faz").replace("mask", "dist_signed11_img").replace("bmp", "png"), cv2.IMREAD_GRAYSCALE)
    faz_dist = cv2.resize(faz_dist, size, interpolation = cv2.INTER_NEAREST)

    vessel_dist = cv2.imread(path.replace("gt_full", "gt_vessel").replace("mask", "dist_signed11_img").replace("bmp", "png"), cv2.IMREAD_GRAYSCALE)
    vessel_dist = cv2.resize(vessel_dist, size, interpolation = cv2.INTER_NEAREST)

    combined_mask = np.stack([faz_dist, vessel_dist], axis=0)
    combined_mask = combined_mask / combined_mask.max()
    return combined_mask
