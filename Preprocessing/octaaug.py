import albumentations as alb
import matplotlib.pyplot as plt
# from albumentations.pytorch import ToTensor
import os, shutil
import cv2
import numpy as np
import random



def mkdir(path):
    # remove leading spaces
    path = path.strip()
    # remove the tail \ symbol
    path = path.rstrip("\\")
    # Determine whether the path exists
    # exist     True
    # does not exist   False
    isExists = os.path.exists(path)
    # critical result
    if not isExists:
        # Create the directory if it does not exist
        # Create a directory operation function
        os.makedirs(path)
        print(path + ' created successfully')
        return True
    else:
        # If the directory exists, it will not be created, and it will prompt that the directory already exists
        print(path + ' directory already exists')
        return False



if __name__ == "__main__":

    basename = './Datasets/processed_OCTA500/combined/'
    IMG_DIR = basename + "image"
    MASK_DIR = basename + "mask"

    # AUG_MASK_DIR = basename.replace('_ori/','_aug/') + "mask"  # Store enhanced XML folder path
    AUG_MASK_DIR = basename + "aug_mask"
    try:
        shutil.rmtree(AUG_MASK_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_MASK_DIR)

    # AUG_IMG_DIR = basename.replace('_ori/','_aug/') + "image" # Store the enhanced image folder path
    AUG_IMG_DIR = basename + "aug_img"
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUGCROP_IMG_DIR = basename + "augcrre_img"
    try:
        shutil.rmtree(AUGCROP_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUGCROP_IMG_DIR)

    AUGCROP_MA_DIR = basename + "augcrre_mask"
    try:
        shutil.rmtree(AUGCROP_MA_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUGCROP_MA_DIR)

    AUGLOOP = 30  # Number of enhancements per image

    aug = alb.Compose([
        alb.RandomRotate90(),
      # albu.Cutout(),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.OneOf([

          alb.augmentations.transforms.CLAHE(clip_limit=3),
          # A.augmentations.transforms.Downscale(scale_min=0.45, scale_max=0.95),
          
          alb.augmentations.transforms.GaussNoise(var_limit=(20.0)),
          alb.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
          # A.imgaug.transforms.IAACropAndPad(percent=0.3, pad_mode="reflect"),
          alb.imgaug.transforms.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
          # A.imgaug.transforms.IAACropAndPad(percent=0.3, pad_mode="reflect"),
          alb.augmentations.transforms.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
          alb.imgaug.transforms.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003, 12.75), per_channel=False, always_apply=False, p=0.5)
        ], p= 0.6),
        alb.OneOf([
          alb.Blur(blur_limit=3),
          alb.GaussianBlur(blur_limit=3, sigma_limit=0, always_apply=False, p=0.5),
          alb.MedianBlur(blur_limit=3, always_apply=False, p=0.5),
          # A.augmentations.transforms.Blur(blur_limit=3, always_apply=False, p=0.5),
        #   A.augmentations.transforms.MotionBlur(blur_limit=3),
          # A.augmentations.transforms.GlassBlur (sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5)
          # A.argumentations.transforms.
        ]),
        alb.imgaug.transforms.IAAAffine(scale=1.0, translate_percent=0, translate_px=None, rotate=(-90, 90), shear=0.0, order=1, cval=0, mode='reflect'),
        # A.augmentations.geometric.rotate.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
        # A.augmentations.transforms.PadIfNeeded (min_height=224, min_width=224, pad_height_divisor=None, pad_width_divisor=None, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0)
    ], p=1)

    random.seed(2021)  #seed
    size = 192

    for root, sub_folders, files in os.walk(MASK_DIR):

        for name in files:
            print(name)

            '''bndbox = read_xml_annotation(MASK_DIR, name)
            shutil.copy(os.path.join(MASK_DIR, name), AUG_MASK_DIR)
            '''
            # shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.png'), AUG_IMG_DIR)

            for epoch in range(AUGLOOP):
                # seq_det = seq.to_deterministic()  # Keep coordinates and images changing synchronously, not randomly read pictures

                img = cv2.imread(os.path.join(IMG_DIR, name[:-4] + '.png'), cv2.IMREAD_GRAYSCALE)
                # sp = img.size

                mask = cv2.imread(os.path.join(MASK_DIR, name[:-4] + '.png'), cv2.IMREAD_GRAYSCALE)
                augmented = aug(image=img, mask=mask)

                img_aug = augmented['image']
                mask_aug = augmented['mask']
                
                ret, mask_aug_thres = cv2.threshold(mask_aug,127,255,cv2.THRESH_BINARY)

                img_aug_path = os.path.join(AUG_IMG_DIR, name[:-4] + '_' + str(epoch+1) + '.png')
                mask_aug_path = os.path.join(AUG_MASK_DIR, name[:-4] + '_' + str(epoch+1) + '.png')
                cv2.imwrite(img_aug_path, img_aug)
                cv2.imwrite(mask_aug_path, mask_aug_thres)

                if ('_3_' in name) or ('area3' in name):
                    img_re = cv2.resize(img_aug, (size, size), interpolation=cv2.INTER_CUBIC)
                    mask_re = cv2.resize(mask_aug, (size, size), interpolation=cv2.INTER_CUBIC)
                    # mask_re = cv2.threshold(mask_re,127,255,cv2.THRESH_BINARY)
                elif ('_6_' in name) or ('area6' in name):
                    h, w = img_aug.shape
                    # a = w/2-size/2
                    # b = w/2+size/2
                    img_re = img_aug[100:300,100:300]
                    mask_re = mask_aug_thres[100:300,100:300]
                    img_re = cv2.resize(img_re, (size, size), interpolation=cv2.INTER_CUBIC)
                    mask_re = cv2.resize(mask_re, (size, size), interpolation=cv2.INTER_CUBIC)
                img_re_path = os.path.join(AUGCROP_IMG_DIR, name[:-4] + '_' + str(epoch+1) + '.png')
                mask_re_path = os.path.join(AUGCROP_MA_DIR, name[:-4] + '_' + str(epoch+1) + '.png')
                cv2.imwrite(img_re_path, img_re)
                cv2.imwrite(mask_re_path, mask_re)