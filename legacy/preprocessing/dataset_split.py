import os
import shutil
import random

def split_dataset(source_dir, img_mask_folders, train_ratio, val_ratio, test_ratio):

    # Get the list of files in the source directory
    file_list = os.listdir(os.path.join(source_dir, img_mask_folders[0]))
    random.shuffle(file_list)

    total_files = len(file_list)
    train_split = int(train_ratio * total_files)
    val_split = int(val_ratio * total_files)

    train_slice = slice(0, train_split,1)
    val_slice = slice(train_split, train_split+val_split,1)
    test_split = slice(train_split+val_split, total_files,1)

    # Create destination directories
    split = {'train':train_slice, 'val':val_slice, 'test':test_split}
    for n in split.keys():
        for im in img_mask_folders:
            path = os.path.join(source_dir, n, im)
            os.makedirs(path, exist_ok=True)

    # Move files to train directory
    for n,s in split.items():
        for im in img_mask_folders:
            for file_name in file_list[s]:
                src_path = os.path.join(source_dir, im, file_name)
                dst_path = os.path.join(source_dir, n, im, file_name)
                shutil.move(src_path, dst_path)

    print("Dataset split completed successfully.")


# Example usage
dataset_folder = './Datasets/processed_OCTA500/combined'  # Replace with your dataset folder path
img_mask_folders = ['aug_img', 'aug_mask']

train_ratio = 0.7  # 70% for training
val_ratio = 0.2  # 20% for validation
test_ratio = 0.1  # 10% for testing

split_dataset(dataset_folder, img_mask_folders, train_ratio, val_ratio, test_ratio)
