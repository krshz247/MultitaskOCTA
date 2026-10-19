from torch.utils.data import DataLoader
from octa.data.dataset import DatasetImageMaskContourDist, mean_and_std, DatasetImageFazMaskContourDist, DatasetImageClassification


def generate_dataset(train_img_names, val_img_names, test_img_names, input_size, batch_size, val_batch_size, distance_type, do_clahe, train_type, train_gt_names=None, val_gt_names=None, test_gt_names=None, num_workers=0, full_eval=False):

    train_mean, train_std = mean_and_std(train_img_names)

    if train_type == "Vessel_FAZ":

        train_dataset = DatasetImageMaskContourDist(train_img_names, train_gt_names, input_size, distance_type, train_mean, train_std, do_clahe, augment=True)
        valid_dataset = DatasetImageMaskContourDist(val_img_names, val_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)
        test_dataset = DatasetImageMaskContourDist(test_img_names, test_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)

    elif train_type == "FAZ":

        train_dataset = DatasetImageFazMaskContourDist(train_img_names, train_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)
        valid_dataset = DatasetImageFazMaskContourDist(val_img_names, val_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)
        test_dataset = DatasetImageFazMaskContourDist(test_img_names, test_gt_names, input_size, distance_type, train_mean, train_std, do_clahe)

    elif train_type == "CLS":
        train_dataset = DatasetImageClassification(train_img_names, input_size, train_mean, train_std, do_clahe)
        valid_dataset = DatasetImageClassification(val_img_names, input_size, train_mean, train_std, do_clahe)
        test_dataset = DatasetImageClassification(test_img_names, input_size, train_mean, train_std, do_clahe)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, drop_last=not full_eval, num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, drop_last=not full_eval, num_workers=num_workers, persistent_workers=num_workers > 0)

    return train_loader, valid_loader, test_loader
