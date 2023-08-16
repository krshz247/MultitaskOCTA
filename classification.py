import time
import torch
import os
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import random
import cv2
import torchmetrics
from tensorboardX import SummaryWriter
import wandb
from utils import create_train_arg_parser, define_loss, generate_dataset, set_max_split_size_mb
from losses import My_multiLoss
import segmentation_models_pytorch as smp
import torchvision.models as models
import numpy as np
from sklearn.metrics import cohen_kappa_score

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def wb_mask(bg_img, pred_mask, true_mask, labels):
  return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels": labels},
    "ground truth" : {"mask_data" : true_mask, "class_labels": labels}})

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def seg_clf_iteration(epoch, model, optimizer, criterion, data_loader, device, loss_weights, startpoint, training=False):

    clf_losses = AverageMeter("Loss", ".16f")
    clf_accs = AverageMeter("Acc", ".8f")
    clf_kappas = AverageMeter("Kappa", ".8f")

    f1 = torchmetrics.F1Score(task="multiclass", num_classes=3).to(device)
    precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=3).to(device)
    recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=3).to(device)
    cohenkappa = torchmetrics.CohenKappa(task="multiclass", num_classes=3).to(device)
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3).to(device)

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(data_loader)):
        inputs = inputs.repeat(1, 3, 1, 1)
        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        class_names = ["Noraml", "DR", "AMD"]


        if training:
            optimizer.zero_grad()

        clf_outputs = model(inputs)

        clf_criterion = criterion

        # Classification evaluation metrics
        clf_labels = torch.argmax(targets[3], dim=2).squeeze(1)
        clf_preds = torch.argmax(clf_outputs, dim=1)

        clf_loss = clf_criterion(clf_outputs.squeeze().float(), targets[3].squeeze().float())
        kappa = cohen_kappa_score(clf_labels.detach().cpu().numpy(), clf_preds.detach().cpu().numpy())
        acc = np.mean(clf_labels.detach().cpu().numpy() == clf_preds.detach().cpu().numpy())


        f1_score = f1(clf_labels, clf_preds)
        Percision = precision(clf_labels, clf_preds)
        Recall = recall(clf_labels, clf_preds)
        conf_matrix = confmat(clf_labels, clf_preds)

        if training:

            loss = clf_loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

        clf_losses.update(clf_loss.item(), inputs.size(0))
        clf_accs.update(acc, inputs.size(0))
        clf_kappas.update(kappa, inputs.size(0))


    total_f1 = f1.compute()
    total_recall = recall.compute()
    total_precision = precision.compute()

    clf_epoch_loss = clf_losses.avg
    clf_epoch_acc = clf_accs.avg
    clf_epoch_kappa = clf_kappas.avg

    data = {
        "cls_loss": clf_epoch_loss,
        "cls_acc" : clf_epoch_acc,
        "cls_kappa" : clf_epoch_kappa,
        "cls_f1" : total_f1,
        "cls_recall" : total_recall,
        "cls_percision" : total_precision,
    }

    return data

class My_Resnet_50(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, num_classes=3):
        super(My_Resnet_50, self).__init__()

        resnet = models.resnet18(pretrained=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.model = resnet

    def forward(self, x):
        out_class = self.model(x)
        return out_class


def main():
    with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False, allow_tf32=False):
        torch.set_num_threads(4)
        set_seed(2021)

        args = create_train_arg_parser().parse_args()
        CUDA_SELECT = "cuda:{}".format(args.cuda_no)
        print("cuda_count:", torch.cuda.device_count())
        torch.cuda.empty_cache()

        log_path = os.path.join(args.save_path, "summary/")
        writer = SummaryWriter(log_dir=log_path)

        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_name = os.path.join(log_path, str(rq) + '.log')
        logging.basicConfig(
            filename=log_name,
            filemode="a",
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M",
            level=logging.INFO,
        )
        logging.info(args)

        device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

        # logging in Wandb
        if args.log_mode:
            wandb.init(
            project="BSDA-Net",
            dir = args.save_path,
            # Track hyperparameters and run metadata
            config={
                    "Encoder":args.encoder,
                    "Augmentation": args.augmentation,
                    "distance_type":args.distance_type,
                    "train_type": args.train_type,
                    "train_batch_size":args.batch_size,
                    "val_batch_size": args.val_batch_size,
                    "num_epochs": args.num_epochs,
                    "loss_type": "dice",
                    "LR_seg": args.LR_seg,
                    "LR_clf": args.LR_clf,
                    "pretrain":args.pretrain,
            })
            wandb.define_metric("epochs")
            wandb.define_metric("train*", step_metric="epochs")
            wandb.define_metric("val*", step_metric="epochs")

        encoder = args.encoder
        usenorm = args.usenorm
        attention_type = args.attention
        if args.pretrain in ['imagenet', 'ssl', 'swsl', 'instagram']:
            pretrain = args.pretrain
        else:
            pretrain = None

        model = My_Resnet_50()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)
        logging.info(model)

        weights = [0.49, 1.88, 2.35]
        class_weights = torch.FloatTensor(weights).cuda()        
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = Adam([
            {"params": model.module.parameters(), "lr": args.LR_clf}
        ])


        train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
        val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))
        random.shuffle(val_file_names)

        trainLoader, devLoader = generate_dataset(train_file_names, val_file_names, args.batch_size, args.val_batch_size, args.distance_type, args.clahe)

        epoch_start = 0
        max_dice = 0.8
        max_acc = 0.6
        loss_weights = [3, 1, 1]
        logging.info(loss_weights)
        startpoint = args.startpoint

        for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

            print('\nEpoch: {}'.format(epoch))
            train_data = seg_clf_iteration(epoch, model, optimizer, criterion, trainLoader, device, loss_weights, startpoint, training=True)
            dev_data = seg_clf_iteration(epoch, model, optimizer, criterion, devLoader, device, loss_weights, startpoint, training=False)

            epoch_info = "Epoch: {}".format(epoch)
            # train_info = f"Tr_SegLoss:{train_data["seg_loss"]}, Tr_MutiLoss:{train_data["multi_loss"]}"
            train_info = f" Train_Clf Loss:{train_data['cls_loss']}, Acc: {train_data['cls_acc']}, Kappa:{train_data['cls_kappa']}"
            val_info = f" Val_Clf Loss:{dev_data['cls_loss']}, Acc: {dev_data['cls_acc']}, Kappa:{dev_data['cls_kappa']}:"
            
            print(train_info)
            print(val_info)
            logging.info(epoch_info)
            logging.info(train_info)
            logging.info(val_info)
            
            if args.log_mode:
                wandb.log({"train": train_data, "epochs": epoch})
                wandb.log({"val": dev_data, "epochs": epoch})

            best_name = os.path.join(args.save_path, "acc_" + str(round(dev_data['cls_acc'], 4)) + "_kap_" + str(round(dev_data['cls_kappa'], 4)) + ".pt")
            save_name = os.path.join(args.save_path, str(epoch) + "acc_" + str(round(dev_data['cls_acc'], 4)) + "_kap_" + str(round(dev_data['cls_acc'], 4)) + ".pt")

            if max_acc <= dev_data['cls_acc']:
                max_acc = dev_data['cls_acc']
                # if epoch > 10:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best clf model saved!')
                logging.warning('Best clf model saved!')

            if epoch % 50 == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), save_name)
                    print('Epoch {} model saved!'.format(epoch))
                else:
                    torch.save(model.state_dict(), save_name)
                    print('Epoch {} model saved!'.format(epoch))

            wandb.finish

if __name__ == "__main__":
    main()
