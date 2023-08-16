# -*- coding: utf-8 -*-
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
from utils import create_train_arg_parser, define_loss, generate_dataset
from losses import My_multiLoss
import segmentation_models_pytorch as smp
import torchvision.models as models
import numpy as np
import matplotlib
from sklearn.metrics import cohen_kappa_score
from smp_model import MyUnetModel, my_get_encoder, MyMultibranchModel

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.allow_tf32 = True
matplotlib.use('Agg')

IN_MODELS = ['unet_smp', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+']

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


def segmentation_iteration(model, optimizer, model_type, criterion, data_loader, device, writer, training=False):
    running_loss = 0.0
    total_size = 0

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        if training:
            optimizer.zero_grad()

        outputs = model(inputs)

        if model_type in IN_MODELS + ["unet"]:
            if not isinstance(outputs, list):
                outputs = [outputs]
            loss = criterion(outputs[0], targets[0])
            dsc_loss = smp.utils.losses.DiceLoss()
            # 只写了一个例子
            # preds = torch.argmax(outputs[0].exp(), dim=1)
            preds = torch.argmax(torch.sigmoid(outputs[0]), dim=1)
            dsc = 1 - dsc_loss(preds, targets[0].squeeze(1))

        elif model_type == "dcan":
            loss = criterion(outputs[0], outputs[1], targets[0], targets[1])

        elif model_type == "dmtn":
            loss = criterion(outputs[0], outputs[1], targets[0], targets[2])

        elif model_type in ["psinet", "convmcd"]:
            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2]
            )
        else:
            raise ValueError('error')

        if training:
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        total_size += inputs.size(0)

    epoch_loss = running_loss / total_size
    # print("total size:", total_size, training)

    return epoch_loss, dsc


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
    seg_losses = AverageMeter("Loss", ".16f")
    multi_losses = AverageMeter("multiLoss", ".16f")
    seg_dices = AverageMeter("Dice_loss", ".8f")
    dice_coefs = AverageMeter("Dice_coeff", ".8f")
    seg_jaccards = AverageMeter("Jaccard", ".8f")
    clf_losses = AverageMeter("Loss", ".16f")
    clf_accs = AverageMeter("Acc", ".8f")
    clf_kappas = AverageMeter("Kappa", ".8f")

    f1 = torchmetrics.F1Score(task="multiclass", num_classes=3).to(device)
    precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=3).to(device)
    recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=3).to(device)
    cohenkappa = torchmetrics.CohenKappa(task="multiclass", num_classes=3).to(device)
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3).to(device)

    mask_list = []

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    if epoch <= startpoint:
            for params in model.module.clf_model.parameters():
                params.requires_grad = False

    else:
        for params in model.module.clf_model.parameters():
            params.requires_grad = True

    for i, (inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(data_loader)):

        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        class_names = ["Noraml", "DR", "AMD"]


        if training:
            optimizer.zero_grad()


        seg_outputs = model.module.seg_forward(inputs)
        if not isinstance(seg_outputs, list):
            seg_outputs = [seg_outputs]

        seg_preds = torch.round(seg_outputs[0])
        clf_outputs = model.module.clf_forward(inputs, seg_outputs[3], seg_outputs[4], seg_outputs[5])


        seg_criterion, dice_criterion, jaccard_criterion, clf_criterion = criterion[0], criterion[1], criterion[2], criterion[3]

        # Multi-Segmentation loss
        multi_criterion = My_multiLoss(loss_weights)
        multi_loss = multi_criterion(seg_outputs[0], seg_outputs[1], seg_outputs[2], targets[0].to(torch.float32), targets[1], targets[2])

        # Segmentation evaluation metrics
        seg_loss = seg_criterion(seg_outputs[0], targets[0].to(torch.int))
        seg_dice = dice_criterion(seg_preds.squeeze(1), targets[0].squeeze(1))
        seg_jaccard = jaccard_criterion(seg_preds.squeeze(1), targets[0].squeeze(1))

        dice = torchmetrics.Dice().to(device)
        dice_coef = dice(seg_preds.squeeze(1), targets[0].squeeze(1).to(torch.int))

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
            if epoch <= startpoint:
                # loss = seg_loss
                loss = multi_loss
            else:
                # loss = (seg_loss + clf_loss)
                loss = (multi_loss + clf_loss)
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            # scheduler.step()

        seg_losses.update(seg_loss.item(), inputs.size(0))
        multi_losses.update(multi_loss.item(), inputs.size(0))

        seg_dices.update(seg_dice.item(), inputs.size(0))
        dice_coefs.update(dice_coef.item(), inputs.size(0))
        seg_jaccards.update(seg_jaccard.item(), inputs.size(0))
        clf_losses.update(clf_loss.item(), inputs.size(0))
        clf_accs.update(acc, inputs.size(0))
        clf_kappas.update(kappa, inputs.size(0))

    if not training:
        image = inputs[0][0].cpu().detach().numpy()
        mask_pred = seg_outputs[0][0][0].cpu().detach().numpy()
        mask_pred = cv2.threshold(mask_pred, 0.7, 1, cv2.THRESH_BINARY)[1]
        # mask_pred = np.where(mask_pred > 0.5, 1, 0)
        # boundary = seg_outputs[0][0][0].cpu().detach().numpy()
        # dist = seg_outputs[0][0][0].cpu().detach().numpy()

        mask_gt = targets1[0][0].cpu().detach().numpy()
        labels = {1:"FAZ"}
        # label= targets4[0].cpu().detach().numpy()
        # boundary_gt = targets2[0][0].cpu().detach().numpy()   
        # dist_gt = targets3[0][0].cpu().detach().numpy()

        mask_log = wb_mask(image, mask_pred, mask_gt, labels)

        # log all composite images to W&B
        wandb.log({"predictions" : mask_log})
        if epoch >= startpoint:
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=clf_labels.cpu().detach().numpy(), 
                            preds=clf_preds.cpu().detach().numpy(),
                            class_names=class_names)})

    total_f1 = f1.compute()
    total_recall = recall.compute()
    total_precision = precision.compute()

    seg_epoch_loss = seg_losses.avg
    multi_epoch_loss = multi_losses.avg
    seg_epoch_dice = seg_dices.avg
    dice_coef_epoch = dice_coefs.avg
    seg_epoch_jaccard = seg_jaccards.avg
    clf_epoch_loss = clf_losses.avg
    clf_epoch_acc = clf_accs.avg
    clf_epoch_kappa = clf_kappas.avg

    data = {
        "multi_loss" : multi_epoch_loss,
        "seg_loss" : seg_epoch_loss,
        "seg_dice_loss" : seg_epoch_dice,
        "seg_dice_coef" : dice_coef_epoch,
        "seg_jaccard_loss" : seg_epoch_jaccard,
        "cls_loss": clf_epoch_loss,
        "cls_acc" : clf_epoch_acc,
        "cls_kappa" : clf_epoch_kappa,
        "cls_f1" : total_f1,
        "cls_recall" : total_recall,
        "cls_percision" : total_precision,
    }

    return data


class CotrainingModel(nn.Module):

    def __init__(self, encoder, pretrain, classnum):
        super().__init__()
        self.seg_model = MyUnetModel(
            encoder_name=encoder, encoder_depth=5, encoder_weights=pretrain, decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=1, classes=1,
            activation='sigmoid', aux_params=None
        )
        self.clf_model = my_get_encoder(encoder, in_channels=1, depth=5, weights=pretrain, num_classes=classnum)

    def seg_forward(self, x):
        return self.seg_model(x)

    def clf_forward(self, x, decoder_features):
        return self.clf_model(x, decoder_features)


class CotrainingModelMulti(nn.Module):

    def __init__(self, encoder, pretrain, usenorm, attention_type, classnum):
        super().__init__()
        self.seg_model = MyMultibranchModel(
            encoder_name=encoder, encoder_depth=5, encoder_weights=pretrain, decoder_use_batchnorm=usenorm,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=attention_type, in_channels=1, classes=1,
            activation='sigmoid', aux_params=None
        )
        self.clf_model = my_get_encoder(encoder, in_channels=1, depth=5, weights=pretrain, decoder_channels=(256, 128, 64, 32, 16), num_classes=classnum)

    def seg_forward(self, x):
        return self.seg_model(x)

    def clf_forward(self, x, decoder1_features, decoder2_features, decoder3_features):
        return self.clf_model(x, decoder1_features, decoder2_features, decoder3_features)


def main():
    with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False, allow_tf32=False):
        torch.set_num_threads(4)
        set_seed(2021)

        args = create_train_arg_parser().parse_args()
        CUDA_SELECT = "cuda:{}".format(args.cuda_no)
        print("cuda_count:", torch.cuda.device_count())

        log_path = os.path.join(args.save_path, "summary/")
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
                    "img_depth" : args.img_path.split("/")[-1],
                    "gt_type": args.gt_path.split("/")[-1],
                    "Encoder":args.encoder,
                    "Augmentation": args.augmentation,
                    "distance_type":args.distance_type,
                    "train_type": args.train_type,
                    "train_batch_size":args.batch_size,
                    "val_batch_size": args.val_batch_size,
                    "num_epochs": args.num_epochs,
                    "loss_type": args.loss_type,
                    "startpoint": args.startpoint,
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

        model = CotrainingModelMulti(encoder, pretrain, usenorm, attention_type, args.classnum) 

        # model= nn.DataParallel(model)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # device = torch.device("cuda:1") 
        model.to(device)
        logging.info(model)

        weights = [0.49, 1.88, 2.35]
        class_weights = torch.FloatTensor(weights).cuda()        
        criterion = [
            define_loss(args.loss_type),
            smp.losses.DiceLoss("binary"),
            smp.losses.JaccardLoss("binary"),
            torch.nn.CrossEntropyLoss()
        ]

        optimizer = Adam([
            {"params": model.module.seg_model.parameters(), "lr": args.LR_seg},
            {"params": model.module.clf_model.parameters(), "lr": args.LR_clf}
    ])


        # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        if args.train_type == "Vessel_FAZ":
            img_names = glob.glob(os.path.join(args.img_path, "*.bmp"))
            gt_names = list()
            random.shuffle(img_names)
            for name in img_names:
                gt_names.append(name.replace("image_surface", args.gt_path.split("/")[-2] + "/mask"))

            train_index = int(len(img_names) * args.train_percentage)
            val_index = int(len(img_names) * (args.train_percentage + args.val_percentage))
            train_img_names = img_names[:train_index]
            val_img_names = img_names[train_index : val_index]

            train_gt_names = gt_names[:train_index]
            val_gt_names = gt_names[train_index : train_index + val_index]
        else:
            train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
            val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))
            random.shuffle(val_file_names)

        trainLoader, devLoader = generate_dataset(train_img_names, train_gt_names, val_img_names, val_gt_names, args.batch_size, args.val_batch_size, args.distance_type, args.clahe)

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
            train_info = f"TrainSeg Loss:{train_data['seg_loss']}, TrMutiLoss:{train_data['multi_loss']}, Dice_Loss: {train_data['seg_dice_loss']}, Dice_Coeff: {train_data['seg_dice_coef']}, Jaccard: {train_data['seg_jaccard_loss']}, TrainClf Loss:{train_data['cls_loss']}, Acc: {train_data['cls_acc']}, Kappa:{train_data['cls_kappa']}"
            val_info = f"ValSeg Loss:{dev_data['seg_loss']}, VaMutiLoss:{dev_data['multi_loss']}, Dice_Loss: {train_data['seg_dice_loss']}, Dice_Coeff: {train_data['seg_dice_coef']}, Jaccard: {dev_data['seg_jaccard_loss']}, ValClf Loss:{dev_data['cls_loss']}, Acc: {dev_data['cls_acc']}, Kappa:{dev_data['cls_kappa']}:"
            
            print(train_info)
            print(val_info)
            logging.info(epoch_info)
            logging.info(train_info)
            logging.info(val_info)
            
            if args.log_mode:
                wandb.log({"train": train_data, "epochs": epoch})
                wandb.log({"val": dev_data, "epochs": epoch})

            best_name = os.path.join(args.save_path, "dice_loss_" + str(round(dev_data['seg_dice_loss'], 5)) + "_jaccard_" + str(round(dev_data['seg_jaccard_loss'], 5)) + "_acc_" + str(round(dev_data['cls_acc'], 4)) + "_kap_" + str(round(dev_data['cls_kappa'], 4)) + ".pt")
            save_name = os.path.join(args.save_path, str(epoch) + "_dice_loss_" + str(round(dev_data['seg_dice_loss'], 5)) + "_jaccard_" + str(round(dev_data['seg_jaccard_loss'], 5)) + "_acc_" + str(round(dev_data['cls_acc'], 4)) + "_kap_" + str(round(dev_data['cls_acc'], 4)) + ".pt")

            if max_dice <= dev_data['seg_dice_loss']:
                max_dice = dev_data['seg_dice_loss']
                # if epoch > 10:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best seg model saved!')
                logging.warning('Best seg model saved!')
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
