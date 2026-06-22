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
from losses import FAZ_multiLoss
import segmentation_models_pytorch as smp
import torchvision.models as models
from losses import FAZ_multiLoss
import numpy as np
import matplotlib
from sklearn.metrics import cohen_kappa_score
from smp_model import MyUnetModel, my_get_encoder, MyMultibranchModel
from sklearn.metrics import confusion_matrix

IN_MODELS = ['unet_smp', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+']

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def wb_mask(image, pred_mask, true_mask, labels):
  
    return wandb.Image(image, masks={
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


def seg_clf_iteration(epoch, model, optimizer, criterion, data_loader, device, loss_weights, startpoint, batch_size, training=False, test=False):

    stage = ''
    if test:
        stage = 'test'
    elif training:
        stage = 'train'
    else:
        stage = 'val'

    conf_pred = []
    conf_gt = []

    seg_losses = AverageMeter("Loss", ".16f")
    multi_losses = AverageMeter("multiLoss", ".16f")

    vessel_cldice_losses = AverageMeter("vessel_cldice", ".16f")
    faz_dice_losses = AverageMeter("faz_dice", ".16f")
    background_dice_losses = AverageMeter("background_dice", ".16f")

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
    # confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3).to(device)

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
        
        inputs = inputs.to(torch.float32).to(device)
        targets1, targets2 = targets1.to(torch.float32).to(device), targets2.to(torch.float32).to(device)
        targets3, targets4 = targets3.to(torch.float32).to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        seg_target = targets[0]
        cls_target = targets[3]

        if training:
            optimizer.zero_grad()


        seg_outputs = model.module.seg_forward(inputs)
        if not isinstance(seg_outputs, list):
            seg_outputs = [seg_outputs]

        seg_preds = torch.round(seg_outputs[0])

        # clf_outputs = model.module.clf_forward(inputs, seg_outputs[3], seg_outputs[4], seg_outputs[5])

        # dice_criterion, jaccard_criterion, clf_criterion = criterion[0], criterion[1], criterion[2]

        # # Segmentation evaluation metrics
        # seg_jaccard = jaccard_criterion(seg_preds, targets[0])
        # dice_coef = dice_criterion(seg_preds.squeeze(1), targets[0].squeeze(1).to(torch.int))

        # # Multi-Segmentation loss
        # multi_criterion = FAZ_multiLoss(loss_weights)
        # multi_loss, seg_loss = multi_criterion(seg_outputs[0], seg_outputs[1], seg_outputs[2], targets[0].to(torch.float32), targets[1], targets[2])

        # # Classification evaluation metrics
        # clf_labels = torch.argmax(targets[3], dim=2).squeeze(1)
        # clf_preds = torch.argmax(clf_outputs, dim=1)

        # clf_loss = clf_criterion(clf_outputs.squeeze(1).float(), targets[3].squeeze(1).float())
        # kappa = cohen_kappa_score(clf_labels.detach().cpu().numpy(), clf_preds.detach().cpu().numpy())
        # acc = np.mean(clf_labels.detach().cpu().numpy() == clf_preds.detach().cpu().numpy())

        seg_criterion, dice_criterion, jaccard_criterion, clf_criterion = criterion[0], criterion[1], criterion[2], criterion[3]
        dice = torchmetrics.Dice().to(device)

        # Segmentation evaluation metrics
        seg_loss = seg_criterion(seg_outputs[0], seg_target.to(torch.int))
        seg_jaccard = jaccard_criterion(seg_preds.squeeze(1), seg_target.squeeze(1))
        dice_coef = dice(seg_preds.squeeze(1), seg_target.squeeze(1).to(torch.int))

        # Multi-Segmentation loss
        multi_criterion = FAZ_multiLoss(loss_weights)
        multi_loss = multi_criterion(seg_outputs[0], seg_outputs[1], seg_outputs[2], seg_target, targets[1], targets[2])

        # Classification evaluation metrics
        clf_outputs = model.module.clf_forward(inputs, seg_outputs[3], seg_outputs[4], seg_outputs[5])
        clf_labels = torch.argmax(cls_target, dim=2).squeeze(1)
        clf_preds = torch.argmax(clf_outputs, dim=1)

        clf_loss = clf_criterion(clf_outputs.squeeze(1).float(), cls_target.squeeze(1).float())
        kappa = cohen_kappa_score(clf_labels.detach().cpu().numpy(), clf_preds.detach().cpu().numpy(), labels=[0,1,2])
        # kappa = cohen_kappa_score(clf_labels.detach().cpu().numpy(), clf_preds.detach().cpu().numpy(), labels=[0,1,2])
        # kappa = cohen_kappa_score(clf_labels.detach().cpu().numpy(), clf_preds.detach().cpu().numpy())
        acc = np.mean(clf_labels.detach().cpu().numpy() == clf_preds.detach().cpu().numpy())

        conf_gt = np.concatenate((conf_gt, clf_labels.detach().cpu().numpy()))
        conf_pred = np.concatenate((conf_pred, clf_preds.detach().cpu().numpy()))

        f1_score = f1(clf_labels, clf_preds)
        Percision = precision(clf_labels, clf_preds)
        Recall = recall(clf_labels, clf_preds)
        # conf_matrix = confmat(clf_labels, clf_preds)

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

        # seg_dices.update(seg_dice.item(), inputs.size(0))
        dice_coefs.update(dice_coef.item(), inputs.size(0))
        seg_jaccards.update(seg_jaccard.item(), inputs.size(0))
        clf_losses.update(clf_loss.item(), inputs.size(0))
        clf_accs.update(acc, inputs.size(0))
        clf_kappas.update(kappa, inputs.size(0))

  
    # Wandb multi-label segmentation log
    batch_images = inputs.squeeze().cpu().detach().numpy()
    batch_pred = seg_preds.squeeze().cpu().detach().numpy()

    mask_gt = seg_target.squeeze().cpu().detach().numpy()

    labels = {0:"Background",1:"FAZ"}
    idx = 0
    mask_log = wb_mask(batch_images[idx, :, :], batch_pred[idx, :, :], mask_gt[idx, :, :], labels)


    total_f1 = f1.compute()
    total_recall = recall.compute()
    total_precision = precision.compute()

    seg_epoch_loss = seg_losses.avg

    multi_epoch_loss = multi_losses.avg
    # vessel_cldice_epoch_loss = vessel_cldice_losses.avg
    # faz_dice_epoch_loss = faz_dice_losses.avg
    # background_dice_epoch_loss = background_dice_losses.avg
    dice_coef_epoch = dice_coefs.avg
    seg_epoch_jaccard = seg_jaccards.avg
    clf_epoch_loss = clf_losses.avg
    clf_epoch_acc = clf_accs.avg
    clf_epoch_kappa = clf_kappas.avg

    print("Conf Matrix", stage)
    print(confusion_matrix(conf_gt, conf_pred))
    wandb.log({"conf_mat_" +  stage : wandb.plot.confusion_matrix(probs=None,
                        y_true=conf_gt, preds=conf_pred,
                        class_names=["Normal", "DR", "AMD"])})

    data = {
        "multi_loss" : multi_epoch_loss,
        "seg_loss" : seg_epoch_loss,
        # "vessel_clDice_loss": vessel_cldice_epoch_loss,
        # "faz_dice_loss": faz_dice_epoch_loss,
        # "background_dice_loss": background_dice_epoch_loss,
        "seg_dice_coef" : dice_coef_epoch,
        "seg_jaccard_loss" : seg_epoch_jaccard,
        "cls_loss": clf_epoch_loss,
        "cls_acc" : clf_epoch_acc,
        "cls_kappa" : clf_epoch_kappa,
        "cls_f1" : total_f1,
        "cls_recall" : total_recall,
        "cls_percision" : total_precision,
    }

    return data, mask_log


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
            decoder_attention_type=attention_type, in_channels=1, out_channel= 1, classes=1,
            activation="sigmoid", aux_params=None
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

        model = nn.DataParallel(model)

        # device = torch.device("cuda:1") 
        model.to(device)
        logging.info(model)

        weights = [0.49, 1.88, 2.35]
        class_weights = torch.FloatTensor(weights).cuda()        
        criterion = [
            smp.losses.DiceLoss("binary"),
            smp.losses.DiceLoss("binary"),
            smp.losses.JaccardLoss("binary"),
            torch.nn.CrossEntropyLoss()
        ]

        optimizer = Adam([
            {"params": model.module.seg_model.parameters(), "lr": args.LR_seg},
            {"params": model.module.clf_model.parameters(), "lr": args.LR_clf}
    ])


        # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        img_names = glob.glob(os.path.join(args.img_path, "*.bmp"))
        gt_names = list()
        random.shuffle(img_names)
        for name in img_names:
            gt_names.append(name.replace("image_surface", args.gt_path.split("/")[-2] + "/mask"))

        train_end_index = int(len(img_names) * args.train_percentage)
        val_end_index = int(len(img_names) * (args.train_percentage + args.val_percentage))
        train_img_names = img_names[:train_end_index]
        val_img_names = img_names[train_end_index : val_end_index]

        train_gt_names = gt_names[:train_end_index]
        val_gt_names = gt_names[train_end_index : val_end_index]

        test_img_names = img_names[val_end_index:]
        test_gt_names = gt_names[val_end_index:]

        trainLoader, devLoader, testLoader = generate_dataset(train_img_names, val_img_names, test_img_names, args.input_size, args.batch_size, args.val_batch_size, args.distance_type, args.clahe, args.train_type, train_gt_names, val_gt_names, test_gt_names)

        epoch_start = 0
        max_dice = 0.8
        max_acc = 0.6
        loss_weights = [3, 1, 1]
        logging.info(loss_weights)
        startpoint = args.startpoint

        for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

            print('\nEpoch: {}'.format(epoch))
            train_data , train_seg_log = seg_clf_iteration(epoch, model, optimizer, criterion, trainLoader, device, loss_weights, startpoint, args.batch_size, training=True)
            dev_data, val_seg_log = seg_clf_iteration(epoch, model, optimizer, criterion, devLoader, device, loss_weights, startpoint, args.val_batch_size, training=False)
            _, _ = seg_clf_iteration(epoch, model, optimizer, criterion, testLoader, device, loss_weights, startpoint, args.val_batch_size, training=False, test=True)

            epoch_info = "Epoch: {}".format(epoch)
            # train_info = f"Tr_SegLoss:{train_data["seg_loss"]}, Tr_MutiLoss:{train_data["multi_loss"]}"
            train_info = f"TrainSeg Loss:{train_data['seg_loss']}, TrMutiLoss:{train_data['multi_loss']}, Dice_Coeff: {train_data['seg_dice_coef']}, Jaccard: {train_data['seg_jaccard_loss']}, TrainClf Loss:{train_data['cls_loss']}, Acc: {train_data['cls_acc']}, Kappa:{train_data['cls_kappa']}"
            val_info = f"ValSeg Loss:{dev_data['seg_loss']}, VaMutiLoss:{dev_data['multi_loss']}, Dice_Coeff: {train_data['seg_dice_coef']}, Jaccard: {dev_data['seg_jaccard_loss']}, ValClf Loss:{dev_data['cls_loss']}, Acc: {dev_data['cls_acc']}, Kappa:{dev_data['cls_kappa']}:"
            
            print(train_info)
            print(val_info)
            logging.info(epoch_info)
            logging.info(train_info)
            logging.info(val_info)
            
            if args.log_mode:
                wandb.log({"train" : train_seg_log, "epochs": epoch})
                wandb.log({"validation" : val_seg_log, "epochs": epoch})

                wandb.log({"train": train_data, "epochs": epoch})
                wandb.log({"val": dev_data, "epochs": epoch})

            best_name = os.path.join(args.save_path, "dice_loss_"  + "_jaccard_" + str(round(dev_data['seg_jaccard_loss'], 5)) + "_acc_" + str(round(dev_data['cls_acc'], 4)) + "_kap_" + str(round(dev_data['cls_kappa'], 4)) + ".pt")
            save_name = os.path.join(args.save_path, str(epoch)  + "_jaccard_" + str(round(dev_data['seg_jaccard_loss'], 5)) + "_acc_" + str(round(dev_data['cls_acc'], 4)) + "_kap_" + str(round(dev_data['cls_acc'], 4)) + ".pt")

            if max_dice <= dev_data['seg_loss']:
                max_dice = dev_data['seg_loss']
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

        wandb.finish()

if __name__ == "__main__":
    main()
