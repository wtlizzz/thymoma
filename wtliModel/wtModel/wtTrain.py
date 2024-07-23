import numpy as np
import torch
import time
import os
from torch import nn
from scipy import ndimage
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tqdm import tqdm
from utils.logger import log
from wtModel.wtModel import generate_model
from wtModel.wtEarlyStop import WtES


def wtTrain(
    data_loader_train,
    data_loader_val,
    summaryWriter,
    model,
    optimizer,
    scheduler,
    criterion,
    total_epochs,
    save_interval,
    save_folder,
    sets,
):
    # settings
    batches_per_epoch = len(data_loader_train)
    log.info(
        "{} epochs in total, {} batches per epoch".format(
            total_epochs, batches_per_epoch
        )
    )
    wtes = WtES(save_folder)
    # device = torch.device("cpu")
    print(f"==sets.gpu_id = {sets.gpu_id}")
    if torch.cuda.is_available():
        device = torch.device("cuda", sets.gpu_id)
    else:
        device = torch.device("cpu")
    print(f"device = {device}")
    for epoch in range(total_epochs):
        start = time.time()
        per_epoch_loss = 0
        num_correct = 0
        score_list = []
        label_list = []

        val_num_correct = 0
        val_score_list = []
        val_label_list = []

        model.train()
        with torch.enable_grad():
            for x, label, img_name in tqdm(data_loader_train):
                x = x.float()
                x = x.to(device)
                is_True = False
                label_list.extend(label.numpy())
                label = label.to(device)
                logits = model(x)
                logits = logits.reshape([label.cpu().numpy().shape[0]])
                prob_out = nn.Sigmoid()(logits)
                pro_list = prob_out.detach().cpu().numpy()
                for i in range(pro_list.shape[0]):
                    if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
                        is_True = True
                        num_correct += 1
                score_list.extend(pro_list)
                logits = logits.to(device)
                loss = criterion(logits, label.float())

                print(
                    f"---{num_correct} {is_True} epoch:{epoch}  image_name:{img_name}"
                )
                print(
                    f"model result -> logits:{logits}  after sigmoid -> prob_out:{prob_out} label:{label.float()}"
                )
                per_epoch_loss += loss.item()
                print(
                    f"---loss.item():{loss.item()}------",
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            score_array = np.array(score_list)
            label_array = np.array(label_list)
            fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(
                label_array, score_array
            )
            auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

            print(
                "Train EVpoch: {}\t Loss: {:.6f}\t Acc: {:.6f} AUC: {:.6f} ".format(
                    epoch,
                    per_epoch_loss / len(data_loader_train),
                    num_correct / len(data_loader_train.dataset),
                    auc_keras_1,
                )
            )
            summaryWriter.add_scalars(
                "loss", {"loss": (per_epoch_loss / len(data_loader_train))}, epoch
            )
            summaryWriter.add_scalars(
                "acc", {"acc": num_correct / len(data_loader_train.dataset)}, epoch
            )
            summaryWriter.add_scalars("auc", {"auc": auc_keras_1}, epoch)

        model.eval()
        eva_loss = 0
        with torch.no_grad():
            for x, label, img_name in tqdm(data_loader_val):
                x = x.float()
                x = x.to(device)
                is_True = False
                val_label_list.extend(label.numpy())
                label = label.to(device)
                logits = model(x)
                logits = logits.reshape([label.cpu().numpy().shape[0]])
                logits = logits.to(device)
                eva_loss_item = criterion(logits, label.float())
                eva_loss += eva_loss_item
                prob_out = nn.Sigmoid()(logits)
                pro_list = prob_out.detach().cpu().numpy()
                for i in range(pro_list.shape[0]):
                    if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
                        is_True = True
                        val_num_correct += 1
                val_score_list.extend(pro_list)
                print(
                    f"---【val-{val_num_correct}-{is_True}】  epoch:{epoch}  image_name:{img_name}"
                )
                print(f"logits:{logits}  prob_out:{prob_out}  label:{label} ------")

            score_array = np.array(val_score_list)
            label_array = np.array(val_label_list)
            fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(
                label_array, score_array
            )
            auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

            print(
                f"val Epoch: epocht Loss:{eva_loss} Acc: {val_num_correct / len(data_loader_val.dataset)} AUC: {auc_keras_1} "
            )
            summaryWriter.add_scalars(
                "acc",
                {"val_acc": val_num_correct / len(data_loader_val.dataset)},
                epoch,
            )
            summaryWriter.add_scalars(
                "loss", {"val_loss": (eva_loss / len(data_loader_val))}, epoch
            )
            summaryWriter.add_scalars("auc", {"val_auc": auc_keras_1}, epoch)
            summaryWriter.add_scalars(
                "time", {"val_time": (time.time() - start)}, epoch
            )

        scheduler.step()

        wtes(eva_loss, model)
        if wtes.early_stop:
            print("Early stopping")
            break

        if (epoch + 1) % save_interval == 0:
            path = (
                save_folder
                # "/home/wangwenmiao/wtli-workspace/project-thymoma/wtlizzz-core/wtliModel/wtWeight/model"
                + str(epoch + 1)
                + ".pth"
            )
            torch.save(model.state_dict(), path)
