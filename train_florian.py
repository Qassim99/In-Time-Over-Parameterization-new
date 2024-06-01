# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

#own
from torch.utils.data import Dataset
from PIL import Image
import wandb
import random
import os
import pandas as pd
import numpy as np

# from peft import get_peft_model, LoraConfig, get_peft_model


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

"""
def main(argv):
    #argv = ['-m', 'cheng2020-anchor', '-d', 'tmp/Kodak', '--batch-size', '16', '-lr', '1e-4', '--save', '--cuda']

    args = parse_args(argv)
    #print(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=3)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
"""


def configure_optimizers(net, learning_rate, aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": learning_rate},
        "aux": {"type": "Adam", "lr": aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]







def train_model(model='cheng2020-anchor', dataset='./', dataset_name=None, train_dataloader=None, val_dataloader=None, matplot=True, batch_size=16, 
seed=None, cuda=True, save=True, patch_size=(256, 256), learning_rate=1e-4, aux_learning_rate=1e-3, lmbda=1e-2, epochs=100, 
num_workers=4, test_batch_size=64, clip_max_norm=1.0, checkpoint=None):
    #argv = ['-m', 'cheng2020-anchor', '-d', 'tmp/Kodak', '--batch-size', '16', '-lr', '1e-4', '--save', '--cuda']

    # start a new wandb run to track this script
    project_name = 'training ' + model + ' on ' + dataset_name
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": model,
        "dataset": dataset_name,
        "epochs": epochs,
        }
    )
    
    #print(argv)
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)



    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"


    net = image_models[model](quality=3)
    net = net.to(device)

    if cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    """
    if sys.argv[2] == "lora":
        peft_config = LoraConfig(target_modules='g_a' ,inference_mode=False, r=4, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(net, peft_config)
        model.print_trainable_parameters()
    """
    optimizer, aux_optimizer = configure_optimizers(net, learning_rate, aux_learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda)

    last_epoch = 0
    if checkpoint:  # load from previous checkpoint
        print("Loading", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    epochs_own = []
    losses_own = []
    best_loss = float("inf")
    for epoch in range(last_epoch, epochs):
        #print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            clip_max_norm,
        )
        loss = test_epoch(epoch, val_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )
        epochs_own.append(epoch)
        losses_own.append(loss.item())

        wandb.log({"epoch": epoch, "loss": loss})

    wandb.finish()

    print("epoch, loss")
    for i in range(len(epochs_own)):
        print("(" + str(epochs_own[i]) + "," + str(losses_own[i]) + ")", end="")
    print("")
    print("epochs: " + str(epochs_own))
    print("losses: " + str(losses_own))

    #save training data in txt file 
    #fileneme = model + " trained on " + dataset_name + ".txt"
    #file = open(fileneme,"w+")
    #file.write("epochs: " + str(epochs_own) + "\n")
    #file.write("losses: " + str(losses_own))
    #file.close()



    #save training data in csv file
    savefile = model + "_trained_on_" + dataset_name + '.csv'
    df = pd.DataFrame({'epochs': epochs_own,
                        'losses': losses_own})
    df.to_csv(savefile, index=False) 
    




class ImageDataset(Dataset):

    def __init__(self, datalist, image_dir, transformer, set_):

        self.datalist = datalist
        self.image_dir = image_dir
        self.transformer = transformer
        self.images_per_sequence = 7
        self.set = set_

        file = open(self.datalist)
        self.datalist_lines=file.readlines()

        #length of datalist_lines for train dataset is 64612
        if self.set == "train":
            self.datalist_lines = self.datalist_lines[:58152]
            self.images_per_sequence = 7
        if self.set == "train7":
            self.datalist_lines = self.datalist_lines[:58152]
            self.images_per_sequence = 1
        elif self.set == "train28":
            self.datalist_lines = self.datalist_lines[:14539]
            self.images_per_sequence = 1
        elif self.set == "val":
            self.datalist_lines = self.datalist_lines[58151:]
            self.images_per_sequence = 7
        elif self.set == "val7":
            self.datalist_lines = self.datalist_lines[58151:]
            self.images_per_sequence = 1
        elif self.set == "test":
            self.datalist_lines = self.datalist_lines
            self.images_per_sequence = 7
        elif self.set == "test7":
            self.datalist_lines = self.datalist_lines
            self.images_per_sequence = 1
        elif self.set == "all":
            self.datalist_lines = self.datalist_lines
            self.images_per_sequence = 7


    def __len__(self):
        return len(self.datalist_lines) * self.images_per_sequence

    def __getitem__(self, index):
        line = index // self.images_per_sequence                #for full sequence
        sequnce_pic = (index % self.images_per_sequence) + 1    #for full sequence

        path = self.image_dir + '/' + self.datalist_lines[line].strip() + '/im' + str(sequnce_pic) + '.png' #for full sequence
        #path = self.image_dir + '/' + self.datalist_lines[line].strip() + '/im1.png'                        #for just the first pic of the sequence
        image = Image.open(path).convert('RGB')

        return self.transformer(image)


def get_dataloader(path=None, dataset=None, patch_size=(256, 256), transformer=None, batch_size=16):
    
    #DIV2K for example
    if dataset is None:
        dataset = ImageFolder(path, transform=transformer)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=(device == "cuda")
    )
    return dataloader

patch_size=(256, 256)
train_transforms = transforms.Compose(
    [transforms.RandomCrop(patch_size), transforms.ToTensor()]
)
test_transforms = transforms.Compose(
    [transforms.CenterCrop(patch_size), transforms.ToTensor()]
)

#train_dataloader = get_dataloader(path='/data/videocoding/dnnvc/datasets/DIV2K/tmp/DIV2K_train_HR', patch_size=patch_size, transformer=train_transforms)
#test_dataloader = get_dataloader(path='/data/videocoding/dnnvc/datasets/DIV2K/tmp/DIV2K_valid_HR', patch_size=patch_size, transformer=test_transforms)


train_list_path = str(sys.argv[1]) + 'sep_trainlist.txt'   #/data/videocoding/dnnvc/datasets/Vimeo-90k/tmp/vimeo_septuplet/sep_trainlist.txt
test_list_path = str(sys.argv[1]) + 'sep_testlist.txt'     #/data/videocoding/dnnvc/datasets/Vimeo-90k/tmp/vimeo_septuplet/sep_testlist.txt
sequence_folder = str(sys.argv[1]) + 'sequences'           #/data/videocoding/dnnvc/datasets/Vimeo-90k/tmp/vimeo_septuplet/sequences


#train_dataset = ImageDataset(train_list_path, sequence_folder, train_transforms, "train")
train_dataset = ImageDataset(train_list_path, sequence_folder, train_transforms, "train7")
#train_dataset = ImageDataset(train_list_path, sequence_folder, train_transforms, "train28")

#val_dataset = ImageDataset(train_list_path, sequence_folder, test_transforms, "val")
val_dataset = ImageDataset(train_list_path, sequence_folder, test_transforms, "val7")
"""
train_a = ImageDataset(train_list_path, sequence_folder, train_transforms, "all")
train_part = ImageDataset(train_list_path, sequence_folder, train_transforms, "train")
val_ = ImageDataset(train_list_path, sequence_folder, train_transforms, "val")
test_ = ImageDataset(test_list_path, sequence_folder, train_transforms, "test")

print(len(train_a))
print(len(train_part))
print(len(val_))
print(len(test_))
"""
train_dataloader = get_dataloader(dataset=train_dataset, patch_size=patch_size, transformer=None)
val_dataloader = get_dataloader(dataset=val_dataset, patch_size=patch_size, transformer=None)

#python compressai/examples/train.py /data/videocoding/dnnvc/datasets/Vimeo-90k/tmp/vimeo_septuplet/
train_model(model='cheng2020-anchor', save=True, seed=1234, epochs=70, train_dataloader=train_dataloader, val_dataloader=val_dataloader, dataset_name='Vimeo-90k', num_workers=4)
#train_model(model='cheng2020-anchor', save=True, seed=1234, epochs=10, split=100, dataset='/data/videocoding/dnnvc/datasets/DIV2K/tmp/DIV2K_train_HR')