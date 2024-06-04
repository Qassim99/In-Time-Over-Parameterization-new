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
import copy
import os
import random
import shutil
import sys
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.datasets.stack import StackDataset
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models


#sparselearning packages
import torch.backends.cudnn as cudnn

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
import sparselearning.core
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True


#from florain
from torch.utils.data import Dataset
from PIL import Image
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


def reinitialize_optimizer(model, args):
    optimizer, aux_optimizer = configure_optimizers(model, args)
    return optimizer, aux_optimizer




def train_one_epoch(
    device, model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, mask=None
):
    model.train()

    for i, d in enumerate(train_dataloader):
        inputs = d
        inputs = inputs.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        out_net = model(inputs)
      
        
        out_criterion = criterion(out_net, inputs)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        
        if mask is not None:
            mask.step()
        else:
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





def test_epoch(epoch,device, test_dataloader, model, criterion):
    model.eval()
    #device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            inputs = d
            inputs = inputs.to(device)
            out_net = model(inputs)
            out_criterion = criterion(out_net, inputs)

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

    return loss.avg, mse_loss.avg, bpp_loss.avg, aux_loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


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
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    sparselearning.core.add_sparse_args(parser)
    args = parser.parse_args(argv)
    return args



def main(argv):
    args = parse_args(argv)

    project_name = 'training ' + args.model + ' on ' + 'DIV2K'
    if args.sparse:
        project_name += ' with density ' + str(args.density) + 'prune critiria ' + args.death 
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "architecture": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        }
    )

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

    net = image_models[args.model](pretrained=False, quality=3)
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

    mask = None
    if args.sparse:
        decay = CosineDecay(args.death_rate, len(train_dataloader)*(args.epochs*args.num_workers))
        mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                        redistribution_mode=args.redistribution, args=args)
        mask.add_module(net, sparse_init=args.sparse_init, density=args.density)

    # Reinitialize the optimizer after pruning
    optimizer, aux_optimizer = reinitialize_optimizer(net, args)


    epochs_own = []
    losses_own = []
    mse_losses_own = []
    bpp_losses_own = []
    aux_losses_own = []
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            device,
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            mask,
        )
        loss, mse, bpp, aux = test_epoch(epoch, device, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best:
            model_dir = "./models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "model.pth")
        torch.save(net.state_dict(), model_path)

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
        epochs_own.append(epoch)
        losses_own.append(loss.item())
        mse_losses_own.append(mse.item())
        bpp_losses_own.append(bpp.item())
        aux_losses_own.append(aux.item())

        wandb.log({"loss": loss, "mes": mse, "bpp": bpp, "aux": aux})

    wandb.finish()

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    savefile_dir = os.path.join(".", args.model + "_graphic")
    if not os.path.exists(savefile_dir):
        os.makedirs(savefile_dir)
    if args.sparse:
        suffix= str(args.density) + '_' + args.death
    else: 
        suffix = '1.0_none'
    filename =  args.model + "_" + suffix + "_DIV2K_"+ hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8]
    savefile = os.path.join(savefile_dir, filename + '.csv')
    df = pd.DataFrame({'epochs': epochs_own, 'losses': losses_own})
    df.to_csv(savefile, index=False)

    

# Simulated data for visualization purposes
    epochs_range = list(range(1, len(epochs_own) + 1))

    with PdfPages(filename + '_all_losses.pdf') as pdf:
        # Plot Loss
        plt.figure(figsize=(12, 8))
        plt.plot(epochs_range, losses_own, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.legend()
        plt.grid(True)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
        # Plot MSE Loss
        plt.figure(figsize=(12, 8))
        plt.plot(epochs_range, mse_losses_own, label='MSE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.title('Training and Testing MSE Loss')
        plt.legend()
        plt.grid(True)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
        # Plot BPP Loss
        plt.figure(figsize=(12, 8))
        plt.plot(epochs_range, bpp_losses_own, label='BPP Loss')
        plt.xlabel('Epochs')
        plt.ylabel('BPP Loss')
        plt.title('Training and Testing BPP Loss')
        plt.legend()
        plt.grid(True)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
        # Plot Aux Loss
        plt.figure(figsize=(12, 8))
        plt.plot(epochs_range, aux_losses_own, label='Aux Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Aux Loss')
        plt.title('Training and Testing Aux Loss')
        plt.legend()
        plt.grid(True)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
