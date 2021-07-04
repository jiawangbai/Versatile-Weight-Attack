import torch
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torch
import torch.utils
from torchvision.datasets.folder import default_loader
import torch.nn as nn
import torch.distributed as dist


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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



def validate(val_loader, model, criterion, local_rank=None, nprocs=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            if local_rank==None:
                target = target.cuda(async=True)
                input = input.cuda()
            else:
                target = target.cuda(local_rank, async=True)
                input = input.cuda(local_rank)

            # compute output
            output = model(input)
            loss = criterion(output, target)


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))


            if nprocs!=None:
                reduced_loss = reduce_mean(loss, nprocs)
                reduced_prec1 = reduce_mean(prec1, nprocs)
                reduced_prec5 = reduce_mean(prec5, nprocs)
                losses.update(reduced_loss.item(), input.size(0))
                top1.update(reduced_prec1.item(), input.size(0))
                top5.update(reduced_prec5.item(), input.size(0))
            else:
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

        # if local_rank == None or local_rank == 0:
        #     print(
        #         '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        #         .format(top1=top1, top5=top5, error1=100 - top1.avg))

    return top1.avg, top5.avg, losses.avg


def validate_trigger(val_loader, trigger, trigger_mask, target_class, model, criterion, local_rank=None, nprocs=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if local_rank==None:
                target = target.cuda(async=True) * 0 + target_class
                input = input.cuda()
            else:
                target = target.cuda(local_rank, async=True) * 0 + target_class
                input = input.cuda(local_rank)

            # compute output
            output = model(input * (1 - trigger_mask) + trigger * trigger_mask)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if nprocs!=None:
                reduced_loss = reduce_mean(loss, nprocs)
                reduced_prec1 = reduce_mean(prec1, nprocs)
                reduced_prec5 = reduce_mean(prec5, nprocs)
                losses.update(reduced_loss.item(), input.size(0))
                top1.update(reduced_prec1.item(), input.size(0))
                top5.update(reduced_prec5.item(), input.size(0))
            else:
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

        # if local_rank == None or local_rank == 0:
        #     print(
        #         '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        #         .format(top1=top1, top5=top5, error1=100 - top1.avg))

    return top1.avg, top5.avg, losses.avg


class ImageFolder_cifar10(Dataset):

    def __init__(self, samples, targets, transform=None):

        self.samples = samples
        self.transform = transform
        self.target = targets

    def __getitem__(self, index):

        sample = self.samples[index]
        if self.transform is not None:
            sample = self.transform(Image.fromarray(sample))
        return sample, self.target[index]

    def __len__(self):
        return len(self.samples)


class ImageFolder_imagenet(Dataset):

    def __init__(self, paths, targets, transform=None):

        self.paths = paths
        self.transform = transform
        self.target = targets

    def __getitem__(self, index):

        sample = default_loader(self.paths[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.target[index]

    def __len__(self):
        return len(self.paths)


def project_box(x):
    xp = x
    xp[x>1]=1
    xp[x<0]=0

    return xp


def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones(x.size)
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec

    return xp


def project_positive(x):
    xp = np.clip(x, 0, None)
    return xp


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]
