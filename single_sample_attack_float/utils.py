import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import time

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import vanilla_resnet
from models.quantization import *
import numpy as np
import config


def load_model(arch):
    model_path = config.float_model_root

    model = torch.nn.DataParallel(vanilla_resnet.__dict__[arch](10))

    model.cuda()

    model.load_state_dict(torch.load(os.path.join(model_path, "model.th"))["state_dict"])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for m in model.modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.detach().cpu().numpy()
            bias = m.bias.data.detach().cpu().numpy()
    return weight, bias


def load_data(arch):
    mid_dim = 64
    model_path = config.float_model_root
    arch = arch + "_mid"

    model = torch.nn.DataParallel(vanilla_resnet.__dict__[arch](10))

    model.cuda()

    model.load_state_dict(torch.load(os.path.join(model_path, "model.th"))["state_dict"])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    val_set = datasets.CIFAR10(root=config.cifar_root, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=256, shuffle=False, pin_memory=True)

    mid_out = np.zeros([10000, mid_dim])
    labels = np.zeros([10000])
    start = 0
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        # compute output before FC layer.
        output = model(input_var)
        mid_out[start: start + 256] = output.detach().cpu().numpy()

        labels[start: start + 256] = target.numpy()
        start += 256

    mid_out = torch.tensor(mid_out).float().cuda()
    labels = torch.tensor(labels).float()

    return mid_out, labels

load_model("resnet20")
mid_out, labels = load_data("resnet20")
