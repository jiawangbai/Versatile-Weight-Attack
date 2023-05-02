import torch.nn as nn
import torch
import torch.nn.functional as F
from single_sample_attack_float.convert import float2bit, bit2float_device

class Attacked_model(nn.Module):
    def __init__(self, model, dataset, arch):
        super(Attacked_model, self).__init__()

        self.model = model

        if dataset == "cifar10":
            if arch[:len("resnet20")] == "resnet20":
                self.w = model.linear.weight.data.detach()
                self.b = nn.Parameter(model.linear.bias.data, requires_grad=True)
            elif arch[:len("vgg16_bn")] == "vgg16_bn":
                self.w = model.classifier[6].weight.data.detach()
                self.b = nn.Parameter(model.classifier[6].bias.data, requires_grad=True)
        elif dataset == "imagenet":
            if arch[:len("resnet18")] == "resnet18":
                self.w = model.fc.weight.data.detach()
                self.b = nn.Parameter(model.fc.bias.data, requires_grad=True)
            elif arch[:len("vgg16_bn")] == "vgg16_bn":
                self.w = model.classifier[6].weight.data.detach()
                self.b = nn.Parameter(model.classifier[6].bias.data, requires_grad=True)

        self.w_twos = nn.Parameter(torch.zeros([self.w.shape[0], self.w.shape[1], 32]), requires_grad=True) #.to(self.w.device)

        self.reset_w_twos()

    def forward(self, x):

        x = self.model(x)
        # covert w_twos to float
        w = bit2float_device(self.w_twos)

        # calculate output
        x = F.linear(x, w, self.b)

        return x

    def reset_w_twos(self):
        self.w_twos.data += float2bit(self.w).data

