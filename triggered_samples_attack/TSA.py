import warnings
warnings.filterwarnings("ignore")

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from triggered_samples_attack.utils import *
from models.quantization import *
from models import quan_resnet
import numpy as np
import config
from models.model_wrap import Attacked_model
import copy
import argparse

parser = argparse.ArgumentParser(description='Triggered Samples Attack')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--trigger-size', dest='trigger_size', type=int, default=10)
parser.add_argument('--target', dest='target', type=int, default=0)

parser.add_argument('--gpu-id', dest='gpu_id', type=str, default='0')

parser.add_argument('--lam1', dest='lam1', default=100, type=float)
parser.add_argument('--lam2', dest='lam2', default=1, type=float)
parser.add_argument('--k-bits', '-k_bits', default=[5, 10, 20, 40], nargs='+', type=int)
parser.add_argument('--n-aux', '-n_aux', default=128, type=int)

parser.add_argument('--max-search', '-max_search', default=8, type=int)
parser.add_argument('--ext-max-iters', '-ext_max_iters', default=3000, type=int)
parser.add_argument('--inn-max-iters', '-inn_max_iters', default=5, type=int)
parser.add_argument('--initial-rho1', '-initial_rho1', default=0.0001, type=float)
parser.add_argument('--initial-rho2', '-initial_rho2', default=0.0001, type=float)
parser.add_argument('--initial-rho3', '-initial_rho3', default=0.00001, type=float)
parser.add_argument('--max-rho1', '-max_rho1', default=100, type=float)
parser.add_argument('--max-rho2', '-max_rho2', default=100, type=float)
parser.add_argument('--max-rho3', '-max_rho3', default=10, type=float)
parser.add_argument('--rho-fact', '-rho_fact', default=1.01, type=float)
parser.add_argument('--inn-lr-bit', '-inn_lr_bit', default=0.001, type=float)
parser.add_argument('--inn-lr-trigger', '-inn_lr_trigger', default=1, type=float)
parser.add_argument('--stop-threshold', '-stop_threshold', default=1e-4, type=float)
parser.add_argument('--projection-lp', '-projection_lp', default=2, type=int)

parser.add_argument("--silent", action='store_true', help='execute attack silently')

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

print("Prepare data ... ")
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

test_dir = config.cifar_root
val_set = datasets.CIFAR10(root=test_dir, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
]))

val_loader = torch.utils.data.DataLoader(
    dataset=val_set,
    batch_size=args.batch_size, shuffle=False, pin_memory=True)


class_num = 10
input_size = 32


bit_length = 8
model = torch.nn.DataParallel(quan_resnet.resnet20_quan_mid(class_num, bit_length))


checkpoint = torch.load(os.path.join(config.model_root, "model.th"))
model.load_state_dict(checkpoint["state_dict"])

if isinstance(model, torch.nn.DataParallel):
    model = model.module

for m in model.modules():
    if isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()
model.cuda()

load_model = Attacked_model(model, "cifar10", "resnet20_quan_8")
load_model.cuda()
load_model.eval()

model = torch.nn.DataParallel(model)
load_model.model = torch.nn.DataParallel(load_model.model)

criterion = nn.CrossEntropyLoss().cuda()


n_aux = args.n_aux  # the size of auxiliary sample set
lam1 = args.lam1
lam2 = args.lam2
ext_max_iters = args.ext_max_iters
inn_max_iters = args.inn_max_iters
initial_rho1 = args.initial_rho1
initial_rho2 = args.initial_rho2
initial_rho3 = args.initial_rho3
max_rho1 = args.max_rho1
max_rho2 = args.max_rho2
max_rho3 = args.max_rho3
rho_fact = args.rho_fact
inn_lr_bit = args.inn_lr_bit
inn_lr_trigger = args.inn_lr_trigger
stop_threshold = args.stop_threshold

projection_lp = args.projection_lp

target_class = args.target

np.random.seed(512)
aux_idx = np.random.choice(len(val_loader.dataset), args.n_aux, replace=False)

normalize = Normalize(mean=[0.4914, 0.4822, 0.4465],
                      std=[0.2023, 0.1994, 0.2010])
transform = transforms.Compose([
    transforms.RandomRotation(degrees=(10, 10)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    # normalize,
])
aux_dataset = ImageFolder_cifar10(val_loader.dataset.data[aux_idx],
                                  np.array(val_loader.dataset.targets)[aux_idx],
                                  transform=transform)

aux_loader = torch.utils.data.DataLoader(
    dataset=aux_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True)


def pnorm(x, p=2):
    batch_size = x.size(0)
    norm = x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)
    norm = torch.max(norm, torch.ones_like(norm) * 1e-6)
    return norm


def loss_func(output, labels, output_trigger, labels_trigger, lam1, lam2, w,
              b_ori, k_bits, y1, y2, y3, z1, z2, z3, k, rho1, rho2, rho3):

    l1 = F.cross_entropy(output_trigger, labels_trigger)
    l2 = F.cross_entropy(output, labels)

    y1, y2, y3, z1, z2, z3 = torch.tensor(y1).float().cuda(), torch.tensor(y2).float().cuda(), torch.tensor(y3).float().cuda(), \
                             torch.tensor(z1).float().cuda(), torch.tensor(z2).float().cuda(), torch.tensor(z3).float().cuda()

    b_ori = torch.tensor(b_ori).float().cuda()
    b = w.view(-1)

    l3 = z1@(b-y1) + z2@(b-y2) + z3*(torch.norm(b - b_ori) ** 2 - k + y3)

    l4 = (rho1/2) * torch.norm(b - y1) ** 2 + (rho2/2) * torch.norm(b - y2) ** 2 \
       + (rho3/2) * (torch.norm(b - b_ori)**2 - k_bits + y3) ** 2

    return lam1 * l1 + lam2 * l2 + l3 + l4, l1.item(), l2.item()


def attack_func(k_bits, lam1, lam2):

    attacked_model = copy.deepcopy(load_model)
    attacked_model_ori = copy.deepcopy(load_model)
    validate(val_loader, nn.Sequential(normalize, attacked_model), criterion)

    b_ori = attacked_model.w_twos.data.view(-1).detach().cpu().numpy()
    b_new = b_ori

    y1 = b_ori
    y2 = y1
    y3 = 0

    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y1)
    z3 = 0

    rho1 = initial_rho1
    rho2 = initial_rho2
    rho3 = initial_rho3

    stop_flag = False

    trigger = torch.randn([1, 3, input_size, input_size]).float().cuda()
    trigger_mask = torch.zeros([1, 3, input_size, input_size]).cuda()
    trigger_mask[:, :, input_size-args.trigger_size:input_size, input_size-args.trigger_size:input_size] = 1

    for ext_iter in range(ext_max_iters):

        y1 = project_box(b_new + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_new + z2 / rho2, projection_lp)
        y3 = project_positive(-np.linalg.norm(b_new - b_ori, ord=2) ** 2 + k_bits - z3 / rho3)

        for inn_iter in range(inn_max_iters):

            for i, (input, target) in enumerate(aux_loader):
                input_var = torch.autograd.Variable(input, volatile=True).cuda()
                target_var = torch.autograd.Variable(target, volatile=True).cuda()
                target_trigger_var = torch.zeros_like(target_var) + target_class
                trigger = torch.autograd.Variable(trigger, requires_grad=True)

                output = attacked_model(normalize(input_var))
                output_trigger = attacked_model(normalize(input_var * (1 - trigger_mask) + trigger * trigger_mask))
                reg_mask = torch.ones(input_var.shape[0]).cuda()
                reg_mask[torch.where(target_var==target_class)] = 0

                loss, loss1, loss2 = loss_func(output, target_var, output_trigger, target_trigger_var,
                                 lam1, lam2, attacked_model.w_twos,
                                 b_ori, k_bits, y1, y2, y3, z1, z2, z3, k_bits, rho1, rho2, rho3)

                loss.backward(retain_graph=True)

                attacked_model.w_twos.data = attacked_model.w_twos.data - \
                                                           inn_lr_bit * attacked_model.w_twos.grad.data
                if ext_iter < 1000:
                    trigger.data = trigger.data - inn_lr_trigger * trigger.grad.data
                else:
                    trigger.data = trigger.data - inn_lr_trigger * 0.1 * trigger.grad.data

                for name, param in attacked_model.named_parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
                trigger.grad.zero_()

                trigger = torch.clamp(trigger, min=0.0, max=1.0)

        b_new = attacked_model.w_twos.data.view(-1).detach().cpu().numpy()

        z1 = z1 + rho1 * (b_new - y1)
        z2 = z2 + rho2 * (b_new - y2)
        z3 = z3 + rho3 * (np.linalg.norm(b_new - b_ori, ord=2) ** 2 - k_bits + y3)

        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (np.linalg.norm(b_new - y1)) / max(np.linalg.norm(b_new), 2.2204e-16)
        temp2 = (np.linalg.norm(b_new - y2)) / max(np.linalg.norm(b_new), 2.2204e-16)
        if ext_iter % 50 == 0 and not args.silent:
            print('iter: %d, stop_threshold: %.6f loss: %.4f' % (
                ext_iter, max(temp1, temp2), loss.item()))

        if max(temp1, temp2) <= stop_threshold and ext_iter > 100:
            stop_flag = True
            break

    attacked_model.w_twos.data[attacked_model.w_twos.data > 0.5] = 1.0
    attacked_model.w_twos.data[attacked_model.w_twos.data < 0.5] = 0.0

    n_bit = torch.norm(attacked_model.w_twos.data.view(-1) - attacked_model_ori.w_twos.data.view(-1), p=0).item()

    clean_acc, _, _ = validate(val_loader, nn.Sequential(normalize, attacked_model), criterion)
    trigger_acc, _, _ = validate_trigger(val_loader, trigger, trigger_mask, target_class, nn.Sequential(normalize, attacked_model), criterion)

    aux_clean_acc, _, _ = validate(aux_loader, nn.Sequential(normalize, attacked_model), criterion)
    aux_trigger_acc, _, _ = validate_trigger(aux_loader, trigger, trigger_mask, target_class, nn.Sequential(normalize, attacked_model), criterion)

    return clean_acc, trigger_acc, n_bit, aux_trigger_acc


def main():
    all_k_bits = args.k_bits if isinstance(args.k_bits, list) else [args.k_bits]

    for k_bits in all_k_bits:
        print("Attack Start, k =", k_bits)
        clean_acc, trigger_acc, n_bit, aux_trigger_acc = attack_func(k_bits, lam1, lam2)
        if aux_trigger_acc > 98:
            print("target:{0} clean_acc:{1:.4f} asr:{2:.4f} bit_flips:{3}".format(
                   args.target, clean_acc, trigger_acc, n_bit, aux_trigger_acc))
            break


if __name__ == '__main__':
    main()