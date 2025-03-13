'''Implementation of Loo et al.’s method

Reference:
[1] Noel Loo, Ramin Hasani, Mathias Lechner, Alexander Amini, and Daniela Rus. 
    Understanding reconstruction attacks with the neural tangent kernel and dataset distillation. 
    In The Twelfth International Conference on Learning Representations, 2024.
'''
import os
import torch
import random
import logging
import argparse
import collections
import numpy as np
from models import *
from tqdm import tqdm
import torch.nn as nn
from utils import ReparamModule
import torch.nn.functional as F
from torchvision import models


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tlr",
        default=1e-2,
        type=float,
        help="learning rate used for training the model (train.py)",
    )
    parser.add_argument("--model_name", default="resnet", type=str)
    parser.add_argument("--num_imgs_per_class", default=2, type=int)
    parser.add_argument("--width", default=1000, type=int)
    parser.add_argument("--bs", default=20, type=int)
    parser.add_argument(
        "--loss_type",
        default=0,
        type=int,
        help="0 means cosine similarity loss, and 1 means l2 loss",
    )
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--rec_lr",
        default=3e-4,
        type=float,
        help="Learning rate used for updating the dummy dataset images",
    )
    parser.add_argument(
        "--rec_lambda_lr",
        default=1e-6,
        type=float,
        help="Learning rate used for updating the dual parameters",
    )
    args = parser.parse_args()
    return args


def set_logger(log_name, log_dir, args):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class*num_classes}_bsz{args.bs}_tlr{args.tlr}_2step.log",
        )
    )
    file_handler.setLevel(logging.DEBUG)

    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # console_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


args = parse_args()

log_interval = 500
save_interval = 500

c, h, w = 3, 32, 32
extraction_init_scale = 5e-1
num_classes = 10
num_imgs_per_class = args.num_imgs_per_class
width = args.width
batch_size = args.bs
model_name = args.model_name
lr = args.tlr
rec_lr = args.rec_lr
rec_lambda_lr = args.rec_lambda_lr
rec_ori_ratio = 2
num_imgs_to_reconstruct = num_classes * num_imgs_per_class * rec_ori_ratio
num_rec_steps = 100000

logger = set_logger("reconstruction_logger", "logs/recons", args)

recover_x_dir = "./ckpts/recover_x/"
os.makedirs(recover_x_dir, exist_ok=True)

save_path = os.path.join(
    recover_x_dir,
    f"{model_name}_{args.dataset}_num_img{num_imgs_per_class * num_classes}"
    + f"_bsz{batch_size}_tlr{lr}_1step_reclr{rec_lr}_lambdalr{rec_lambda_lr}/",
)

if not os.path.exists(save_path):
    os.makedirs(save_path)


x = torch.randn(num_imgs_to_reconstruct, c, h, w).to(device) * extraction_init_scale
x.requires_grad_(True)
l_i = torch.rand(num_imgs_to_reconstruct, 1) - 0.5
l_i = l_i.to(device)
l_i.requires_grad_(True)
l_f = torch.rand(num_imgs_to_reconstruct, 1) - 0.5
l_f = l_f.to(device)
l_f.requires_grad_(True)

random_label = np.zeros(num_imgs_to_reconstruct)
for c in range(num_classes):
    random_label[
        int(c * num_imgs_to_reconstruct / num_classes) : int(
            (c + 1) * num_imgs_to_reconstruct / num_classes
        )
    ] = c
random.shuffle(random_label)
random_label = torch.tensor(random_label).to(device)

y = random_label.long()
torch.save(y, save_path + "label.pth")


def total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def get_params_vec(model):
    vec = []
    for param in model.parameters():
        if param.requires_grad:
            vec.append(param.reshape(-1, 1))
    vec = torch.cat(vec, 0)
    vec = vec.squeeze()
    return vec


def calc_steps_loss_sgd(
    x,
    l_i,
    l_f,
    model_i,
    model_f,
    y,
    weight_change,
    step,
):
    value_i = model_i(x)
    loss_i = criterion(value_i, y)
    loss_i_mean = torch.matmul(loss_i, l_i)
    grad_i = torch.autograd.grad(loss_i_mean, filter(lambda p: p.requires_grad, model_i.parameters()), create_graph=True, retain_graph=True)
    grad_vec_i = []
    for g in grad_i:
        grad_vec_i.append(g.reshape(-1,1))
    grad_vec_i = torch.cat(grad_vec_i, 0)

    value_f = model_f(x)
    loss_f = criterion(value_f, y)
    loss_f_mean = torch.matmul(loss_f, l_f)
    grad_f = torch.autograd.grad(loss_f_mean, filter(lambda p: p.requires_grad, model_f.parameters()), create_graph=True, retain_graph=True)
    grad_vec_f = []
    for g in grad_f:
        grad_vec_f.append(g.reshape(-1,1))
    grad_vec_f = torch.cat(grad_vec_f, 0)

    grad_vec = grad_vec_i + grad_vec_f

    l2_loss = (grad_vec.flatten() - weight_change.flatten()).pow(2).sum()
    sim = torch.nn.functional.cosine_similarity(
        grad_vec.flatten(), weight_change.flatten(), 0, 1e-20
    )

    cos_loss = 1 - sim
    tv_loss = 0.01 * total_variation(x)

    if step < 600:
        loss = l2_loss / 100
    elif step < 1600:
        loss = l2_loss / 30
    elif step < 4000:
        loss = l2_loss / 20
    elif step < 8000:
        loss = l2_loss
    elif step < 12000:
        loss = l2_loss * 5
    else:
        loss = cos_loss * 30 + 0.1 * tv_loss
    # l2 loss has same influence with cosines similarity loss
    # making sure that loss converges is ok
    
    
    return loss, l2_loss, cos_loss, tv_loss


def build_models():
    if args.model_name == "resnet":
        model_init = ResNet18()
        num_ftrs = model_init.linear.in_features
        model_init.linear = nn.Linear(num_ftrs, num_classes)
        model_diff = ResNet18()
        model_diff.linear = nn.Linear(num_ftrs, num_classes)
        model_final = ResNet18()
        model_final.linear = nn.Linear(num_ftrs, num_classes)
    elif args.model_name == "mlp":
        model_init = mlp(3072, [width, width], num_classes)
        model_diff = mlp(3072, [width, width], num_classes)
        model_final = mlp(3072, [width, width], num_classes)
    elif args.model_name == "vgg16":
        model_init = models.vgg16(pretrained=False)
        model_init.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
        model_diff = models.vgg16(pretrained=False)
        model_diff.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
        model_final = models.vgg16(pretrained=False)
        model_final.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    else:
        raise NotImplementedError(f"Model {model_name} not support yet")

    model_ckpt_dir = os.path.join(
        "./ckpts/model_ckpt/",
        f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}"
        + f"_bsz{batch_size}_lr{lr}",
    )

    init_param_path = model_ckpt_dir + "/ori.pth"
    final_param_path = model_ckpt_dir + "/final.pth"

    weight_init = torch.load(init_param_path)
    weight_final = torch.load(final_param_path)

    weight_keys = list(weight_init.keys())
    diff_state_dict = collections.OrderedDict()
    for key in weight_keys:
        diff_state_dict[key] = weight_init[key] - weight_final[key]

    model_init.load_state_dict(weight_init)
    model_init.to(device)
    model_init.eval()
    model_final.load_state_dict(weight_final)
    model_final.to(device)
    model_final.eval()
    model_diff.load_state_dict(diff_state_dict)
    model_diff.to(device)
    model_diff.eval()

    return model_init, model_final, model_diff


def train_dummy_dataset():
    for step in tqdm(range(num_rec_steps)):
        loss, l2_loss, cos_loss, tv_loss = calc_steps_loss_sgd(
            x,
            l_i,
            l_f,
            model_init,
            model_final,
            y,
            weight_change,
            step,
            )

        if step % log_interval == 0:
            logger.info(f"epoch {step}")
            logger.info(f"total_loss: {loss.item()}")
            logger.info(f"l2_loss: {l2_loss.item()}")
            logger.info(f"cos_loss: {cos_loss.item()}")
            logger.info(f"tv_loss: {tv_loss.item()}")

        opt_x.zero_grad()
        opt_lf.zero_grad()
        opt_li.zero_grad()
        loss.backward()
        opt_x.step()
        opt_lf.step()
        opt_li.step()
        scheduler.step()

        if step % save_interval == 0:
            path_x = save_path + "x_" + str(step) + ".pth"
            torch.save(x, path_x)
            path_li = save_path + "li_" + str(step) + ".pth"
            torch.save(l_i, path_li)
            path_lf = save_path + "lf_" + str(step) + ".pth"
            torch.save(l_f, path_lf)


if __name__ == "__main__":
    model_init, model_final, model_diff = build_models()
    weight_change = get_params_vec(model_diff)

    criterion = nn.CrossEntropyLoss(reduction='none')
    opt_x = torch.optim.Adam(
        [x], lr=rec_lr, betas=(0.8, 0.9)
    )
    opt_li = torch.optim.SGD([l_i], lr=rec_lambda_lr, momentum=0.9)
    opt_lf = torch.optim.SGD([l_f], lr=rec_lambda_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_x, T_max=2000
    )

    train_dummy_dataset()
