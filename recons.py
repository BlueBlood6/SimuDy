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
        "--rbs",
        default=0,
        type=int,
        help="batch size used for simulating the training dynamics",
    )
    parser.add_argument(
        "--steps", 
        default=8, 
        type=int, 
        help="Number of steps to simulate the training dynamics",
    )
    parser.add_argument(
        "--rlr",
        default=0,
        type=float,
        help="Assuming tlr is unknown, we use rlr to train the model and obtain the simulated training dynamics",
    )
    parser.add_argument(
        "--grid_search",
        default=0,
        type=int,
        help="0 means no grid search, and 1 means grid search",
    )
    parser.add_argument(
        "--loss_type",
        default=0,
        type=int,
        help="0 means cosine similarity loss, and 1 means l2 loss",
    )
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--rec_lr",
        default=1e-2,
        type=float,
        help="Learning rate used for updating the dummy dataset images",
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
            f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class*num_classes}_bsz{args.bs}_tlr{args.tlr}_rbsz{args.rbs}_rlr{args.rlr}.log",
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
grid_search = args.grid_search

log_interval = 100
save_interval = 100

c, h, w = 3, 32, 32
extraction_init_scale = 5e-1
num_classes = 10
num_imgs_per_class = args.num_imgs_per_class
width = args.width
batch_size = args.bs
model_name = args.model_name
lr = args.tlr
steps = args.steps
rlr = args.rlr if args.rlr else lr
rbs = args.rbs if args.rbs else batch_size
reconstruction_lr = args.rec_lr
num_imgs_to_reconstruct = num_classes * num_imgs_per_class

logger = set_logger("reconstruction_logger", "logs/recons", args)

if num_imgs_to_reconstruct % rbs == 0:
    num_batch = num_imgs_to_reconstruct // rbs
else:
    num_batch = num_imgs_to_reconstruct // rbs + 1

num_steps_per_epoch = 1500
num_train_imgs_epochs = 20
if grid_search == 1:
    num_steps_per_epoch = 50
    num_train_imgs_epochs = 1

recover_x_dir = "./ckpts/recover_x/"
os.makedirs(recover_x_dir, exist_ok=True)

save_path = os.path.join(
    recover_x_dir,
    f"{model_name}_{args.dataset}_num_img{num_imgs_per_class * num_classes}"
    + f"_bsz{batch_size}_tlr{lr}_nsteps{steps}_rbs{rbs}_rlr{rlr}_reclr{reconstruction_lr}/",
)

if not os.path.exists(save_path):
    os.makedirs(save_path)


x = {}
for i in range(num_batch):
    if i == num_batch - 1:
        x[i] = torch.randn(num_imgs_to_reconstruct - rbs * (num_batch - 1), c, h, w).to(device) * extraction_init_scale
    else:
        x[i] = torch.randn(rbs, c, h, w).to(device) * extraction_init_scale
    x[i].requires_grad_(True)

random_label = np.zeros(num_imgs_to_reconstruct)
for c in range(num_classes):
    random_label[
        int(c * num_imgs_to_reconstruct / num_classes) : int(
            (c + 1) * num_imgs_to_reconstruct / num_classes
        )
    ] = c
random.shuffle(random_label)
random_label = torch.tensor(random_label).to(device)

y = {}
for i in range(num_batch):
    if i == num_batch - 1:
        y[i] = torch.zeros(num_imgs_to_reconstruct - rbs * (num_batch - 1)).to(device)
    else:
        y[i] = torch.zeros(rbs).to(device)
    for j in range(len(y[i])):
        y[i][j] = random_label[rbs * i + j]
    y[i] = y[i].long()
    
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
    model,
    start_params,
    y,
    weight_change,
    steps,
    num_batch,
    lr,
    epoch,
):

    grads = {}
    losses = {}
    values = {}
    loss = 0

    maxnorm = 20

    params = [start_params[0]]
    for i in range(steps):
        for j in range(num_batch):
            forward_params = params[-1]
            values[i * num_batch + j] = model(x[j], flat_param=forward_params)
            losses[i * num_batch + j] = criterion(values[i * num_batch + j], y[j])
            grads[i * num_batch + j] = torch.autograd.grad(
                losses[i * num_batch + j], params[-1], create_graph=True
            )[0]

            if torch.norm(grads[i * num_batch + j]) > maxnorm and epoch < 5:
                grads[i * num_batch + j] = (
                    grads[i * num_batch + j] / torch.norm(grads[i * num_batch + j]) * 5
                )
            params.append(params[-1] - lr * grads[i * num_batch + j])

    simulated_change = grads[num_batch * steps - 1]
    for i in range(num_batch * steps - 1):
        simulated_change += grads[i]

    sim = torch.nn.functional.cosine_similarity(
        simulated_change.flatten(), weight_change.flatten(), 0, 1e-20
    )
    cos_loss = 1 - sim

    tv_loss = 0
    for i in range(num_batch):
        tv_loss += 0.01 * total_variation(x[i])

    if args.loss_type == 0:
        loss = cos_loss + 0.1 * tv_loss
    elif args.loss_type == 1:
        loss = (simulated_change * lr - weight_change).pow(2).sum() + 0.1 * tv_loss

    return loss, cos_loss, tv_loss


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
    model_init = ReparamModule(model_init)

    return model_init, model_final, model_diff


def train_dummy_dataset():
    for epoch in range(num_train_imgs_epochs):
        # Actually, a step of training the dummy dataset means an epoch of training the dummy dataset
        # therefore, every image in the dummy dataset will be tuned `num_train_imgs_epochs` * `num_steps_per_epoch` times
        for step in tqdm(range(num_steps_per_epoch)):
            loss, cos_loss, tv_loss = calc_steps_loss_sgd(
                x,
                model_init,
                start_params,
                y,
                weight_change,
                steps,
                num_batch,
                rlr,
                epoch,
            )

            if step % log_interval == 0 or grid_search == 1:
                logger.info(f"epoch {epoch} | step {step}")
                logger.info(f"total_loss: {loss.item()}")
                logger.info(f"cos_loss: {cos_loss.item()}")
                logger.info(f"tv_loss: {tv_loss.item()}")

            dummy_dataset_optimizer.zero_grad()
            loss.backward()
            dummy_dataset_optimizer.step()
            scheduler.step()

            if step % save_interval == 0 and args.grid_search == 0:
                path_x = save_path + str(epoch + 0) + "_" + str(step) + ".pth"
                torch.save(x, path_x)


if __name__ == "__main__":
    model_init, model_final, model_diff = build_models()
    weight_change = get_params_vec(model_diff)
    start_params = [
        torch.cat(
            [p.data.to("cuda").reshape(-1) for p in model_init.parameters()], 0
        ).requires_grad_(True)
    ]

    criterion = nn.CrossEntropyLoss()
    dummy_dataset_optimizer = torch.optim.Adam(
        [x[i] for i in range(num_batch)], lr=reconstruction_lr, betas=(0.8, 0.9)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dummy_dataset_optimizer, T_max=100
    )

    train_dummy_dataset()
