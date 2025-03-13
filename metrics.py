import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision.utils import save_image
import os
import argparse
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--tlr", default=1e-2, type=float)
parser.add_argument("--rlr", default=1e-2, type=float)
parser.add_argument("--rec_lr", default=1e-2, type=float)
parser.add_argument("--rec_lambda_lr", default=1e-6, type=float)
parser.add_argument("--model_name", default="resnet", type=str)
parser.add_argument("--num_imgs_per_class", default=2, type=int)
parser.add_argument("--bs", default=20, type=int)
parser.add_argument("--rbs", default=20, type=int)
parser.add_argument("--steps", default=8, type=int)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--resolution", default=32, type=int)
parser.add_argument("--linear",
        default=0,
        type=int,
        help="0 means SimuDy, 1 means Buzaglo et al.’s method, 2 means Loo et al.’s method",
    )
parser.add_argument("--eval_step", default=12, type=int)
parser.add_argument("--eval_epoch", default=5, type=int)

args = parser.parse_args()
offset = 0
num_classes = 10

def get_best_pairs_ssim(train_images, recons):
    recon_flattened = recons.reshape(recons.shape[0], -1)
    dists = np.zeros([train_images.shape[0], recon_flattened.shape[0]])

    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            dists[i, j] = 1 - ssim(
                train_images[i],
                recons[j],
                data_range=train_images[i].max() - train_images[i].min(),
                channel_axis=2,
            )

    best_pairs = []
    pair_dists = []

    ans = 0

    while len(best_pairs) < train_images.shape[0] and np.min(dists) < np.inf:
        loc = np.unravel_index(dists.argmin(), dists.shape)
        # print(loc)
        ans += 1 - dists.min()
        best_pairs.append([loc[0], loc[1]])

        pair_dists.append(dists[loc])

        dists[loc[0]] = np.inf
        dists[:, loc[1]] = np.inf

    return np.array(best_pairs), ans


recon_images_dir = "./ckpts/recover_x/"
if args.linear == 0:
    recon_images_dict = torch.load(
        os.path.join(
            recon_images_dir,
            f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}_bsz{args.bs}"
            + f"_tlr{args.tlr}_nsteps{args.steps}_rbs{args.rbs}_rlr{args.rlr}_reclr{args.rec_lr}"
            + f"/{args.eval_epoch}_{args.eval_step * 100}.pth",
        ),
        map_location=torch.device("cpu"),
    )

    bs_num = len(recon_images_dict)  # the number of batches
    bs = len(recon_images_dict[0])  # batch size
    img_num = 0  # the number of recosntructed images
    for i in range(bs_num):
        img_num += len(recon_images_dict[i])
    recon_images = torch.zeros(img_num, 3, args.resolution, args.resolution)
    for i in range(bs_num):
        if i == bs_num - 1:
            recon_images[i * bs : img_num] = recon_images_dict[i].detach().cpu()
        else:
            recon_images[i * bs : i * bs + bs] = recon_images_dict[i].detach().cpu()
elif args.linear == 1:
    recon_images = torch.load(
        os.path.join(
            recon_images_dir,
            f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}_bsz{args.bs}"
            + f"_tlr{args.tlr}_1step_reclr{args.rec_lr}_lambdalr{args.rec_lambda_lr}"
            + f"/x_{args.eval_step * 100}.pth",
        ),
        map_location=torch.device("cpu"),
    )
    recon_images = recon_images.detach().cpu()
elif args.linear == 2:
    recon_images = torch.load(
        os.path.join(
            recon_images_dir,
            f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}_bsz{args.bs}"
            + f"_tlr{args.tlr}_2step_reclr{args.rec_lr}_lambdalr{args.rec_lambda_lr}"
            + f"/x_{args.eval_step * 100}.pth",
        ),
        map_location=torch.device("cpu"),
    )
    recon_images = recon_images.detach().cpu()

# load real images, which serves as trainset in sgd_train.py
training_set_dict = (
    torch.load(f"./ckpts/datasets/train/{args.dataset}-{args.num_imgs_per_class * 10}-{offset}.pth").detach().cpu()
)

train_images = np.array(training_set_dict)
recon_images = np.array(recon_images)

train_images = train_images.transpose(0, 2, 3, 1)
recon_images = recon_images.transpose(0, 2, 3, 1)

best_pairs, ans = get_best_pairs_ssim(train_images, recon_images)
img_num = len(training_set_dict)
avg_ssim = ans / img_num

train_images_best = train_images[best_pairs[:, 0][:img_num]]
recon_images_ordered = recon_images[best_pairs[:, 1][:img_num]]

train_images_best = torch.tensor(train_images_best)
train_images_best = train_images_best.permute(0, 3, 1, 2)

recon_images_ordered = torch.tensor(recon_images_ordered)
recon_images_ordered = recon_images_ordered.permute(0, 3, 1, 2)

mean = torch.tensor([0.4914, 0.4822, 0.4465]).to("cpu")
std = torch.tensor([0.2023, 0.1994, 0.2010]).to("cpu")

train_pics = train_images_best * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
recon_pics = recon_images_ordered * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

if args.linear == 0:
    metrics_dir = (
        f"./metrics/{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}_bsz{args.bs}"
        + f"_tlr{args.tlr}_nsteps{args.steps}_rbs{args.rbs}_rlr{args.rlr}_reclr{args.rec_lr}"
        + f"_eepoch{args.eval_epoch}_estep{args.eval_step}"
    )
elif args.linear == 1:
    metrics_dir = (
        f"./metrics/{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}_bsz{args.bs}"
        + f"_tlr{args.tlr}_1step_reclr{args.rec_lr}_lambdalr{args.rec_lambda_lr}"
        + f"_estep{args.eval_step}"
    )
elif args.linear == 2:
    metrics_dir = (
        f"./metrics/{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}_bsz{args.bs}"
        + f"_tlr{args.tlr}_2step_reclr{args.rec_lr}_lambdalr{args.rec_lambda_lr}"
        + f"_estep{args.eval_step}"
    )
ori_pics_dir = os.path.join(metrics_dir, "ori")
recover_pics_dir = os.path.join(metrics_dir, "recover")
os.makedirs(ori_pics_dir, exist_ok=True)
os.makedirs(recover_pics_dir, exist_ok=True)

for i in range(len(train_pics)):
    save_image(
        train_pics[i].double(),
        os.path.join(ori_pics_dir, f"{i}.jpg"),
    )
    save_image(
        recon_pics[i].double(),
        os.path.join(recover_pics_dir, f"{i}.jpg"),
    )


def calculate_mse(original, reconstructed):
    mse = F.mse_loss(original, reconstructed, reduction="mean")
    return mse.item()


def calculate_psnr(original, reconstructed):
    mse = F.mse_loss(original, reconstructed, reduction="mean")
    if mse == 0:
        return float("inf")
    max_pixel = 1.0  # Assuming the images are normalized [0, 1]
    psnr = 10 * torch.log10(max_pixel**2 / mse)
    return psnr.item()


mse = 0
psnr = 0

for i in range(len(train_pics)):
    mse += calculate_mse(train_images_best[i], recon_images_ordered[i]) / len(
        train_pics
    )
    psnr += calculate_psnr(train_images_best[i], recon_images_ordered[i]) / len(
        train_pics
    )

with open(f"{metrics_dir}/avg_ssim_{avg_ssim:06f}.txt", "w") as f:
    f.write(f"Average SSIM: {avg_ssim}\n")
    f.write(f"MSE: {mse}\n")
    f.write(f"PSNR: {psnr} dB\n")
