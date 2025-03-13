import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os
import argparse
from utils import progress_bar
from models import *
import logging


num_classes = 10  # num of classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_train_epochs = 20
offset = 0  # index of the first selected image of dataset  


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--model_name", default="resnet", type=str)
    parser.add_argument("--num_imgs_per_class", default=2, type=int)
    parser.add_argument("--width", default=1000, type=int)
    parser.add_argument("--bs", default=20, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    return parser.parse_args()


def set_logger(log_name, log_dir, args):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class*num_classes}_bsz{args.bs}_lr{args.lr}.log",
        )
    )
    file_handler.setLevel(logging.DEBUG)

    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # console_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def build_model(num_classes, args):
    if args.model_name == "resnet":
        model = ResNet18()
        num_ftrs = model.linear.in_features
        model.linear = nn.Linear(num_ftrs, num_classes)
        path_pretrained = "./models/pretrained.pth"
        weights_pretrained = torch.load(path_pretrained)
        model.load_state_dict(weights_pretrained)
    elif args.model_name == "mlp":
        model = mlp(3072, [args.width, args.width], num_classes)
        args.model_name = f"mlp_{args.width}"
    elif args.model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)
    else:
        raise NotImplementedError(f"Model {args.model_name} not support yet")

    model.to(device)

    return model


def build_optimizer():
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0,
    )
    return optimizer


def build_dataset():
    new_trainset = []
    new_data = []
    train_pth = []
    # transform data
    if args.dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )

    elif args.dataset == "tiny":
        data_dir = "./tiny-imagenet-200"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
                ),
            ]
        )
        trainset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=transform
        )
        class_ids = [23, 45, 65, 34, 25, 12, 67, 187, 135, 5] # random choice is also ok
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not support yet")

    # get the first `args.num_imgs_per_class`(idx offset by `offset`) images
    # of `num_classes` classes, and combine them into a new dataset,
    # Then we will train the model on these images
    if args.dataset == "cifar10":
        for img_idx in range(len(trainset)):
            real_img_idx = img_idx + offset
            if real_img_idx < len(trainset):
                if count[trainset.targets[real_img_idx]] < args.num_imgs_per_class:
                    new_trainset.append(trainset[real_img_idx])
                    count[trainset.targets[real_img_idx]] += 1
                    new_data.append(trainset[real_img_idx][0])

        for data in new_data:
            new_data = data.reshape((3, 32, 32)).numpy()
            train_pth.append(new_data)

    elif args.dataset == "tiny":
        for img_idx in range(len(trainset)):
            if trainset.targets[img_idx] in class_ids:
                label_i = class_ids.index(trainset.targets[img_idx])
                if count[label_i] < args.num_imgs_per_class:
                    new_trainset.append(tuple([trainset[img_idx][0], label_i]))
                    count[label_i] += 1
                    new_data.append(trainset[img_idx][0])

        for data in new_data:
            newdata = data.reshape((3, 64, 64)).numpy()
            train_pth.append(newdata)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not support yet")

    # save the original images
    train_pth = torch.tensor(train_pth)
    train_dir = "./ckpts/datasets/train"
    os.makedirs(train_dir, exist_ok=True)
    ori_imgs_filename = (
        f"{args.dataset}-{str(args.num_imgs_per_class * num_classes)}-{str(offset)}.pth"
    )
    torch.save(train_pth, os.path.join(train_dir, ori_imgs_filename))

    train_loader = torch.utils.data.DataLoader(
        new_trainset, batch_size=args.bs, shuffle=False
    )
    return train_loader


def train(epoch):
    logger.info("\nEpoch: %d" % epoch)
    model.eval()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        logger.info(
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,)
            )

        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            ),
        )


if __name__ == "__main__":
    args = parse_args()
    logger = set_logger("train_logger", "logs/train", args)
    model = build_model(num_classes, args)
    model.eval()
    count = np.zeros(num_classes)

    optimizer = build_optimizer()
    train_loader = build_dataset()
    criterion = nn.CrossEntropyLoss()

    # save ckpts of \theta_0 and \theta_f
    ckpt_dir = "./ckpts/model_ckpt/"
    model_ckpt_save_dir = os.path.join(
        ckpt_dir,
        f"{args.model_name}_{args.dataset}_num_img{args.num_imgs_per_class * num_classes}_bsz{args.bs}_lr{args.lr}",
    )
    if not os.path.exists(model_ckpt_save_dir):
        os.makedirs(model_ckpt_save_dir)

    # \theta_0
    init_state_dict = model.state_dict()
    torch.save(init_state_dict, model_ckpt_save_dir + "/ori.pth")

    for epoch in range(num_train_epochs):
        train(epoch)

    # \theta_f
    final_state_dict = model.state_dict()
    torch.save(final_state_dict, model_ckpt_save_dir + "/final.pth")
