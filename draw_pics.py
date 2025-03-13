from PIL import Image
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tlr", default=1e-2, type=float)
parser.add_argument("--rlr", default=0, type=float)
parser.add_argument("--rec_lr", default=1e-2, type=float)
parser.add_argument("--rec_lambda_lr", default=1e-6, type=float)
parser.add_argument("--model_name", default="resnet", type=str)
parser.add_argument("--num_imgs_per_class", default=2, type=int)
parser.add_argument("--bs", default=20, type=int)
parser.add_argument("--rbs", default=20, type=int)
parser.add_argument("--steps", default=8, type=int)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--linear", default=0, type=int)
parser.add_argument("--eval_step", default=10, type=int)
parser.add_argument("--eval_epoch", default=5, type=int)
args = parser.parse_args()


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder), key=natural_sort_key):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(
            ("jpg", "jpeg", "png")
        ):
            img = Image.open(img_path)
            images.append(img)
    return images


def merge_images(folder1, folder2, images_per_row, output_path, padding=10):
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)

    # Assume all images are the same size
    width, height = images1[0].size

    # Add padding to the size calculations
    max_images = max(len(images1), len(images2))
    # max_images = 40
    total_rows = (max_images + images_per_row - 1) // images_per_row * 2
    total_columns = images_per_row

    result_width = width * total_columns + padding * (total_columns - 1)
    result_height = height * total_rows + padding * (total_rows - 1)

    # Create a new image with white background
    result_image = Image.new(
        "RGB", (result_width, result_height), color=(255, 255, 255)
    )

    for i in range(total_rows):
        for j in range(images_per_row):
            img_index = (i // 2) * images_per_row + j
            if i % 2 == 0 and img_index < len(images1):
                x = j * (width + padding)
                y = i * (height + padding)
                result_image.paste(images1[img_index], (x, y))
            elif i % 2 == 1 and img_index < len(images2):
                x = j * (width + padding)
                y = i * (height + padding)
                result_image.paste(images2[img_index], (x, y))

    result_image.save(output_path)

num_classes = 10

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
images_per_row = 10  # number of images per row in the output image
output_path = os.path.join(metrics_dir, "reconstruction.jpg")

merge_images(recover_pics_dir, ori_pics_dir, images_per_row, output_path, padding=1)
