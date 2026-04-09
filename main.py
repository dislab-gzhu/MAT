import argparse
from tqdm import tqdm
import numpy as np
import torch
from loader import ImageNet
from torch.utils.data import DataLoader
from MAT import get_transforms, clip_by_tensor, save_image, check_if_already_generated, MAS_with_attention, load_model

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv')
parser.add_argument('--input_dir', type=str, default='./dataset/images')
parser.add_argument('--output_dir', type=str, default='./MAT')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]))
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]))
parser.add_argument('--max_epsilon', type=float, default=16.0)
parser.add_argument('--num_iter_set', type=int, default=10)
parser.add_argument('--image_width', type=int, default=299)
parser.add_argument('--image_height', type=int, default=299)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--momentum', type=float, default=1.0)
parser.add_argument('--N', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=2.0)
opt = parser.parse_args()

name_mapping = {
    'twins_pcpvt_base': 'Twins-B',
}

def main():
    for model_name, mapped_name in name_mapping.items():
        if check_if_already_generated(opt.output_dir, mapped_name):
            continue
        model, image_size = load_model(model_name, opt)
        transforms = get_transforms(image_size)
        X = ImageNet(opt.input_dir, opt.input_csv, transforms)
        data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for images, images_ID, gt_cpu in tqdm(data_loader):
            gt = gt_cpu.cuda()
            images = images.cuda()
            images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
            images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
            adv_img = MAS_with_attention(images, gt, model, images_min, images_max, model_name, image_size, opt)
            adv_img_np = adv_img.cpu().numpy()
            adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
            save_image(adv_img_np, images_ID, opt.output_dir, mapped_name)

if __name__ == '__main__':
    main()
