import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import pretrainedmodels
import random
import timm
import math
from attack_methods import DI, gkern, Admix
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader

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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

name_mapping = {
    'twins_pcpvt_base': 'Twins-B',
}

def get_transforms(image_size):
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

def clip_by_tensor(t, t_min, t_max):
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images, names, output_dir, model_name):
    model_output_dir = os.path.join(output_dir, f"MAT-{model_name}")
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        save_name = name
        img.save(os.path.join(model_output_dir, save_name))

def check_if_already_generated(output_dir, model_name):
    model_output_dir = os.path.join(output_dir, f"MAT-{model_name}")
    if os.path.exists(model_output_dir) and len(os.listdir(model_output_dir)) > 0:
        return True
    return False

def get_multilayer_attention(model, images, gt, model_name):
    images = V(images, requires_grad=True)
    features = []
    def hook_fn(module, input, output):
        features.append(output)
    hooks = []
    base_model = model[1]
    try:
        if model_name == 'inceptionv3':
            hooks = [
                base_model.Mixed_5d.register_forward_hook(hook_fn),
                base_model.Mixed_6e.register_forward_hook(hook_fn),
                base_model.Mixed_7c.register_forward_hook(hook_fn)
            ]
        elif model_name in ['resnet50', 'resnet101']:
            hooks = [
                base_model.layer2.register_forward_hook(hook_fn),
                base_model.layer3.register_forward_hook(hook_fn),
                base_model.layer4.register_forward_hook(hook_fn)
            ]
        elif model_name == 'densenet121':
            hooks = [
                base_model.features.denseblock2.register_forward_hook(hook_fn),
                base_model.features.denseblock3.register_forward_hook(hook_fn),
                base_model.features.denseblock4.register_forward_hook(hook_fn)
            ]
        elif model_name == 'mobilenetv2_100':
            hooks = [
                base_model.blocks[5][-1].conv_pwl.register_forward_hook(hook_fn),
                base_model.blocks[6][-1].conv_pwl.register_forward_hook(hook_fn),
                base_model.conv_head.register_forward_hook(hook_fn)
            ]
        elif model_name in ['vit_base_patch16_224', 'vit_base_patch32_224']:
            hooks = [
                base_model.blocks[9].register_forward_hook(hook_fn),
                base_model.blocks[10].register_forward_hook(hook_fn),
                base_model.blocks[11].register_forward_hook(hook_fn)
            ]
        elif model_name == 'vit_large_patch32_224':
            hooks = [
                base_model.blocks[21].register_forward_hook(hook_fn),
                base_model.blocks[22].register_forward_hook(hook_fn),
                base_model.blocks[23].register_forward_hook(hook_fn)
            ]
        elif model_name == 'twins_pcpvt_base':
            try:
                hooks = [
                    base_model.blocks[2][-1].attn.register_forward_hook(hook_fn),
                    base_model.blocks[3][-1].attn.register_forward_hook(hook_fn)
                ]
            except AttributeError as e:
                raise
        elif model_name == 'swin_tiny_patch4_window7_224':
            hooks = [
                base_model.layers[2].register_forward_hook(hook_fn),
                base_model.layers[3].register_forward_hook(hook_fn)
            ]
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except AttributeError as e:
        output = model(images)
        attention = torch.ones(images.shape[0], 1, images.shape[2], images.shape[3]).cuda()
        return attention
    output = model(images)
    for hook in hooks:
        hook.remove()
    if not features:
        attention = torch.ones(images.shape[0], 1, images.shape[2], images.shape[3]).cuda()
        return attention
    loss = -output[range(len(gt)), gt].sum()
    model.zero_grad()
    grads = torch.autograd.grad(loss, [f for f in features], create_graph=False)
    attention = 0
    for feature, grad in zip(features, grads):
        if model_name in ['vit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch32_224']:
            batch, num_patches, dim = feature.shape
            spatial_size = int(math.sqrt(num_patches - 1))
            weight = grad.abs()[:, 1:, :].mean(dim=2, keepdim=True)
            weighted_feature = (feature[:, 1:, :] * weight).sum(dim=2, keepdim=True)
            weighted_feature = weighted_feature.view(batch, 1, spatial_size, spatial_size)
        elif model_name in ['twins_pcpvt_base']:
            if len(feature.shape) == 4:
                weight = grad.abs().mean(dim=1, keepdim=True)
                weighted_feature = (feature * weight).mean(dim=1, keepdim=True)
            else:
                batch, num_patches, dim = feature.shape
                spatial_size = int(math.sqrt(num_patches))
                weight = grad.abs().mean(dim=2, keepdim=True)
                weighted_feature = (feature * weight).sum(dim=2, keepdim=True)
                weighted_feature = weighted_feature.view(batch, 1, spatial_size, spatial_size)
        else:
            weight = grad.abs().mean(dim=1, keepdim=True)
            weighted_feature = (feature * weight).mean(dim=1, keepdim=True)
        weighted_feature = F.interpolate(weighted_feature, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
        attention += weighted_feature
    attention = attention / len(features)
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    return attention

def DI_transform(x, max_size=330):
    batch_size, channels, height, width = x.shape
    rnd_size = torch.randint(height, max_size + 1, (1,)).item()
    resize = T.Resize((rnd_size, rnd_size))
    x_resized = resize(x)
    pad_size = (max_size - rnd_size) // 2
    padding = (pad_size, pad_size, pad_size, pad_size)
    x_padded = F.pad(x_resized, padding, mode='constant', value=0)
    if x_padded.shape[-1] > height:
        start_h = torch.randint(0, x_padded.shape[-2] - height + 1, (1,)).item()
        start_w = torch.randint(0, x_padded.shape[-1] - width + 1, (1,)).item()
        x_padded = x_padded[:, :, start_h:start_h + height, start_w:start_w + width]
    return x_padded

def random_augment(x, attention):
    batch_size, _, height, width = x.shape
    augmented_images = []
    for i in range(batch_size):
        img = x[i]
        att = attention[i]
        transforms_list = [
            T.RandomRotation(degrees=(-10, 10)),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.RandomResizedCrop(size=(height, width), scale=(0.9, 1.1))
            
        ]
        selected_transforms = random.sample(transforms_list, 1)
        transform = T.Compose(selected_transforms)
        img_aug = transform(img)
        img_adjusted = img + (img_aug - img) * (1 - att)
        augmented_images.append(img_adjusted)
    return torch.stack(augmented_images)

def MAT_with_attention(images, gt, model, min, max, model_name, image_size):
    Resize = T.Resize(size=(image_size, image_size))
    momentum = opt.momentum
    num_iter = opt.num_iter_set
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone().detach().cuda()
    grad = torch.zeros_like(x).cuda()
    N = opt.N
    beta = opt.beta
    gamma = opt.gamma
    attention = get_multilayer_attention(model, images, gt, model_name)
    for i in range(num_iter):
        noise = 0
        #### core function will be provided soon
        noise = noise / N
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        grad = momentum * grad + noise
        x = x + alpha * torch.sign(grad)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def load_model(model_name):
    if model_name in ['inceptionv3', 'resnet50', 'resnet101', 'densenet121']:
        base_model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').eval().cuda()
    else:
        base_model = timm.create_model(model_name, pretrained=True, num_classes=1000).eval().cuda()
    if model_name == 'inceptionv3':
        image_size = 299
        opt.mean = np.array([0.5, 0.5, 0.5])
        opt.std = np.array([0.5, 0.5, 0.5])
    else:
        image_size = 224
        opt.mean = np.array([0.485, 0.456, 0.406])
        opt.std = np.array([0.229, 0.224, 0.225])
    opt.image_width = image_size
    opt.image_height = image_size
    model = torch.nn.Sequential(
        Normalize(opt.mean, opt.std),
        base_model
    ).cuda()
    return model, image_size

def main():
    for model_name, mapped_name in name_mapping.items():
        if check_if_already_generated(opt.output_dir, mapped_name):
            continue
        model, image_size = load_model(model_name)
        transforms = get_transforms(image_size)
        X = ImageNet(opt.input_dir, opt.input_csv, transforms)
        data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for images, images_ID, gt_cpu in tqdm(data_loader, desc=f" ({mapped_name})"):
            gt = gt_cpu.cuda()
            images = images.cuda()
            images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
            images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
            adv_img = MAT_with_attention(images, gt, model, images_min, images_max, model_name, image_size)
            adv_img_np = adv_img.cpu().numpy()
            adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
            save_image(adv_img_np, images_ID, opt.output_dir, mapped_name)

if __name__ == '__main__':
    main()
