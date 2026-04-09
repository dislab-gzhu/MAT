import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as st
import torchvision.models as models
import timm
import os
from PIL import Image
import pandas as pd
from torchvision import transforms
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
# import torch_dct as dct
import gc
from dataload import *
from collections import OrderedDict
# from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR100_MEAN, CIFAR100_STD, \
#     DMWideResNet, Swish, DMPreActResNet
# from robustbench.model_zoo.architectures.resnet import PreActBlock, PreActResNet, PreActBlockV2, \
#     ResNet, BasicBlock
# from robustbench.model_zoo.architectures.resnext import CifarResNeXt, ResNeXtBottleneck
# from robustbench.model_zoo.architectures.wide_resnet import WideResNet
# from robustbench.model_zoo.enums import ThreatModel
# from robustbench.model_zoo.architectures.CARD_resnet import LRR_ResNet, WidePreActResNet
# from defense.hgd.inres import get_model as get_model2
# from defense.rs.architectures import get_architecture
# from defense.rs.datasets import load_images, get_labels, load_labels
# from defense.rs.core import Smooth
# from defense.nrp.networks import *
# from defense.nrp.utils import *


def prepare_comparison_data(*results_dicts, method_names):
    if len(results_dicts) != len(method_names):
        raise ValueError("结果字典的数量必须与方法名称的数量相匹配")

    # 模型名称映射（与论文格式一致）
    name_mapping = {
        'resnet18': 'ResNet-18',
        'resnet101': 'ResNet-101',
        'inception_v3': 'Inception-v3',
        'resnet50': 'ResNet-50',
        'densenet121': 'DenseNet-121',
        'mobilenet_v2': 'MobileNet',
        'vit_b_16': 'ViT-B',
        'vit_b_32': 'ViT-B-32',
        'vit_l_16': 'VIT-L16',
        'vit_l_32': 'VIT-L',
        'swin_t': 'Swin-T',
        'mobilevit_s': 'MobileViT',
        'vgg16': 'VGG-16',
        'vit_base_patch32_224': 'VIT_B_32',
        'vit_small_r26_s32_224': 'ViT-Res26',
        'vit_base_r50_s16_224': 'VIT_B_RS5016',
        'vit_large_patch16_224': 'VIT_L_16_timm',
        'vgg19': 'VGG-19',
        'twins_pcpvt_base': 'Twins-B',
        'swin_s': 'Swin-S',
        'swin_b': 'Swin-B'
    }

    # 获取所有模型名称（假设第一个结果字典包含所有模型）
    models = list(results_dicts[0].keys())

    comparison_data = {}
    for model in models:
        mapped_name = name_mapping.get(model, model)  # 如果没有映射，使用原名称
        comparison_data[mapped_name] = {
            method: results[model] for method, results in zip(method_names, results_dicts)
        }

    return comparison_data


def plot_asr_comparison(comparison_data, method_names, output_path):
    """绘制专业的 ASR 对比柱状图"""
    plt.figure(figsize=(14, 7))

    # 扩展颜色列表（支持更多方法）
    COLORS = ['#2F6D9E', '#D76032', '#4A9C8A', '#F0C808', '#8C564B', '#E377C2', '#17BECF', '#BCBD22', '#7F7F7F']

    # 模型顺序（保持一致性）
    # ordered_models = [
    #     'ResNet-18', 'Swin_T', 'Swin_S', 'Swin_B', 'VIT_B_16', 'VIT_B_32',
    #     'VIT_L_16', 'VIT_L_32', 'VIT_S_RS2632', 'MobileViT', 'Twins_B'
    # ]

    # ordered_models = [
    #     'ResNet-18', 'ResNet-101', 'Inception-v3', 'ResNet-50', 'DenseNet-121', 'MobileNet', 'VIT_B_16', 'Swin_T'
    # ]

    # ordered_models = [
    # 'augmix', 'cutmix', 'noisymix', 'noisymix_new', 'sin_in', 'ResNet50_adv', 'ResNet101_adv','vit_b_adv','ResNet50_Chen2024']

    ordered_models = [
        'ResNet-50', 'ResNet-101', 'Inception-v3', 'ViT-B', 'ViT-B-32', 'Swin-T', 'Twins-B']

    # ordered_models = ['Wang2023_WRN70', 'Cui2023_autoaug', 'Cui2023_WRN34', 'Chen2024_WRN', 'WRN34_OAAT', 'WRN34_LAS_AT']
    # 动态计算柱宽
    num_methods = len(method_names)
    bar_width = 0.8 / num_methods  # 总宽度 0.8 均分给每个方法
    x_indices = np.arange(len(ordered_models))

    # 绘制每个方法的柱状图
    bars_list = []
    for idx, method in enumerate(method_names):
        values = [comparison_data[m].get(method, 0) for m in ordered_models]  # 缺失值默认为 0
        bars = plt.bar(
            x_indices + idx * bar_width,
            values,
            bar_width,
            color=COLORS[idx % len(COLORS)],  # 颜色循环使用
            label=method,
            edgecolor='white'
        )
        bars_list.append(bars)

    # 美化图表
    plt.title('Cross-Model Attack Success Rate Comparison', fontsize=14, pad=15)
    plt.xlabel('target modle', fontsize=12, labelpad=10)
    plt.ylabel('ASR (%)', fontsize=12, labelpad=10)
    plt.xticks(x_indices + (num_methods - 1) * bar_width / 2, ordered_models, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 添加数值标签
    for method_idx, bars in enumerate(bars_list):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只标注非零值
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=COLORS[method_idx % len(COLORS)]
                )

    # 图例优化
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=num_methods,
        frameon=False,
        fontsize=11
    )

    # 保存高清图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"对比结果已保存至：{output_path}")


def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


def save_images1(adversaries, filenames, output_dir):
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))


def wrap_model(model):
    model_name = model.__class__.__name__
    Resize = 224
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model)  # 将 PreprocessingModel 和原始模型组合成一个序列模型，确保输入数据先经过预处理。


def load_model(model_name, device):
    def load_single_model(model_name):
        if model_name in models.__dict__.keys():
            print(f'=> Loading model {model_name} from torchvision.models')
            model = models.__dict__[model_name](weights="DEFAULT")
        elif model_name in timm.list_models():
            print(f'=> Loading model {model_name} from timm.models')
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError(f'Model {model_name} not supported')
        return wrap_model(model.eval().to(device))

    if isinstance(model_name, list):  # 检查输入是否为列表，列表即为做集成攻击
        return EnsembleModel([load_single_model(name) for name in model_name])
    else:
        return load_single_model(model_name)


def wrap_model_adv(model, model_name):
    # 定义需要额外归一化的对抗训练模型列表
    adv_models = ['ResNet50_adv', 'ResNet101_adv', 'vit_b_adv']
    defense = ['HGD']
    # 检查模型名称是否在需要归一化的列表中
    if model_name in adv_models:
        # 对抗训练模型需要额外的归一化，使用 ImageNet 的标准均值和标准差
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        preprocess = PreprocessingModel(resize=224, mean=mean, std=std)
        return nn.Sequential(preprocess, model)  # 将预处理和模型组合为序列模型
    else:
        return model


def load_defense_model(weight_folder='/home/dis/students/lc/code/SIA/defense1', weight_file=None, device='cuda'):
    model_mapping = {
        'augmix.pt': lambda: models.resnet50(pretrained=False),
        'cutmix.pth.tar': lambda: models.resnet50(pretrained=False),
        'noisymix.pt': lambda: models.resnet50(pretrained=False),
        'noisymix_new.pt': lambda: models.resnet50(pretrained=False),
        'sin_in.pt': lambda: models.resnet50(pretrained=False),
        'ResNet50_adv.pth': lambda: models.resnet50(pretrained=False),
        'ResNet101_adv.pth': lambda: models.resnet101(pretrained=False),
        'vit_b_adv.pth': lambda: timm.create_model('vit_base_patch16_224', pretrained=False),
        'ResNet50_Chen2024.pt': lambda: models.resnet50(width_per_group=64 * 2, pretrained=False),
        'swin_b_advRM2024.pt': lambda: timm.create_model('swin_base_patch4_window7_224', pretrained=False),
        # 可以根据需要添加更多防御模型的映射
    }
    # 生成可能的候选键（按优先级排序）
    candidate_suffixes = ['.pt', '.pth', '.pth.tar', 'ckpt']
    candidate_keys = [f"{weight_file}{suffix}" for suffix in candidate_suffixes]
    # 查找匹配的键
    matched_key = None
    for key in candidate_keys:
        if key in model_mapping:
            matched_key = key
            break
    # 实例化模型
    model = model_mapping[matched_key]()  # 调用lambda函数创建模型实例
    # 构建权重文件路径
    weight_path = os.path.join(weight_folder, matched_key)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在：{weight_path}")
    # 加载权重
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    # 处理分布式训练前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # 去除可能的分布式训练前缀
        new_state_dict[name] = v
    # 加载模型参数
    model.load_state_dict(new_state_dict)
    model = wrap_model_adv(model, weight_file)
    model.eval()
    model.to(device)
    print(f"成功加载模型：{matched_key}")
    return model


class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize, antialias=True)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


def load_hgd_model(model_name, device='cuda'):
    config, inresmodel = get_model2()
    model = inresmodel.net
    checkpoint = torch.load('/home/dis/students/lc/code/SIA/defense1/HGD.ckpt')
    state_dict = checkpoint['state_dict']
    # 去除键名中的"net."前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('net.', '')  # 关键：移除前缀
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)
    print(f"成功加载模型{model_name}")
    return model


def RS_load_eval(model_name, device='cuda'):
    # load the base classifier
    checkpoint = torch.load('/home/dis/students/lc/code/SIA/defense1/RS.pth.tar')
    base_classifier = get_architecture('resnet50', 'imagenet')
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    f2l = load_labels('/home/dis/students/lc/code/SIA/Input/val_rs1.csv')
    smoothed_classifier = Smooth(base_classifier, 1000, 0.5)

    # prepare output file
    imagenet_dataset = AdvDataset1(root_dir='/home/dis/students/lc/code/SIA/Input', mode='train', img_size=224)
    data_loader = DataLoader(imagenet_dataset, batch_size=64, shuffle=True, num_workers=0)
    correct, total = 0, 0
    for images, labels, filename in data_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            prediction = smoothed_classifier.predict(images, 1000, 0.5, 32)
            if prediction == labels:
                succ += 1  # defense succeeded
            total += labels.shape[0]
            if total % 200 == 0:
                acc = 100 * (succ / total)
                print(" {:.2f}%".format(100. * succ / total))
    return acc


def HRP_load_eval(model_name, device='cuda'):
    netG = NRP_resG(3, 3, 64, 23)
    netG.load_state_dict(torch.load("/home/dis/students/lc/code/SIA/defense1/NRP_resG.pth"))
    netG = netG.to(device)
    netG.eval()

    return