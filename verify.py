import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import nn
from torchvision import transforms as T
import timm
from torch.utils.data import DataLoader
import matplotlib.font_manager as fm

try:
    from Normalize import Normalize, TfNormalize
    from loader import ImageNet
    from torch_nets import tf_inception_v3
except ImportError as e:
    raise ImportError(f"缺少自定义模块: {e}. 请确保 Normalize.py, loader.py 和 torch_nets.py 文件存在并正确定义。")

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# 设置中文字体
available_fonts = fm.findSystemFonts()
if any('noto' in f.lower() for f in available_fonts):
    font_name = 'Noto Sans CJK SC'
elif any('wenquanyi' in f.lower() for f in available_fonts):
    font_name = 'WenQuanYi Zen Hei'
elif any('simsun' in f.lower() for f in available_fonts):
    font_name = 'SimSun'
else:
    print("警告：未找到支持中文的字体，建议安装 'Noto Sans CJK SC' 或 'WenQuanYi Zen Hei'。")
    font_name = 'DejaVu Sans'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置配色
sns.set_palette("husl")

# 模型名称映射
MODEL_NAME_MAPPING = {
    'resnet50_ares': 'Res50-ARES',
    'resnet101_ares': 'Res101-ARES',
    'swin_base_ares': 'SwinBase-ARES',
    'vit_base_ares': 'ViTBase-ARES',
    'Erichson2022NoisyMix_new': 'NoisyMix',
    'Geirhos2018_SIN': 'SIN',
    'Hendrycks2020AugMix': 'AugMix',
    'Wong2020Fast': 'Fast',
    'Engstrom2019Robustness': 'Robustness',
    'resnet50': 'ResNet-50',
    'resnet101': 'ResNet-101',
    'inceptionv3': 'Inc-v3',
    'densenet121': 'DenseNet-121',
    'mobilenetv2_100': 'MobileNet',
    'vit_base_patch32_224': 'ViT-B-32',
    'swin_tiny_patch4_window7_224': 'Swin-T',
    'twins_pcpvt_base': 'Twins-B',
    'vit_small_patch16_224': 'ViT-S/16',
    'deit_tiny_patch16_224': 'DeiT-T/16',
    'deit_small_patch16_224': 'DeiT-S/16',
    'deit_base_patch16_224': 'DeiT-B/16',
    'swin_base_patch4_window7_224': 'Swin-B/4/7'
}

# 模型权重路径
MODEL_CONFIGS = {
    'resnet50_ares': '/home/dis/students/dpf/.cache/torch/hub/checkpoints/ARES_ResNet50_AT.pth',
    'resnet101_ares': '/home/dis/students/dpf/.cache/torch/hub/checkpoints/ARES_ResNet101_AT.pth',
    'swin_base_ares': '/home/dis/students/dpf/.cache/torch/hub/checkpoints/ARES_Swin_base_patch4_window7_224_AT.pth',
    'vit_base_ares': '/home/dis/students/dpf/.cache/torch/hub/checkpoints/ARES_ViT_base_patch16_224_AT.pth',
    'resnet50': '/home/dis/students/dpf/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth',
    'resnet101': '/home/dis/students/dpf/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth',
}

# 配置参数
batch_size = 10
input_csv = './dataset/images.csv'
input_dir = './dataset/images'
models_path = './models/'
mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]


def get_model(model_name):
    try:
        if model_name == 'inceptionv3':
            model_path = os.path.join(models_path, 'tf_inception_v3.npy')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型权重文件 {model_path} 不存在")
            model = nn.Sequential(TfNormalize('tensorflow'), tf_inception_v3.KitModel(model_path).eval().cuda())
        elif model_name in MODEL_CONFIGS:
            model = timm.create_model(model_name.replace('_ares', ''), pretrained=False)
            checkpoint = torch.load(MODEL_CONFIGS[model_name], map_location='cuda:0')
            state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model = nn.Sequential(Normalize(mean=mu, std=sigma), model)
        else:
            model = timm.create_model(model_name, pretrained=True)
            model = nn.Sequential(Normalize(mean=mu, std=sigma), model)
        return model.eval().cuda()
    except Exception as e:
        print(f"加载模型 {model_name} 失败: {e}")
        return None


def verify(model_name, adv_dir):
    if not os.path.exists(adv_dir):
        print(f"对抗样本目录 {adv_dir} 不存在，跳过 {model_name} 的评估")
        return 0.0

    model = get_model(model_name)
    if model is None:
        return 0.0

    transform = T.Compose([T.Resize((299, 299) if model_name == 'inceptionv3' else (224, 224)), T.ToTensor()])
    dataset = ImageNet(adv_dir, input_csv, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    error_count = 0
    total_samples = 0

    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            output = model(images)
            if model_name == 'inceptionv3' and isinstance(output, tuple):
                output = output[0]
            preds = output.argmax(1)
            error_count += (preds != (gt + 1 if model_name == 'inceptionv3' else gt)).sum().item()
        total_samples += images.size(0)

    if total_samples == 0:
        print(f"警告: {model_name} 在 {adv_dir} 没有样本")
        return 0.0

    attack_success_rate = error_count / total_samples
    print(f"{model_name} 在 {adv_dir} 下的攻击成功率: {attack_success_rate:.2%}")
    return attack_success_rate


def visualize_results(results, save_path, model_name):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = [MODEL_NAME_MAPPING.get(k, k) for k in results.keys()]
    values = [results[model]['FIA'] * 100 for model in results.keys()]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.ylim(0, 100)
    plt.ylabel('Attack Success Rate (%)')
    plt.title(f'FIA Attack Performance on {MODEL_NAME_MAPPING.get(model_name, model_name)}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存至 {save_path}")


def save_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for model_name, methods in results.items():
            f.write(f"模型: {MODEL_NAME_MAPPING.get(model_name, model_name)}\n")
            f.write(f"  FIA: {methods['FIA']:.2%}\n\n")
    print(f"结果已保存至 {save_path}")


def main():
    # 测试模型列表
    models = [
        # 'resnet50_ares',
        # 'resnet101_ares',
        # 'swin_base_ares',
        # 'vit_base_ares',
        # 'Erichson2022NoisyMix_new',
        # 'Geirhos2018_SIN',
        # 'Hendrycks2020AugMix',
        # 'Wong2020Fast',
        # 'Engstrom2019Robustness',
        'inceptionv3',
        'densenet121',
        'mobilenetv2_100',
        'vit_base_patch32_224',
        'swin_tiny_patch4_window7_224',
        'twins_pcpvt_base',
        'vit_small_patch16_224',
        'deit_tiny_patch16_224',
        'deit_small_patch16_224',
        'deit_base_patch16_224',
        'swin_base_patch4_window7_224'
    ]

    # FIA 攻击路径
    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/FIA/FIA-ResNet-101',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/FIA2/FIA-ResNet-101',
    #      '2': '/home/dis/students/dpf/xd/STM/FIA3/FIA-ResNet-101'
    # }

    fia_paths = {
        # 'densenet121': '/home/dis/students/dpf/xd/STM/FIA/FIA-ResNet-101',
        'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/FIA/FIA-ResNet-101',
        '1': '/home/dis/students/dpf/xd/STM/FMAA/FMAA-ResNet-101',
        '2': '/home/dis/students/dpf/xd/STM/NAA/NAA-ResNet-101',
        '3': '/home/dis/students/dpf/xd/STM/FIA/FIA-ViT-B',
        '4': '/home/dis/students/dpf/xd/STM/NAA/NAA-ViT-B',
        '5': '/home/dis/students/dpf/xd/STM/NAA1/NAA-ViT-B',
        '6': '/home/dis/students/dpf/xd/STM/FMAA/FMAA-ViT-B',
        '7': '/home/dis/students/dpf/xd/STM/FIA1/FIA-ViT-B',
    }
    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/FIA1/FIA-Inception-v3',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/FIA1/FIA-MobileNet',
    #      '101': '/home/dis/students/dpf/xd/STM/FIA1/FIA-ResNet-101'
    # }
    #
    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/FIA1/FIA-Inception-v3',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/FIA1/FIA-MobileNet',
    #     '101': '/home/dis/students/dpf/xd/STM/FIA1/FIA-ResNet-101'
    # }

    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/FIA/FIA-DenseNet-121',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/FIA/FIA-ViT-B'
    # }

    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/FMAA/FMAA-DenseNet-121',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/FMAA/FMAA-ViT-B'
    # }

    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/FMAA1/FMAA-DenseNet-121',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/FMAA1/FMAA-ViT-B'
    # }

    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/NAA/NAA-ResNet-101',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/NAA/NAA-ViT-B'
    # }

    # fia_paths = {
    #     'densenet121': '/home/dis/students/dpf/xd/STM/NAA1/NAA-ResNet-101',
    #     'vit_base_patch32_224': '/home/dis/students/dpf/xd/STM/NAA1/NAA-ViT-B'
    # }
    for pretrained_model, adv_dir in fia_paths.items():
        print(f"\n=== 评估对抗样本生成模型: {MODEL_NAME_MAPPING.get(pretrained_model, pretrained_model)} ===")
        results = {model: {'FIA': 0.0} for model in models}

        for model in models:
            success_rate = verify(model, adv_dir)
            results[model]['FIA'] = success_rate
        print("---------------------------------------------------")

        # 保存结果
        save_path_txt = f"/home/dis/students/dpf/xd/STM/result1/sss/{MODEL_NAME_MAPPING.get(pretrained_model, pretrained_model)}_attack_results.txt"
        save_results(results, save_path_txt)

        # 可视化结果
        save_path_png = f"/home/dis/students/dpf/xd/STM/result1/sss/{MODEL_NAME_MAPPING.get(pretrained_model, pretrained_model)}_attack_results.png"
        visualize_results(results, save_path_png, pretrained_model)
        print(f"=== 完成 {MODEL_NAME_MAPPING.get(pretrained_model, pretrained_model)} 的评估 ===")


if __name__ == '__main__':
    main()