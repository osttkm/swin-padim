import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import transformers as T
from transformers import AutoImageProcessor
from src.dataset_code.mvtec import get_mvtec_loader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# swinv2 = T.Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")

# swinv2.eval()
# swinv2.to(device)

# swinv2_emmbeding = swinv2.embeddings
# swinv2_layer1 = swinv2.encoder.layers[0]
# swinv2_layer2 = swinv2.encoder.layers[1]
# swinv2_layer3 = swinv2.encoder.layers[2]
# swinv2_layer4 = swinv2.encoder.layers[3]

# dinov2 = torch.hub.load('facebookresearch/dino:main', 'dino_vitl14')
# dinov2.eval()
# dinov2.to(device)
# vit_L = models.vit_l_16(pretrained=True)
# vit_L.eval()
# vit_L.to(device)

swin = models.swin_v2_b(models.Swin_B_Weights)
swin.eval()
swin.to(device)
import pdb;pdb.set_trace()
# resnet18  = models.resnet18(pretrained=True)

# resnet18.eval()
# resnet18.to(device)

# resnet18_layer1 = resnet18.layer1
# resnet18_layer2 = resnet18.layer2
# resnet18_layer3 = resnet18.layer3
# resnet18_layer4 = resnet18.layer4

resnet_wide50  = models.wide_resnet50_2(pretrained=True)

resnet_wide50.eval()
resnet_wide50.to(device)

resnet18_layer1 = resnet_wide50.layer1
resnet18_layer2 = resnet_wide50.layer2
resnet18_layer3 = resnet_wide50.layer3
resnet18_layer4 = resnet_wide50.layer4

def denormalize(data):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    # data is normalized with mean and std above
    # this function denormalize data
    for i in range(3):
        data[i] = data[i] * std[i] + mean[i]
    return data


train_loader, test_loader = get_mvtec_loader('bottle')
for idx,batch in enumerate(test_loader):
    data,label,_ = batch
    data,label = data.to(device),label.to(device)
    with torch.no_grad():
        emmbed = swinv2_emmbeding(data)
        res_emmbed = resnet_wide50.maxpool(resnet_wide50.relu(resnet_wide50.bn1(resnet_wide50.conv1(data))))
        # res_emmbed = resnet18.maxpool(resnet18.relu(resnet18.bn1(resnet18.conv1(data))))
        print('layer直残の畳み込み')
        print(res_emmbed.shape,emmbed[0].shape)
        # import pdb;pdb.set_trace()
        out = swinv2_layer1(emmbed[0],input_dimensions=emmbed[1])
        res_out = resnet18_layer1(res_emmbed)
        print(res_out.shape,out[0].shape)
        # tesorからpilに変換し画像として保存
        # img = transforms.ToPILImage()(denormalize(data[0].cpu()))
        # img.save(f'./test_img/{idx}.png')
        # import pdb;pdb.set_trace()

        res_out = resnet18_layer2(res_out)
        out = swinv2_layer2(out[0],input_dimensions=(data.shape[-1]//8,data.shape[-1]//8))
        print(res_out.shape,out[0].shape)

        res_out = resnet18_layer3(res_out)
        out = swinv2_layer3(out[0],input_dimensions=(data.shape[-1]//16,data.shape[-1]//16))
        print(res_out.shape,out[0].shape)

        res_out = resnet18_layer4(res_out)
        out = swinv2_layer4(out[0],input_dimensions=(data.shape[-1]//32,data.shape[-1]//32))
        print(res_out.shape,out[0].shape)

    break
        
