import torch
import torchvision.models as models
from PIL import Image
import numpy as np
import transformers as T
from src.dataset_code.mvtec import get_mvtec_loader
from src.model.swin import Swinv2
from src.model.vit import Vit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


swinv2 = Swinv2("swin_v2_b")
swinv2.to(device)
swinv2.eval()

vit = Vit("vit_b_16")
vit.to(device)
vit.eval()


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
    assert data.shape[-1] == 224
    data,label = data.to(device),label.to(device)
    import pdb;pdb.set_trace()
    with torch.no_grad():
        swin_emmbed = swinv2.emmbeding(data)
        vit_emmbed = vit.emmbeding(data)
        print('layer直残の畳み込み')
        # import pdb;pdb.set_trace()
        swin_out = swinv2.layer1(swin_emmbed)
        vit_out = vit.layerN(1,vit_emmbed)
        print(swin_out.shape,vit_out.shape)

        swin_out = swinv2.layer1(swin_emmbed)
        vit_out = vit.layerN(2,vit_emmbed)
        print(swin_out.shape,vit_out.shape)
        # import pdb;pdb.set_trace()
        swin_out = swinv2.layer1(swin_emmbed)
        vit_out = vit.layerN(3,vit_emmbed)
        print(swin_out.shape,vit_out.shape)

        swin_out = swinv2.layer1(swin_emmbed)
        vit_out = vit.layerN(4,vit_emmbed)
        print(swin_out.shape,vit_out.shape)
    

    break
        
