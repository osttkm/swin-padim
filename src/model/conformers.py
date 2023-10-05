import torch
import torch.nn as nn
from src.model.Conformer import models as models

def get_model(name):
        if name == "conformer_b_16":
            model = models.Conformer_base_patch16(pretrained=True)
            return model
        elif name == "conformer_s_16":
            model = models.Conformer_small_patch16(pretrained=True)
            return model
        elif name == "conformer_s_32":
            model = models.Conformer_small_patch32(pretrained=True)
            return model
        elif name == "conformer_t_16":
            model = models.Conformer_tiny_patch16(pretrained=True)
            return model


# main文
if __name__ == "__main__":
    import torchvision.models as Model

    model = get_model("conformer_b_16")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data  = torch.rand((1,3,224,224)).to(device)
    model(data)
    import pdb;pdb.set_trace()

    resnet = Model.resnet18(pretrained=True)
    resnet.eval()
    resnet.to(device)
    resout = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(data))))
    # import pdb;pdb.set_trace()
    for i in range(0,14,1):
        cnn_f,trans_f = model.layerN(i,data)
        print(i,trans_f.shape)
        # import pdb;pdb.set_trace()