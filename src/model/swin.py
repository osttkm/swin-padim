import torch
import torchvision.models as models

class Swinv2(torch.nn.Module):
    def __init__(self,name):
        super(Swinv2,self).__init__()
        self.name = name
        if self.name == "swin_v2_b":
            self.model = models.swin_v2_b(models.Swin_V2_B_Weights)
        elif self.name == "swin_v2_s":
            self.model = models.swin_v2_s(models.Swin_V2_S_Weights)
        elif self.name == "swin_v2_t":
            self.model = models.swin_v2_t(models.Swin_V2_T_Weights)
       
    
    def emmbeding(self,input):
        return self.model.features[0](input)
    def layer1(self,input):
        output = self.model.features[1](input)
        return self.model.features[2](output)
    def layer2(self,input):
        output = self.model.features[3](input)
        return self.model.features[4](output)
    def layer3(self,input):
        output = self.model.features[5](input)
        return self.model.features[6](output)
    def layer4(self,input):
        return self.model.features[7](input)
    def info(self):
        print("input is 224x224")

class Swin(torch.nn.Module):
    def __init__(self,name):
        super(Swin,self).__init__()
        self.name = name
        if self.name == "swin_b":
            self.model = models.swin_b(models.Swin_B_Weights)
        elif self.name == "swin_s":
            self.model = models.swin_s(models.Swin_S_Weights)
        elif self.name == "swin_t":
            self.model = models.swin_b(models.Swin_T_Weights)
    def emmbeding(self,input):
        return self.model.features[0](input)
    def layer1(self,input):
        output = self.model.features[1](input)
        return self.model.features[2](output)
    def layer2(self,input):
        output = self.model.features[3](input)
        return self.model.features[4](output)
    def layer3(self,input):
        output = self.model.features[5](input)
        return self.model.features[6](output)
    def layer4(self,input):
        return self.model.features[7](input)
    def info(self):
        print("input is 224x224")

# main文
if __name__ == "__main__":
    swin = Swinv2("swin_v2_b")
    print(swin)