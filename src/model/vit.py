import torch
import torchvision.models as models
class Vit(torch.nn.Module):
    def __init__(self,name):
        super(Vit,self).__init__()
        self.name = name
        self.layer_num = 0
        self.model = None
        if self.name == "vit_b_16":
            self.model = models.vit_b_16(models.ViT_B_16_Weights)
            print(f'this model has {len(self.model.encoder.layers)} layer')
            self.layer_num = len(self.model.encoder.layers)
        elif self.name == "vit_b_32":
            self.model = models.vit_b_32(models.ViT_B_32_Weights)
            print(f'this model has {len(self.model.encoder.layers)} layer')
            self.layer_num = len(self.model.encoder.layers)
        elif self.name == "vit_l_16":
            self.model = models.vit_l_16(models.ViT_L_16_Weights)
            print(f'this model has {len(self.model.encoder.layers)} layer')
            self.layer_num = len(self.model.encoder.layers)
        elif self.name == "vit_l_32":
            self.model = models.vit_l_32(models.ViT_L_32_Weights)
            print(f'this model has {len(self.model.encoder.layers)} layer')
            self.layer_num = len(self.model.encoder.layers)
        elif self.name == "vit_h_14":
            self.model = models.vit_h_14(models.ViT_H_14_Weights)
            print(f'this model has {len(self.model.encoder.layers)} layer')
            self.layer_num = len(self.model.encoder.layers)
       
    
    def emmbeding(self,input):
        return self.model.conv_proj(input)
    
    def layerN(self,n,input):
        print(self.layer_num)
        assert self.layer_num >= n
        output=0
        for n in range(0,n,1):
            if n==0:
                import pdb;pdb.set_trace()
                output = self.model.encoder.layers[0](input)
            else:
                output = self.model.encoder.layers[n-1](output)
        return output
