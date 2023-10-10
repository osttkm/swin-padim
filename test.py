import torch
import torchvision.models as models
from transformers import CLIPModel, CLIPProcessor,AutoFeatureExtractor,ViTForImageClassification
from PIL import Image

from transformers import AutoImageProcessor, ViTMAEForPreTraining

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img = Image.open('/home/dataset/mvtec/hazelnut/test/good/000.png')



"""MAEの特徴マップ確認"""
processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
# model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
model = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")

model.eval()
model.to(device)

embeddings = model.vit.embeddings
encoder = model.vit.encoder.layer
layernorm = model.vit.layernorm
# decoder_embedding = model.decoder.decoder_embed
# decode_layer = model.decoder.decoder_layers  #0~7の8層
# decode_norm = model.decoder.decoder_norm
# decode_pred = model.decoder.decoder_pred

_input = processor(images=img, return_tensors="pt")['pixel_values'].to(device)
out = embeddings(_input)[0]


# outがtappleかどうかを判定
def if_tuple(out):
    if isinstance(out, tuple):
        print(out[0].shape)
        return out[0]
    else:
        # print('not tapple!!')
        print(out.shape)
        return out
# import pdb;pdb.set_trace()
import pdb;pdb.set_trace()
for i in range(12):
    out = if_tuple(out)
    if i == 0:
        out = encoder[i](out)
import pdb;pdb.set_trace()
out = layernorm(out)
out = decoder_embedding(out)
for i in range(8):
    out = if_tuple(out)
    out = decode_layer[i](out)
out = decode_norm(out[0])
out = decode_pred(out[0])


# outを画像として保存
# import pdb;pdb.set_trace()
# out = out[0][0]
def save_img(out):
    out = out - out.min()
    out = out / out.max()
    out = out * 255
    out = out.cpu().detach().numpy()
    out = out.astype('uint8')
    out = Image.fromarray(out)
    out.save('test.png')




"""vit_bの特徴マップ確認"""
# model = models.vit_b_16(models.ViT_B_16_Weights)
# model.eval()
# model.to(device)
# embeddings = model.conv_proj
# dropout = model.encoder.dropout
# encoder = model.encoder.layers
# out = embeddings(data)
# out = dropout(out)
# out = out - out.min()
# out = out / out.max()
# out = out * 255
# out = out.cpu().detach().numpy()
# out = out.astype('uint8')
# import pdb;pdb.set_trace()
# for i in range(out.shape[1]):
#     _out = out[0][i]
#     _out = Image.fromarray(_out)
#     _out.save('test.png')
#     import pdb;pdb.set_trace()
