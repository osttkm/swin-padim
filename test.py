import torch
import torchvision.models as models
from transformers import CLIPModel, CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval().to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
import pdb;pdb.set_trace()

data = torch.rand((1,3,224,224)).to(device)

clip_emmbedding = clip.vision_model.embeddings
clip_prenorm = clip.vision_model.pre_layrnorm
# # input 1,3,224,224  output 1,257,1024
clip_encode_layers = clip.vision_model.encoder.layers
clip_layer_norm =clip.vision_model.post_layernorm
import pdb;pdb.set_trace()