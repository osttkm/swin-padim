from efficientnet_pytorch import EfficientNet
from matplotlib.pyplot import axis
import torch
from torchinfo import summary
from torchvision.models import wide_resnet50_2, resnet18, resnet50
import logging
import torchvision.transforms as T
import glob
import os
from PIL import Image
import numpy as np
import cv2

# file_names = glob.glob('./*.bmp')
# print(file_names)
# for file_name in file_names:
#     os.remove(file_name)

logging.basicConfig(
    filename='./log.txt',
    filemode='w',
    format="%(asctime)s %(message)s",
    level =logging.INFO)

model = resnet18(pretrained=True, progress=True)
model1 = EfficientNet.from_pretrained("efficientnet-b5")
logging.info(summary(model1,input_size=(1,3,224,224)))
print(model1.get_parameter)

# file_names = glob.glob('./datasets/Rivet_scr/ground_truth/NG/*.jpg')
# # save_path1 = glob.glob('./datasets/Rivet_scr_resize/ground_truth/NG/00')
# # save_path2 = glob.glob('./datasets/Rivet_scr_resize/ground_truth/NG/0')

# for file,i in zip(file_names,range(6,41)):
#     print(i)
#     img = Image.open(file)
#     img_resize_lanczos = img.resize((448,448), Image.LANCZOS) 
#     img_resize_lanczos.save(file.replace("Rivet_scr","Rivet_scr_resize"))


