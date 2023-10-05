import os
from unicodedata import category
from matplotlib.transforms import Transform
import numpy as np
import glob
import copy

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import cv2



"""
そもそもこのコードはMVtecから自分が使いやすい形にデータウィ作るものであって
データをもともと持っいる場合は必要のないもの？？？
"""

class Mask:
    def __init__(self):
        pass
    def __call__(self,img):
        img = np.array(img)
        # マスク用単一色画像を作成
        # r = 85 #半径
        r = 45
        mask = np.full(img.shape[:2], 255, dtype=img.dtype)
        # print("img:" + str(img.shape))
        # print("img:" + str(mask.shape))
        # cv2.circle(mask,center=(224,224),radius=r,color=0,thickness=-1)
        cv2.circle(mask,center=(112,112),radius=r,color=0,thickness=-1)
        img[mask==0] = [0,0,0]
        cv2.imwrite('./mask.png',mask)
        cv2.imwrite('./sample.png',img)
        return img


class Rivet(torch.utils.data.Dataset):

    def __init__(self, root='/home/oshita/padim/datasets', category='Rivet_scr', train: bool=True, transform=None, resize=256, cropsize=224):

        #448でやるときはcropsizeも448

        """
        :param root:        MVTecAD dataset dir/自分の場合は坂田さんに頂いたデータについて設定する
        :param category:    MVTecAD category/ もしかしたら要らないかも？データ次第
        :param train:       If it is true, the training mode
        :param transform:   pre-processing
        """
        self.root = root
        self.category = category
        self.train = train
        self.output_size = cropsize

        self.train_dir = os.path.join(root,category, 'train')
        #test dataのファイル構造はtest/ok　と tesat/ngに分かれているため統合する必要がある。
        #また、傷のみをマスクした画像もあるのでそこの扱いがどうなるかは考えること。多分使ってはいけない。傷のAUROCとか出すのならば必要になったりするのかも？
        self.test_dir = os.path.join(root,category, 'test')
        self.gt_dir = os.path.join(root, category, 'ground_truth')
        """例えば
        self.test_ok_dir = os.path.join(root, 'test','OK)
        とかみたいにわざわざ分けるよりかデータのファイル構造変えたほうがよさげ？
        とりあえずはファイル構造を変えてやってみる。また擬陽性の画像については実行後の考察の段階で実行結果もかねて調べること
        """
        # self.gt_dir = os.path.join(root, category, 'ground_truth')

        # testデータから正常データを除いている？何故？
        self.normal_class = ['OK']
        self.abnormal_class = os.listdir(self.test_dir)
        self.abnormal_class.remove(self.normal_class[0])
        
        
        # Setup transform
        if transform is None:
            self.transform_img = T.Compose([
                                        # [T.Resize(resize, T.InterpolationMode.BICUBIC),
                                        # T.CenterCrop(cropsize),
                                        # T.RandomRotation(degrees=10),
                                        Mask(),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        else:
            self.transform_img = transform
        self.transform_mask = T.Compose([
                                        # [T.Resize(resize, T.InterpolationMode.NEAREST),
                                        #  T.CenterCrop(cropsize),
                                         T.ToTensor()])

        # Setup dataset path
        # train dataについてパスを通している。trainは正常データのみであるため、ok,ngのラベルは必要ない
        if self.train:
            img_paths = glob.glob(os.path.join(
                self.train_dir, self.normal_class[0], '*.jpg'
            ))
            self.img_paths = sorted(img_paths)
            self.labels = len(self.img_paths) * [0]
            self.gt_paths = len(self.img_paths) * [None]
        # testとground_truthについて。ここには正常異常データの両方があるため、OK,NGそれぞれ判別する必要がある。
        # つまりは各々のファイルの直下にさらにOK.NGのファイル構造をとる必要があるので、注意が必要
        else:
            img_paths = []
            labels = []
            gt_paths = []
            for c in os.listdir(self.test_dir):
                paths = glob.glob(os.path.join(
                    self.test_dir, c, '*.jpg'
                ))
                img_paths.extend(sorted(paths))
                
                if c == self.normal_class[0]:
                    labels.extend(len(paths) * [0])
                    gt_paths.extend(len(paths) * [None])
                else:
                    for i,abclass in enumerate(self.abnormal_class):
                        if c == abclass:
                            labels.extend(len(paths) * [i+1])
                    gt_paths.extend(sorted(glob.glob(os.path.join(self.gt_dir, c, '*.jpg'))))
                    

            self.img_paths = img_paths
            self.labels = labels
            self.gt_paths = gt_paths
            # print(gt_paths)
        
        assert len(self.img_paths) == len(self.labels), 'number of x and y should be same'


    def __getitem__(self, index):
        """
        :return:
            original:    original image
            input:  input data to the model
            label:  original image + noise
            mask:   blind spot index
            img_size:train=test=mask=224
        """
        img_path, target, mask_path = self.img_paths[index], self.labels[index], self.gt_paths[index]

       
        # 画像の次元はc,h,wで色は3cなのでそれに合わせて変換。またpltではなくPILで保存しないと型が合わないため注意
        img = Image.open(img_path).convert('RGB')
        img.save("./test.png")
        

        #ここで画像のチャンネル数が３になっているかは不明。要チェック
        #print(img.shape) -> (1,224,224) これではだめ！！
        # img = np.reshape(img,(3,224,224))
        # img = plt.imread(img_path) 
        
        if target == 0:
            mask = torch.zeros([1, self.output_size, self.output_size])
        elif target != 0:
            mask = Image.open(mask_path)
            mask = self.transform_mask(mask)

        if self.transform_img:
            img = self.transform_img(img)
        # print(f'mask_size={mask.shape}_img_size={img.shape}')
        return img, target, mask

    def __len__(self):
        return len(self.img_paths)


def get_rivet_dataset(category, train_transform=None, test_transform=None):
    train_dataset = Rivet(category=category, train=True, transform=train_transform)
    test_dataset = Rivet(category=category, train=False, transform=test_transform)
    return train_dataset, test_dataset


def get_rivet_loader(category, train_transform=None, test_transform=None, batch_size=16):
    train_dataset, test_dataset = get_rivet_dataset(category, train_transform, test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader

# def round_mask(img):
#     # 型を直しておく
#     print(type(img))
#     img = np.array(img)
#     print(type(img))

#     # マスク用単一色画像を作成
#     height = 224 # 生成画像の高さ
#     width = 224 # 生成画像の幅
#     r = 45 #半径
#     mask = np.full(img.shape[:2], 255, dtype=img.dtype)
#     # print(mask)
#     cv2.circle(mask,center=(height/2,width/2),radius=r,color=0,thickness=-1)
#     img[mask==0] = [0,0,0]
#     #マスク画像の保存
#     cv2.imwrite('./mask.png',mask)
#     cv2.imwrite('./sample.png',np.array(img))
#     return img





if __name__ == '__main__':
    # self.transform_img = T.Compose([T.Resize(128, Image.ANTIALIAS),
    #                                     T.CenterCrop(112),
    #                                     T.ToTensor(),
    #                                     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                 std=[0.229, 0.224, 0.225])])
    print('setup loader')
    train_loader, test_loader = get_rivet_loader('Rivet_scr')

    for batch in train_loader:
        print('train')
        print(batch[0].size(), batch[1].size(), batch[2].size())
        break
    
    for batch in test_loader:
        print('test')
        print(batch[0].size(), batch[1].size(), batch[2].size())
        break