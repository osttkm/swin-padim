import random
import math
from random import sample
import argparse
import statistics
import numpy as np
import numba as nb
import os
import csv
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from concurrent import futures

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset_code.rivet import get_rivet_dataset, get_rivet_loader
# from efficientnet_pytorch import EfficientNet

## eff model
from src.dataset_code.efficient_modified import EfficientNetModified

# @nb.jit('void(f4[:,:], f4[:,:], i8, f4[:,:,:])', nopython=True, nogil=True)
# def get_cov(embedding, I, i, cov):
    # _cov = np.empty(I.shape, dtype=np.float32)
    # _cov = np.cov(embedding, rowvar=False) + 0.01 * I
    # for j in range(cov.shape[0]):
    #     for k in range(cov.shape[1]):
    #         cov[j, k, i] = _cov[j, k]
    # return _cov

# @nb.jit('f4(f4[:], f4[:], f4[:,:])', nopython=True, nogil=True)
# def mahalanobis(u, v, cov_inv):
#     delta = u - v
#     m = np.dot(delta, np.dot(cov_inv, delta))
#     return np.sqrt(m)


class PaDiM():
    def __init__(self, config):
        super().__init__()

        print(config)
        # device setup
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.data_category = config.data_category
        self.data_path = config.data_path
        # self.save_path = config.save_path
        self.use_layers = [int(layer) for layer in config.use_layers.split('-')]
        self.arch = config.arch
        self.Rd = config.Rd
        self.use_Rd = not config.non_Rd
        self.seed = config.seed

        self.model, self.idx = self.get_model(self.arch)
        
        self.save_path = os.path.join(config.save_path, self.arch+'_layer'+config.use_layers, self.data_category)
        self.dump_path = os.path.join(self.save_path, 'dump')
        self.test_output_path = os.path.join(self.save_path, 'test')
        self.vis_path = os.path.join(self.save_path, 'visualize')
        self.miss_anormaly = os.path.join(self.save_path, 'missed')

        self.gt_list = []
        
        self.makedirs(self.save_path)
        self.makedirs(self.dump_path)
        self.makedirs(self.test_output_path)
        self.makedirs(self.vis_path)
        self.makedirs(self.miss_anormaly)

    def makedirs(self, path):
        os.makedirs(path, exist_ok=True)

    def get_model(self, arch):
        # ここのchannelsってなに？ = resnet,wide-resnetの１，２，３、４層目のパラメータ数のこと。
        # 論文では３層目までしか行っていなかったが実験的に４層目も追加して確認してみただけらしい
        channels_b7 = [48, 80, 224,384]
        # channels_b7 = [48,640]
        # channels2 = [40, 128, 176]
        channels2 = [40, 128, 176, 304]
        channels3 = [32, 56, 160]
        channels4 = [24, 40, 112]
        # test_channel = [32,640]
        # channels_b6 = [32,576]
        channels_b6 = [576]
        test_channel = [48,80,224,384,640] #Eff7 11-18-38-51-55
        # test_channel = [32,56,160,304,512] #Eff6? 11-18-38-51-55

        if self.arch == 'EfficientNet-B7':
            model = EfficientNetModified.from_pretrained('efficientnet-b7')
            t_d = 0
            for i in self.use_layers:
                t_d += channels_b7[i-1]
                # t_d += test_channel[i-1]
        
        if self.arch == 'EfficientNet-B6':
            model = EfficientNetModified.from_pretrained('efficientnet-b6')
            t_d = 0
            for i in self.use_layers:
                t_d += channels_b6[i-1]
                # t_d += test_channel[i-1]

        if self.arch == 'EfficientNet-B5':
            model = EfficientNetModified.from_pretrained('efficientnet-b5')
            t_d = 0
            for i in self.use_layers:
                # t_d += test_channel[i-1]
                t_d += channels2[i-1]
        if self.arch == 'EfficientNet-B4':
            model = EfficientNetModified.from_pretrained('efficientnet-b4')
            t_d = 0
            for i in self.use_layers:
                t_d += channels3[i-1]
        if self.arch == 'EfficientNet-B1':
            model = EfficientNet.from_pretrained('efficientnet-b1')
            t_d = 0
            for i in self.use_layers:
                t_d += channels4[i-1]
        if self.use_Rd:
            idx = self.get_reduce_dimension_id(t_d)
            # print(f'idx={idx}___idx_shape={idx.shape}')
        else:
            idx = torch.tensor(range(0, t_d))
        model.eval()
        
        return model.to(self.device), idx
    
    # PaDiMの大事なところ。次元削除！
    def get_reduce_dimension_id(self, t_d):
        if self.Rd < t_d:
            idx = torch.tensor(sample(range(0, t_d), self.Rd))
        else:
            idx = torch.tensor(sample(range(0, t_d), t_d))
        return idx
    
    def npca(self, matrix):
        var_list=np.array([])
        normalized_matrix=np.array([])
        sum=0
        for i in range(0,matrix.shape[1]):
            sum += matrix[0][i].flatten().sum()
        # print(f'matrix_shape:{matrix.shape}')
        size = matrix.shape[1] * matrix.shape[2] * matrix.shape[3]
        ave = sum / size
        # print(f'sum={sum}_size={size}_all_mean={ave}')
        var = 0
        for i in range(0,matrix.shape[1]):
            var += ((matrix[0][i] - ave)**2).sum()
        # print(matrix[0][0].to('cpu').detach().numpy().copy().flatten())
        var = var/size

        # print(f'matrix:{matrix}')
        for i in range(0,matrix.shape[1]):
            normalized_matrix = (matrix[0][i] - ave) / (var**1/2)
            # print(normalized_matrix)
            sum = normalized_matrix.sum() 
            mean = normalized_matrix.mean()
            # print(f'sum={sum}_size={matrix.shape[2]*matrix.shape[3]}_mean={mean}')
            var_list = np.append(var_list, (((normalized_matrix-mean)**2) / matrix.shape[2]* matrix.shape[2]).sum())
        # print(np.sort(np.argsort(var_list)))
        # print(np.sort(var_list))
        # print(np.delete(np.sort(np.argsort(var_list)),np.s_[550:]))
        """一定以上分散が大きいような特徴を特定できた。後は埋め込みベクトルを圧縮する際にランダムではなくここで選択したindexの中身を消せばオッケー。寄与率の大きい特徴マップ削除"""
        # print(np.where(var_list > np.sort(var_list)[self.Rd-1]))
        # print(f'level;{np.sort(var_list)[self.Rd-1]}')
        # print(f'var51::{var_list[51]}__var55::{var_list[55]}')
        
        return np.where(var_list > np.sort(var_list)[self.Rd-1])
        
    
    # 特徴量を抽出する。
    def feature_extraction(self, model, loader, idx=None):
        outputs = []
        # これはただ特徴量を埋め込みベクトルとして格納していっているだけ
        def hook(module, input, output):
            outputs.append(output)
        
        if(self.arch == 'EfficientNet-B7'):
            # block_num = torch.tensor([11, 55])
            block_num = torch.tensor([11, 18, 38,51])
            # block_num = torch.tensor([11,18,39,51,55]) #Eff7
            # block_num = torch.tensor([51,55])
            # block_num = torch.tensor([55])
        elif(self.arch == 'EfficientNet-B6'):
            # block_num = torch.tensor([4, 45])
            block_num = torch.tensor([45])
        elif(self.arch == 'EfficientNet-B5'):
            # block_num = torch.tensor([7, 20, 26])
            # block_num = torch.tensor([7,20,26,34,39])
            block_num = torch.tensor([7,20,26,34])
        elif(self.arch == 'EfficientNet-B4'):
            block_num = torch.tensor([3, 7, 17])
        elif(self.arch == 'EfficientNet-B1'):
            block_num = torch.tensor([5, 8, 16])

        
        # 得られた特徴量を層ごとに別々で格納するための配列の準備
        feature_outputs = [(f'layer{i}', []) for i in self.use_layers]
        # 配列をレイヤー番号で検索できるように辞書型に直している
        feature_outputs = OrderedDict(feature_outputs)
        gt_list = []
        gt_mask_list = []
        imgs = []

        for (x, y, mask) in tqdm(loader, '| feature extraction | %s |' % self.data_category):
            imgs.extend(x.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                feats = model.extract_features(x.float().to(self.device), block_num.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(feature_outputs.keys(), feats):
                feature_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in feature_outputs.items():
            feature_outputs[k] = torch.cat(v, 0) # .mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        
        
        layer_names = ['layer'+str(num) for num in self.use_layers]
        embedding_vectors = feature_outputs[layer_names[0]]

        if len(layer_names) != 1:
            for layer_name in layer_names[1:]:
                embedding_vectors = self.embedding_concat(embedding_vectors, feature_outputs[layer_name])
        
        # print(f'embedding_vectors:{embedding_vectors.shape}')
        # delete_list = []
        # if(embedding_vectors.shape[1] > self.Rd): 
        #     delete_list = np.array(self.npca(embedding_vectors))
        #     print(f'delete_list;{delete_list}__num:{len(delete_list[0])}')


        #ここで次元を削減している。576->550
        #idxはすでにランダムに削除された次元の配列
        # print(f'embadding_vector:{embedding_vectors.shape}')
        # if(embedding_vectors.shape[1] > self.Rd):
        #     """特徴量が使用次元数を超える場合に、分散を基準として次元を下げる"""
        #     print(type(delete_list[0]))
        #     embedding_vectors = np.delete(embedding_vectors, random.sample(delete_list[0].tolist(),10),axis=1)
        #     embedding = torch.tensor(embedding_vectors)
        #     print(f'embedding_vectors:{embedding_vectors.shape}')
        #     print(f'embedding:{embedding.shape}')
        # else:
        #     """この部分が元のやつ""" 
        #     embedding = torch.index_select(embedding_vectors, 1, idx)
        embedding = torch.index_select(embedding_vectors, 1, idx)

        self.gt_list = np.array(gt_list)
        
       
        return embedding.numpy(), np.array(imgs), np.array(gt_list), np.array(gt_mask_list)
    

    # ベクトル化
    def embedding_flatten(self, embedding):
        B, C, H, W = embedding.shape
        embedding = embedding.reshape(B, C, H*W)
        return embedding
    

    def get_embedding_mean(self, embedding):
        return np.mean(embedding, axis=0)

    
    def get_embedding_cov(self, embedding):
        if embedding.ndim == 4:
            B, C, H, W = embedding.shape
            HW = H*W
            embedding = self.embedding_flatten(embedding)
            cov = torch.zeros(C, C, H * W).numpy()
        else:
            B, C, HW = embedding.shape
            cov = torch.zeros(C, C, HW).numpy()
        I = np.identity(C, dtype=np.float32)

        # for i in range(HW):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            # cov[:, :, i] = np.cov(embedding[:, :, i], rowvar=False) + 0.01 * I
        
        @nb.jit('f8[:,:](f4[:,:], f4[:,:])', nopython=True, nogil=True)
        def get_cov(embedding, I):
            _cov = np.cov(embedding, rowvar=False) + 0.01 * I
            return _cov
        
        future_list = []
        with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i in range(HW):
                # f = executor.submit(get_cov, embedding[:,:,i], I, i, cov)
                f = executor.submit(get_cov, embedding[:,:,i], I)
                future_list.append(f)
            _ = futures.as_completed(fs=future_list)
        cov = np.array([f.result() for f in future_list]).transpose(1,2,0)
        
        # inv_cov = LedoitWolf().fit(embedding).precision_  # 推定共分散逆行列
        # cov = LedoitWolf().fit(embedding).covariance_  # 推定共分散
        
        return cov

    # ピクセル単位で次元ごとのガウス分布を取得する
    def get_multivariate_gaussian_distribution(self, loader):
        model, idx = self.model, self.idx
        embedding, _, _, _ = self.feature_extraction(model, loader, idx)

        mean = self.get_embedding_mean(self.embedding_flatten(embedding))
        cov = self.get_embedding_cov(embedding).astype(mean.dtype)
        train_outputs = [mean, cov]
        return train_outputs


    def get_distance_matrix(self, train_loader, test_loader):
        train_outputs = self.get_multivariate_gaussian_distribution(train_loader)

        model, idx = self.model, self.idx
        test_embedding, imgs, targets, gt_masks = self.feature_extraction(model, test_loader, idx)
        B, C, H, W = test_embedding.shape
        test_embedding = self.embedding_flatten(test_embedding) # (B, C, H*W)
        # dist_list = []
        # for i in range(H * W):
        #     mean = train_outputs[0][:, i]
        #     conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        #     dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in test_embedding]
        #     # dist = [(sample[:, i] - mean).dot(conv_inv).dot((sample[:, i] - mean).T) for sample in test_embedding]
        #     dist_list.append(dist) # (B, HW)
        dist_list = np.empty((B, H*W), dtype=np.float32)

        # @nb.jit('void(f4[:, :, :], f4[:, :], f4[:, :, :], f4[:, :])', nopython=True, parallel=True)
        def cal_mahalanobis(test_embedding, mean_list, cov_list, dist_list):
            for hw_idx in nb.prange(dist_list.shape[1]):
                mean = mean_list[:, hw_idx] # (C, HW) -> (C,)
                # cov_inv = np.ones((cov_list.shape[0],cov_list.shape[1]), dtype=np.float32)
                cov_inv = np.linalg.inv(cov_list[:, :, hw_idx]) # (C, C, HW) -> (C, C)

                for sample_idx in nb.prange(dist_list.shape[0]):
                    sample = test_embedding[sample_idx, :, hw_idx]
                    delta = sample - mean
                    m = np.dot(delta, np.dot(cov_inv, delta))
                    m = np.sqrt(m)
                    dist_list[sample_idx, hw_idx] = m
        cal_mahalanobis(test_embedding, train_outputs[0], train_outputs[1], dist_list)
        # dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        dist_list = dist_list.reshape(B, H, W)
        dist_list = torch.tensor(dist_list)
        dist_matrix = F.interpolate(dist_list.unsqueeze(1), size=imgs[0].shape[1], mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        return dist_matrix, imgs, targets, gt_masks
    
    # 異常マップの獲得
    def get_score_map(self, dist_matrix):
        # apply gaussian smoothing on the score map
        print(f'dist_max:{dist_matrix.max()}\ndist_min:{dist_matrix.min()}')
        for i in range(dist_matrix.shape[0]):
            dist_matrix[i] = gaussian_filter(dist_matrix[i], sigma=4)

        # Normalization
        max_score = dist_matrix.max()
        min_score = dist_matrix.min()
        scores = (dist_matrix - min_score) / (max_score - min_score)
        return scores

    # 画像の異常特定
    def cal_image_level_roc(self, scores, targets):
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.uint8(targets != 0)
        fpr, tpr, th = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image ROCAUC: %.3f' % (img_roc_auc))

        df = pd.DataFrame({'fpr': fpr,
                           'tpr': tpr,
                           'th': th})
        df.to_csv(os.path.join(self.test_output_path, 'img_tpr_fpr.csv'), index=None)
        df = pd.DataFrame({'anomaly_score': img_scores,
                           'anomaly_label': gt_list,
                           'targets': targets})
        df.to_csv(os.path.join(self.test_output_path, 'img_anomaly_score.csv'), index=None)

        fig, ax = plt.subplots()
        ax.set_xlabel('FPR: False positive rate')
        ax.set_ylabel('TPR: True positive rate')
        ax.set_title('Image-level ROC Curve (area = {:.4f})'.format(img_roc_auc))
        ax.grid()
        ax.plot(fpr, tpr)  # , marker='o')
        plt.savefig(os.path.join(self.save_path, 'image-level-roc-curve.png'), transparent=False)
        # plt.clf()
        plt.close()

        return img_roc_auc

 
    def plot_histgram_anomaly_scores(self, scores, targets):
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.uint8(targets != 0)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.set_xlabel('Anomaly score')
        ax_hist.set_ylabel('Number of Instances')
        ax_hist.set_title('Histogram of anomaly scores')
        normal_scores = img_scores[gt_list == 0]
        abnormal_scores = img_scores[gt_list == 1]
        ax_hist.hist([normal_scores, abnormal_scores], 50, label=['Normal samples', 'Abnormal samples'],
                alpha=0.5, histtype='stepfilled')
        ax_hist.legend()
        ax_hist.grid(which='major', axis='y', color='grey',
                alpha=0.8, linestyle="--", linewidth=1)
        fig_hist.savefig(os.path.join(self.save_path, 'histgram.png'), transparent=False)
        plt.close()
    
    # ピクセル単位での異常の位置特定
    def cal_pixel_level_roc(self, scores, gt_masks):

        """下記の一文について、gt_masks.flatten()ではエラーを吐いた。以下エラー
        File "/home/oshita/padim/padim_main.py", line 26, in <module>
        padim.test()
        File "/home/oshita/padim/src/modules/padim_high.py", line 437, in test
        self.evaluate(train_dataloader, test_dataloader)
        File "/home/oshita/padim/src/modules/padim_high.py", line 413, in evaluate
        pixel_auc, threshold = self.cal_pixel_level_roc(scores, gt_masks)
        File "/home/oshita/padim/src/modules/padim_high.py", line 331, in cal_pixel_level_roc
        fpr, tpr, th = roc_curve(gt_masks.flatten(), scores.flatten())
        File "/opt/miniconda3/envs/develop/lib/python3.8/site-packages/sklearn/metrics/_ranking.py", line 962, in roc_curve
        fps, tps, thresholds = _binary_clf_curve(
        File "/opt/miniconda3/envs/develop/lib/python3.8/site-packages/sklearn/metrics/_ranking.py", line 731, in _binary_clf_curve
        raise ValueError("{0} format is not supported".format(y_type))
    ValueError: continuous format is not supported

結局のところ、変数がfloat型になっていたことが原因らしい。データセットを変えると怒られたがなぜなのかはわからない。"""
        precision, recall, thresholds = precision_recall_curve(gt_masks.astype(int).flatten(),scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, th = roc_curve(gt_masks.astype(int).flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_masks.astype(int).flatten(), scores.flatten())
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig, ax = plt.subplots()
        ax.set_xlabel('FPR: False positive rate')
        ax.set_ylabel('TPR: True positive rate')
        ax.set_title('Pixel-level ROC Curve (area = {:.4f})'.format(per_pixel_rocauc))
        ax.grid()
        ax.plot(fpr, tpr)  # , marker='o')
        plt.savefig(os.path.join(self.save_path, 'pixel-level-roc-curve.png'), transparent=False)
        # plt.clf()
        plt.close()
        return per_pixel_rocauc, threshold

    
    def plot_fig(self, test_img, scores, gts, threshold, data_category):
        num = len(scores)
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.
        
        def _plot(test_img, scores, gts, threshold, data_category, i):
        # for i in range(num):
            img = test_img[i]
            img = self.denormalization(img)
            gt = gts[i].transpose(1, 2, 0).squeeze()
            heat_map = scores[i] * 255
            mask = scores[i]
            mask[mask > threshold] = 1
            mask[mask <= threshold] = 0

            # sum=mask.sum()
            # print(f'img{i}__mask[{i}]:{sum}__gt_list[{i}]:{self.gt_list[i]}')

            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
            fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
            fig_img.subplots_adjust(right=0.9)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image' + data_category + '_{}'.format(i))
            ax_img[1].imshow(gt, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            #下記の一文をコメントアウトしていたら出力がおかしかった
            ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
            ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[2].imshow(img, cmap='gray', interpolation='none')
            ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', vmax=vmax, vmin=vmin)
            ax_img[2].title.set_text('Predicted heat map')
            ax_img[3].imshow(mask, cmap='gray')
            ax_img[3].title.set_text('Predicted mask')
            ax_img[4].imshow(vis_img)
            ax_img[4].title.set_text('Segmentation result')
            # plt.close(ax_img)

            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)
            fig_img.savefig(os.path.join(self.test_output_path, data_category + '_{}'.format(i)), dpi=100)
            # plt.clf() #これはいるんかわからん
            plt.close()

        with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i in range(num):
                executor.submit(_plot, test_img, scores, gts, threshold, data_category, i)

    
    def evaluate(self, train_loader, test_loader):
        dist_matrix, imgs, targets, gt_masks = self.get_distance_matrix(train_loader, test_loader)
        scores = self.get_score_map(dist_matrix)
        image_auc = self.cal_image_level_roc(scores, targets)
        pixel_auc, threshold = self.cal_pixel_level_roc(scores, gt_masks)

        self.plot_histgram_anomaly_scores(scores, targets)
        self.plot_fig(imgs, scores, gt_masks, threshold, self.data_category)
        self.score_write_csv(image_auc, pixel_auc)
    
    def auc_evaluate(self, train_loader, test_loader):
        dist_matrix, imgs, targets, gt_masks = self.get_distance_matrix(train_loader, test_loader)
        scores = self.get_score_map(dist_matrix)
        image_auc = self.cal_image_level_roc(scores, targets)
        return image_auc

    
    def img_level_evaluate(self, train_loader, test_loader):
        dist_matrix, imgs, targets, gt_masks = self.get_distance_matrix(train_loader, test_loader)
        scores = self.get_score_map(dist_matrix)
        image_auc = self.cal_image_level_roc(scores, targets)
        # pixel_auc, threshold = self.cal_pixel_level_roc(scores, gt_masks)
        self.plot_histgram_anomaly_scores(scores, targets)
        # self.plot_fig(imgs, scores, gt_masks, threshold, self.data_category)
        self.score_write_csv(image_auc, 0)

    def test(self):
        train_dataloader, test_dataloader = self.get_loader()
        self.evaluate(train_dataloader, test_dataloader)
    
    def auc_test(self):
        train_dataloader, test_dataloader = self.get_loader()
        auroc = self.auc_evaluate(train_dataloader, test_dataloader)
        return auroc

    def embedding_concat(self, x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z
    
    def score_write_csv(self, image_auc, pixel_auc):
        try:
            if not os.path.exists(os.path.join(self.save_path, '..', 'result.csv')):
                file = open(os.path.join(self.save_path, '..', 'result.csv'), 'w')
                writer = csv.writer(file)
                writer.writerow(['data_class', 'model', 'layers', 'Rd', 'use_Rd','image_ROC', 'pixel_ROC', 'seed'])
            else:
                file = open(os.path.join(self.save_path, '..', 'result.csv'), 'a')
                writer = csv.writer(file)
            writer.writerow([self.data_category, self.arch, '_'.join(map(str, self.use_layers)), self.Rd, self.use_Rd, image_auc, pixel_auc, self.seed])
        finally:
            file.close()
    
    def denormalization(self, x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
        return x

    """自前のデータセットを用いる 
    -> rivet.py の方から持ってきたほうがいい？あれは別物としてこっちで宣言しなおす？
    -> そもそもrivet.pyがデータセットを作るものであるならここに宣言しないといけない、
    ないしは適当に小さなクラスfileを別途作成する必要がある

    ==rivet.pyではあれ単体でデータローダーとかデータセットの設定がしっかりとできているのかを調べられるようになっている。
    ==ここで、rivet.pyの関数はしっかり呼び出すことができている。srcの中のファイルだから？しっかり調べておくこと。
    """
    def get_loader(self, train_transforms=None, test_transforms=None):
        train_loader, test_loader = get_rivet_loader(category=self.data_category, train_transform=train_transforms, test_transform=test_transforms, batch_size=1)
        return train_loader, test_loader
    
    def get_dataset(self, train_transforms=None, test_transforms=None):
        train_dataset, test_dataset = get_rivet_dataset(category=self.data_category, train_transforms=train_transforms, test_transforms=test_transforms)
        return train_dataset, test_dataset




