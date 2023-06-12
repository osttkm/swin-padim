import random
from random import sample
import argparse
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
from torchvision.models import wide_resnet50_2, resnet18, resnet50

from src.dataset_code.mvtec import get_mvtec_dataset, get_mvtec_loader

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

        self.model, self.idx = self.get_model(self.arch)
        
        self.save_path = os.path.join(config.save_path, self.arch+'_layer'+config.use_layers, self.data_category)
        self.dump_path = os.path.join(self.save_path, 'dump')
        self.test_output_path = os.path.join(self.save_path, 'test')
        self.vis_path = os.path.join(self.save_path, 'visualize')
        
        self.makedirs(self.save_path)
        self.makedirs(self.dump_path)
        self.makedirs(self.test_output_path)
        self.makedirs(self.vis_path)

    def makedirs(self, path):
        os.makedirs(path, exist_ok=True)

    def get_model(self, arch):
        #モデル毎のチャンネル数
        channels1 = [64, 128, 256, 512]
        channels2 = [256, 512, 1024, 2048]
        channels3 = [38, 80, 224]
        if self.arch == 'resnet18':
            model = resnet18(pretrained=True, progress=True)
            t_d = 0
            for i in self.use_layers:
                t_d += channels1[i-1]
        elif self.arch == 'resnet50':
            model = resnet50(pretrained=True, progress=True)
            t_d = 0
            for i in self.use_layers:
                t_d += channels2[i-1]
        elif self.arch == 'wide_resnet50_2':
            model = wide_resnet50_2(pretrained=True, progress=True)
            t_d = 0
            for i in self.use_layers:
                t_d += channels2[i-1]
        if self.use_Rd:
            idx = self.get_reduce_dimension_id(t_d)
        else:
            idx = torch.tensor(range(0, t_d))
        model.eval()
        
        return model.to(self.device), idx
    

    def get_reduce_dimension_id(self, t_d):
        if self.Rd < t_d:
            idx = torch.tensor(sample(range(0, t_d), self.Rd))
        else:
            idx = torch.tensor(sample(range(0, t_d), t_d))
        
        return idx
    

    def feature_extraction(self, model, loader, idx=None):
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        
        if 1 in self.use_layers:
            model.layer1[-1].register_forward_hook(hook)
        if 2 in self.use_layers:
            model.layer2[-2].register_forward_hook(hook)
        if 3 in self.use_layers:
            model.layer3[-1].register_forward_hook(hook)
        if 4 in self.use_layers:
            model.layer4[-1].register_forward_hook(hook)

        feature_outputs = [(f'layer{i}', []) for i in self.use_layers]
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
                _ = model(x.float().to(self.device))
            # get intermediate layer outputs
            for k, v in zip(feature_outputs.keys(), outputs):
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
        
        embedding = torch.index_select(embedding_vectors, 1, idx)
        # print(f'gt_list={np.shape(gt_list)}')
        # print(f'gt_mask_list={np.shape(gt_mask_list)}')
        # print(f'imgs={np.shape(imgs)}')
        return embedding.numpy(), np.array(imgs), np.array(gt_list), np.array(gt_mask_list)
    

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

        @nb.jit('void(f4[:, :, :], f4[:, :], f4[:, :, :], f4[:, :])', nopython=True, parallel=True)
        def cal_mahalanobis(test_embedding, mean_list, cov_list, dist_list):
            for hw_idx in nb.prange(dist_list.shape[1]):
                mean = mean_list[:, hw_idx] # (C, HW) -> (C,)
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
    

    def get_score_map(self, dist_matrix):
        # apply gaussian smoothing on the score map
        for i in range(dist_matrix.shape[0]):
            dist_matrix[i] = gaussian_filter(dist_matrix[i], sigma=4)

        # Normalization
        max_score = dist_matrix.max()
        min_score = dist_matrix.min()
        scores = (dist_matrix - min_score) / (max_score - min_score)
        return scores

    
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
        plt.savefig(os.path.join(self.save_path, 'image-level-roc-curve.png'), transparent=True)
        plt.clf()
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
        fig_hist.savefig(os.path.join(self.save_path, 'histgram.png'), transparent=True)
    

    def cal_pixel_level_roc(self, scores, gt_masks):
        if gt_masks.flatten().shape == scores.flatten().shape:
            # print(f'gt_masks:{gt_masks.shape}__scores:{scores.shape}')
            print(f'gt_masks:{gt_masks}')
        precision, recall, thresholds = precision_recall_curve(gt_masks.flatten(), scores.flatten())

        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, th = roc_curve(gt_masks.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_masks.flatten(), scores.flatten())
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig, ax = plt.subplots()
        ax.set_xlabel('FPR: False positive rate')
        ax.set_ylabel('TPR: True positive rate')
        ax.set_title('Pixel-level ROC Curve (area = {:.4f})'.format(per_pixel_rocauc))
        ax.grid()
        ax.plot(fpr, tpr)  # , marker='o')
        plt.savefig(os.path.join(self.save_path, 'pixel-level-roc-curve.png'), transparent=True)
        plt.clf()
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
            ax_img[0].title.set_text('Image')
            ax_img[1].imshow(gt, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
            ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[2].imshow(img, cmap='gray', interpolation='none')
            ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', vmax=vmax, vmin=vmin)
            ax_img[2].title.set_text('Predicted heat map')
            ax_img[3].imshow(mask, cmap='gray')
            ax_img[3].title.set_text('Predicted mask')
            ax_img[4].imshow(vis_img)
            ax_img[4].title.set_text('Segmentation result')
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
            plt.close()
        
        with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i in range(num):
                executor.submit(_plot, test_img, scores, gts, threshold, data_category, i)

    
    def evaluate(self, train_loader, test_loader):
        dist_matrix, imgs, targets, gt_masks = self.get_distance_matrix(train_loader, test_loader)
        scores = self.get_score_map(dist_matrix)
        image_auc = self.cal_image_level_roc(scores, targets)
        pixel_auc, threshold = self.cal_pixel_level_roc(scores, gt_masks)
        # print(f'pixel_auc={pixel_auc}_threshold={threshold}')
        
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
                writer.writerow(['data_class', 'model', 'layers', 'Rd', 'use_Rd','image_ROC', 'pixel_ROC'])
            else:
                file = open(os.path.join(self.save_path, '..', 'result.csv'), 'a')
                writer = csv.writer(file)
            writer.writerow([self.data_category, self.arch, '_'.join(map(str, self.use_layers)), self.Rd, self.use_Rd, image_auc, pixel_auc])
        finally:
            file.close()
    
    def denormalization(self, x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
        
        return x

    def get_loader(self, train_transforms=None, test_transforms=None):
        train_loader, test_loader = get_mvtec_loader(category=self.data_category, train_transform=train_transforms, test_transform=test_transforms, batch_size=1)
        return train_loader, test_loader
    
    def get_dataset(self, train_transforms=None, test_transforms=None):
        train_dataset, test_dataset = get_mvtec_dataset(category=self.data_category, train_transforms=train_transforms, test_transforms=test_transforms)

        return train_dataset, test_dataset




