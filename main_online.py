# Copyright (c) Alibaba Group
import argparse
import torch
import torchvision.datasets as datasets
import torch.nn.functional as F
import os
import math
import numpy as np
import random
import logging
import faiss
from gmpy2 import random_state
from sympy.abc import alpha
from tqdm import tqdm
from sklearn.cluster import KMeans

from utils.detection_util import print_measures, get_and_print_results
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution

import scipy.optimize as sopt

parser = argparse.ArgumentParser(description='OOD scoring for ImageNet')

parser.add_argument('--seed', default=2, type=int, help="random seed") #2

# hyper-parameters
parser.add_argument('--temp', default=0.009, type=float) # 0.01 0.012 (R) 0.01 (A) 0.009 (V2) 0.01 (S) 0.01 (B32) 0.01 (L14) 0.01 (L14336) 0.01 (Res50) 0.009 (common)

parser.add_argument('--temp2', default=0.01, type=float) # KL 12

parser.add_argument('--T', default=0.0, type=float)

parser.add_argument('--ngroups', default=1, type=int)

parser.add_argument('--eta', default=1.45, type=float) #1.2 1.14 (R) 1.18 (A) 1.45 (V2) 0.8 (S) 1.2 (B32) 1.3 (L14) 1.2 (L14336) 1.3 (Res50) 1.15 (common)

parser.add_argument('--r', default=1.05,type=float) # 1.05 1.05 (R) 1.05 (A) 1.05 (V2) 1.05 (S) 1.05 (B32) 1.06 (L14) 1.06 (L14336) 1.06 (Res50) 1.07 (common)

parser.add_argument('--beta', default=-0.08, type=float) #-0.1 0.0 (R) 0.0 (A) -0.08 (v2) 0.06 (S) -0.12 (B32) -0.1 (L14) -0.12 (L14336) -0.1 (Res50) -0.05 (common)

parser.add_argument('--epochs', default=5, type=int) #15 5 (R) 5 (A) 5 (V2) 20 (S) 15 (B32) 15 (L14) 0 (L14336) 30 (Res50) 15 (common)

parser.add_argument('--lr', default=1e-2, type=float) #1e-2 1e-2 (R) 1e-3 (A) 1e-2 (V2) 1e-3 (S) 1e-2 (B32) 1e-3 (L14) 1e-3 (Res50) 1e-2 (common)

# end

torch.cuda.set_device(0)

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)
relu = torch.nn.ReLU()
# 'ImageNet', 'ImageNet-R', 'ImageNet-A', 'ImageNet-V2', 'ImageNet-S'
in_dataset = 'ImageNet-V2'
out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']


def setup_log(args):
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(".\ood_eval_info.log", mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.debug(f"#########eval_ood############")
    return log

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def sph_inter(a,b,s):
    theta = torch.acos( (a*b).sum(dim=[1] )).view(a.shape[0],1)
    n1 = torch.sin(s*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-s)*theta)/torch.sin(theta)*b
    return n1+n2

def get_score(image_features, text_features_pos, text_features_neg, args):

    ngroup = 100
    temp = 0.01

    score = 0

    text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)
    text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

    drop = text_features_neg.shape[0] % ngroup

    if ngroup>1:
        random_permute = True
    else:
        random_permute = False

    if drop > 0:
        text_features_neg = text_features_neg[:-drop,:]

    if random_permute:
        idx = torch.randperm(text_features_neg.size(0)).cuda()
        text_features_neg = text_features_neg[idx]

    text_features_neg = torch.reshape(text_features_neg, (ngroup, -1, text_features_neg.size(1)))

    sim_pos = image_features @ text_features_pos.T

    for j in range(ngroup):

        text_features_neg_j = text_features_neg[j]

        sim_neg_j = image_features @ text_features_neg_j.T

        output = torch.cat((sim_pos, sim_neg_j), dim=1)

        output = torch.softmax(output/temp, dim=1)


    #     sim_pos_neg_j = torch.cat((sim_pos, sim_neg_j), dim=1)
    #
    #     beta  = torch.full(
    #         size = (image_features.size(0),1),
    #         fill_value = args.beta,
    #         dtype = torch.float32,
    #         device=torch.device("cuda")
    #     )
    #
    #     for epoch in range(args.epochs):
    #         diff = relu(sim_pos_neg_j - beta)/ temp
    #         g = diff.pow(r_star).mean(dim=1, keepdim=True)
    #         term = diff.pow(r_star - 1).mean(dim=1, keepdim=True)
    #         grad = 1 - args.eta * (g ** (1.0 / r_star - 1.0)) * term
    #         beta = beta - args.lr * grad
    #
    #     output_pos = torch.exp((sim_pos - beta) / temp)
    #
    #     output_pos_neg_j = relu((sim_pos_neg_j - beta) / temp).pow(r_star).mean(dim=1, keepdim=True).pow(1.0 / r_star)
    #
    #     Z_j = torch.exp(output_pos_neg_j*args.eta)
    #
    #     output = output_pos / Z_j

        score_j = output[:, 0:text_features_pos.shape[0]]

        score = score + score_j / ngroup

    score = to_np(score)

    score = np.mean(score, axis=1)

    return score

def get_score_2(image_features, text_features_pos, text_features_neg, args):

    ngroup = 1
    temp = args.temp
    r_star = args.r/(args.r-1.0)

    text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)
    text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

    score = []

    sim_pos = image_features @ text_features_pos.T

    drop = text_features_neg.shape[0] % ngroup

    if ngroup>1:
        random_permute = True
    else:
        random_permute = False

    if drop > 0:
        text_features_neg = text_features_neg[:-drop,:]

    if random_permute:
        idx = torch.randperm(text_features_neg.size(0)).cuda()
        text_features_neg = text_features_neg[idx]

    text_features_neg = torch.reshape(text_features_neg, (ngroup, -1, text_features_neg.size(1)))

    for j in range(ngroup):

        if ngroup>1:
            index_k = torch.tensor([i for i in range(text_features_neg.shape[0])])[args.I == j]
            text_features_neg_j = torch.index_select(text_features_neg, 0, index_k.cuda())
        else:
            text_features_neg_j = text_features_neg


        beta  = torch.full(
            size = (image_features.size(0),1),
            fill_value = args.beta,
            dtype = torch.float32,
            device=torch.device("cuda")
        )

        # text_features_neg_j = text_features_neg[j]
        #
        # sim_neg_j = image_features @ text_features_neg_j.T
        #
        # # sim_pos_neg_j = sim_neg_j
        # sim_pos_neg_j = torch.cat((sim_pos, sim_neg_j), dim=1)

        text_features_neg_j = text_features_neg[j]

        sim_neg_j = image_features @ text_features_neg_j.T

        # sim_pos_neg_j = sim_neg_j
        sim_pos_neg_j = torch.cat((sim_pos, sim_neg_j), dim=1)

        for epoch in range(args.epochs):
            diff = relu(sim_pos_neg_j - beta)/ temp
            g = diff.pow(r_star).mean(dim=1, keepdim=True)
            term = diff.pow(r_star - 1).mean(dim=1, keepdim=True)
            grad = 1 - args.eta * (g ** (1.0 / r_star - 1.0)) * term
            beta = beta - args.lr * grad

        output_pos = torch.exp((sim_pos - beta) / temp)

        output_pos_neg_j = relu((sim_pos_neg_j - beta)/ temp ).pow(r_star).mean(dim=1, keepdim=True).pow(1.0 / r_star)

        Z_j = torch.exp(output_pos_neg_j*args.eta)

        output = output_pos/Z_j

        score_j = output[:, 0:text_features_pos.shape[0]].sum(dim=1,  keepdim=True)

        score.append(torch.log(score_j))

    score = torch.cat(score, dim=-1)

    score = to_np(score.mean(dim=-1))

    return score

def main():

    args = parser.parse_args()

    setup_seed(args.seed)

    log = setup_log(args)

    # image_feature_dict = torch.load('./CLIP_features_B16.pth')

    # image_feature_dict = torch.load('./CLIP_features_L14.pth')
    # image_feature_dict = torch.load('./CLIP_features_B32.pth')
    # image_feature_dict = torch.load('./CLIP_features_RN50.pth')
    # image_feature_dict = torch.load('./CLIP_features_L14_336.pth')

    # image_feature_dict = torch.load('./CLIP_features_B16_R.pth')
    # image_feature_dict = torch.load('./CLIP_features_B16_A.pth')
    image_feature_dict = torch.load('./CLIP_features_B16_V2.pth')
    # image_feature_dict = torch.load('./CLIP_features_B16_S.pth')

    print(image_feature_dict.keys())

    text_feature_dict = torch.load('./neg_dump_new_B16.pth')
    # text_feature_dict = torch.load('./neg_dump_new_L14.pth')
    # text_feature_dict = torch.load('./neg_dump_new_B32.pth')
    # text_feature_dict = torch.load('./neg_dump_new_RN50.pth')
    # text_feature_dict = torch.load('./neg_dump_new_L14_336.pth')



    print('load pre-trained CLIP text features')
    text_features_pos = text_feature_dict['pos_emb'].cuda()
    id_class_num = text_features_pos.shape[0]

    text_features_neg = text_feature_dict['neg_emb_selected'].cuda()
    ood_class_num = text_features_neg.shape[0]


    print(f'ID_length: {id_class_num}')
    print(f'total_selected_neg_labels: {ood_class_num}')

    auroc_list, aupr_list, fpr_list = [], [], []

    log.debug(f"Evaluting OOD dataset {in_dataset}")

    image_features_in = image_feature_dict[in_dataset].cuda()
    id_sample_num = image_features_in.shape[0]
    print(f'ID_sample_number: {id_sample_num}')

    for out_dataset in out_datasets:

        log.debug(f"Evaluting OOD dataset {out_dataset}")

        out_scores = []

        image_features_out = image_feature_dict[out_dataset].cuda()
        out_sample_num = image_features_out.shape[0]
        print(f'OOD_feature_length: {out_sample_num}')

        # in_scores = get_score(image_features_in, text_features_pos, text_features_neg, args)

        # out_scores = get_score(image_features_out, text_features_pos, text_features_neg, args)

        image_features = torch.cat((image_features_in, image_features_out), dim=0)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # image_features = image_features[-1:]

        # scores = get_score(image_features, text_features_pos, text_features_neg, args)
        scores = get_score_2(image_features, text_features_pos, text_features_neg, args)


        in_scores = scores[0:id_sample_num]
        out_scores = scores[id_sample_num:]

        get_and_print_results(args, log, -in_scores, -out_scores,
                              auroc_list, aupr_list, fpr_list)


    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list))


if __name__ == '__main__':
    main()

