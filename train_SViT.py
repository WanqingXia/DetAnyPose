import glob
from itertools import chain
import os
import random
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance

from utils.dataloader import create_dataloader
from losses.triplet_loss import TripletLoss
from models.simple_vit import SimpleViT


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu


def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size: batch_size * 2]
    neg_embeddings = embeddings[batch_size * 2:]

    return anc_embeddings, pos_embeddings, neg_embeddings, model


def train():
    # Define hyperparameter
    dataroot = '/media/iai-lab/wanqing/YCB_Video_Dataset'
    epochs = 20
    embedding_dimension = 512
    batch_size = 64
    num_workers = 8
    learning_rate = 3e-5
    margin = 0.2
    image_size = 256
    start_epoch = 0
    seed = 42
    gamma = 0.7

    # Define seed for whole training
    seed_everything(seed)

    # Define ViT model
    model = SimpleViT(
        image_size=image_size,
        patch_size=32,
        num_classes=embedding_dimension,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048
    )

    model, gpu_flag = set_model_gpu_mode(model)

    # loss function
    criterion = TripletLoss(margin=margin)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    for epoch in tqdm(range(start_epoch, epochs)):
        epoch_loss = 0

        train_loader = create_dataloader(dataset="train", batch_size=batch_size, shuffle=True)
        valid_loader = create_dataloader(dataset="val", batch_size=batch_size, shuffle=True)

        for data in tqdm(train_loader):
            l2_distance = PairwiseDistance(p=2)

            anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                imgs=data,
                model=model,
                batch_size=batch_size
            )

            # Hard Negative triplet selection
            #  (negative_distance - positive_distance < margin)
            #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
            pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)
            all_pos = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
            valid_triplets = np.where(all_pos == 1)

            anc_valid_embeddings = anc_embeddings[valid_triplets]
            pos_valid_embeddings = pos_embeddings[valid_triplets]
            neg_valid_embeddings = neg_embeddings[valid_triplets]

            triplet_loss = criterion.forward(
                anchor=anc_valid_embeddings,
                pos=pos_valid_embeddings,
                neg=neg_valid_embeddings
            )

            optimizer.zero_grad()
            triplet_loss.backward()
            optimizer.step()

            epoch_loss += triplet_loss / len(train_loader)

        with torch.no_grad():
            epoch_val_loss = 0
            for data in valid_loader:

                anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                    imgs=data,
                    model=model,
                    batch_size=batch_size
                )

                val_loss = criterion.forward(
                    anchor=anc_embeddings,
                    pos=pos_embeddings,
                    neg=neg_embeddings
                )

                epoch_val_loss += val_loss / len(valid_loader)


if __name__ == '__main__':
    train()

