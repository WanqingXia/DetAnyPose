import os
import random

from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance

from utils.dataloader import create_dataloader
from utils.process_data import create_folder
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
    dataroot = '/home/wanqing/YCB_Video_Dataset'
    save_root = Path('./results')
    epochs = 100
    embedding_dimension = 512
    batch_size = 16
    num_workers = 4
    learning_rate = 1e-3
    margin = 0.2
    image_size = 256
    start_epoch = 0
    seed = 42
    gamma = 0.7

    # Define seed for whole training
    seed_everything(seed)

    # Format the current date and time to be accurate to minutes, excluding the year
    formatted_now = datetime.now().strftime("%m-%d_%H:%M")
    save_path = create_folder(save_root / formatted_now)

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

    train_loader, train_set = create_dataloader(dataroot,
                                                type='train',
                                                imgsz=image_size,
                                                batch_size=batch_size,
                                                workers=num_workers)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        for batch_train, train_data in tqdm(enumerate(train_loader), total=min(1000, len(train_loader)),
                                                 desc=f'Training, iteration {epoch+1} out of {epochs}'):
            l2_distance = PairwiseDistance(p=2)
            concatenated_data = torch.cat((train_data[0], train_data[1], train_data[2]), dim=0)

            anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                imgs=concatenated_data,
                model=model,
                batch_size=batch_size
            )
            # Hard Negative triplet selection
            #  (negative_distance - positive_distance < margin)
            #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
            # pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            # neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)
            # all_pos = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
            # valid_triplets = np.where(all_pos == 1)
            #
            # anc_valid_embeddings = anc_embeddings[valid_triplets]
            # pos_valid_embeddings = pos_embeddings[valid_triplets]
            # neg_valid_embeddings = neg_embeddings[valid_triplets]

            triplet_loss = criterion.forward(
                anchor=anc_embeddings,
                pos=pos_embeddings,
                neg=neg_embeddings
            )

            optimizer.zero_grad()
            triplet_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += triplet_loss / len(train_loader)

            if batch_train >= 999:
                break
        print('The Triplet loss for Train epoch {} is {}.'.format(epoch+1, epoch_loss))

        with torch.no_grad():
            epoch_val_loss = 0
            val_iteration_limit = 100
            for batch_val, val_data in tqdm(enumerate(train_loader), total=val_iteration_limit,
                                                 desc=f'Validating, iteration {epoch+1} out of {epochs}'):
                concatenated_data = torch.cat((val_data[0], val_data[1], val_data[2]), dim=0)

                anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                    imgs=concatenated_data,
                    model=model,
                    batch_size=batch_size
                )

                val_loss = criterion.forward(
                    anchor=anc_embeddings,
                    pos=pos_embeddings,
                    neg=neg_embeddings
                )

                epoch_val_loss += val_loss / val_iteration_limit
                if batch_val == val_iteration_limit - 1:
                    break
            print('The Triplet loss for Validation epoch {} is {}.'.format(epoch+1, epoch_val_loss))

        # Save model checkpoint
        state = {
            'epoch': epoch,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_model_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss.item(),
            'val_loss': epoch_val_loss.item()
        }
        if gpu_flag:
            state['model_state_dict'] = model.module.state_dict()

        # Save model checkpoint
        torch.save(state, save_path / 'triplet_epoch_{}.pt'.format(epoch))


if __name__ == '__main__':
    train()

