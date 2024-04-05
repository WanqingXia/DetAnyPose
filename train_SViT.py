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
import wandb

from utils.dataloader import create_dataloader
from utils.process_data import create_folder
from losses.triplet_loss import TripletLoss
from models.simple_vit import SimpleViT
from torch.optim.lr_scheduler import ExponentialLR


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
    dataroot = '/home/iai-lab/Documents/YCB_Video_Dataset'
    save_root = Path('./results')
    epochs = 100
    embedding_dimension = 512
    batch_size = 64
    num_workers = 18
    margin = 0.2
    margin_change = 0.02
    hard_negative_threshold = 0.6
    image_size = 256
    start_epoch = 0
    seed = 42
    iter_per_epoch = 1000
    gamma = 0.95

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
        mlp_dim=2048,
        channels=4
    )

    model, gpu_flag = set_model_gpu_mode(model)
    # Define warm-up period
    warmup_epochs = 5

    # loss function
    triplet_loss = TripletLoss(margin=margin)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler
    start_lr = 1e-3
    final_lr = 1e-5
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    train_loader, train_set = create_dataloader(dataroot,
                                                type='train',
                                                imgsz=image_size,
                                                batch_size=batch_size,
                                                workers=num_workers)
    val_loader, val_set = create_dataloader(dataroot,
                                            type='val',
                                            imgsz=image_size,
                                            batch_size=batch_size,
                                            workers=num_workers)

    best_val_loss = 1000000

    for epoch in range(start_epoch, epochs):
        epoch_loss, val_epoch_loss = 0, 0
        ap_dist, an_dist, train_loss, val_loss = [], [], [], []
        # Warm-up learning rate adjustment
        if epoch < warmup_epochs:
            # Warm-up: Linearly increase or decrease LR
            # Calculate the warmup factor
            warmup_factor = epoch / warmup_epochs
            lr = final_lr + (start_lr - final_lr) * warmup_factor
            # Update optimizer's learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Apply exponential decay
            scheduler.step()

        for batch_train, train_data in tqdm(enumerate(train_loader), total=min(iter_per_epoch, len(train_loader)),
                                            desc=f'Training, iteration {epoch + 1} out of {epochs}'):
            if batch_train == iter_per_epoch:
                break
            l2_distance = PairwiseDistance(p=2)
            optimizer.zero_grad()

            RGBD_anc = torch.cat((train_data[0], train_data[3]/10000), dim=1)
            RGBD_pos = torch.cat((train_data[1], train_data[4]/10000), dim=1)
            RGBD_neg = torch.cat((train_data[2], train_data[5]/10000), dim=1)
            concatenated_data = torch.cat((RGBD_anc, RGBD_pos, RGBD_neg), dim=0)
            anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                imgs=concatenated_data,
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

            # Inside training loop after calculating distances
            total_triplets = anc_embeddings.size(0)
            hard_triplets = (neg_dists - pos_dists < margin).sum().item()
            hard_ratio = hard_triplets / total_triplets

            # Adjust margin after each epoch or after a fixed number of iterations
            if hard_ratio > hard_negative_threshold:
                margin += margin_change
            else:
                margin -= margin_change
            margin = max(margin, 0)  # Ensure the margin does not go negative

            anc_valid_embeddings = anc_embeddings[valid_triplets]
            pos_valid_embeddings = pos_embeddings[valid_triplets]
            neg_valid_embeddings = neg_embeddings[valid_triplets]

            triplet_loss.margin = margin
            loss = triplet_loss.forward(
                anchor=anc_valid_embeddings,
                pos=pos_valid_embeddings,
                neg=neg_valid_embeddings
            )

            # Get the current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics to wandb using the unique step as the x-axis
            train_loss.append(loss.item())
            ap_dist.append(l2_distance.forward(anc_embeddings, pos_embeddings).mean().item())
            an_dist.append(l2_distance.forward(anc_embeddings, neg_embeddings).mean().item())

            loss.backward()
            optimizer.step()
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            val_iteration_limit = 100  # can support up to 100 epochs for batch size 64
            for batch_val, val_data in tqdm(enumerate(val_loader), total=val_iteration_limit,
                                            desc=f'Validating, iteration {epoch + 1} out of {epochs}'):

                if batch_val == val_iteration_limit:
                    break

                RGBD_anc = torch.cat((val_data[0], val_data[3]/10000), dim=1)
                RGBD_pos = torch.cat((val_data[1], val_data[4]/10000), dim=1)
                RGBD_neg = torch.cat((val_data[2], val_data[5]/10000), dim=1)
                concatenated_data = torch.cat((RGBD_anc, RGBD_pos, RGBD_neg), dim=0)

                anc_embeddings, pos_embeddings, neg_embeddings, model = forward_pass(
                    imgs=concatenated_data,
                    model=model,
                    batch_size=batch_size
                )

                triplet_loss.margin = 0.2
                loss = triplet_loss.forward(
                    anchor=anc_embeddings,
                    pos=pos_embeddings,
                    neg=neg_embeddings
                )
                val_loss.append(loss.item())
                val_epoch_loss += loss / val_iteration_limit

            print('The Triplet loss for Validation epoch {} is {}.'.format(epoch + 1, val_epoch_loss))

            # Save model checkpoint for the best validation results
            if val_epoch_loss.item() < best_val_loss or epoch == epochs - 1:
                if epoch >= warmup_epochs:
                    best_val_loss = val_epoch_loss.item()
                    state = {
                        'epoch': epoch,
                        'embedding_dimension': embedding_dimension,
                        'batch_size_training': batch_size,
                        'model_state_dict': model.state_dict(),
                        'optimizer_model_state_dict': optimizer.state_dict(),
                        'train_loss': epoch_loss.item(),
                        'val_loss': val_epoch_loss.item()
                    }
                    if gpu_flag:
                        state['model_state_dict'] = model.module.state_dict()
                    if epoch == epochs - 1:
                        torch.save(state, save_path / 'last.pt')
                        print('Last model saved \n')
                    else:
                        # Save model checkpoint
                        torch.save(state, save_path / 'best.pt')
                        print('New best model saved \n')

        val_loss_index = 0
        val_loss_log_frequency = iter_per_epoch // len(val_loss)  # Frequency to log each val_loss 10 times

        for step in range(len(train_loss)):
            log_data = {
                "training_loss": train_loss[step],
                "ap_dist": ap_dist[step],
                "an_dist": an_dist[step],
                "epoch_training_loss": epoch_loss,  # Log epoch loss for each step
                "epoch_validation_loss": val_epoch_loss,  # Log validation epoch loss for each step
            }

            # Check if it's time to log the next validation loss
            if step % val_loss_log_frequency == 0 and step // val_loss_log_frequency < len(val_loss):
                val_loss_index = step // val_loss_log_frequency

            # Assuming we are to log the current validation loss 10 times within its designated interval
            if (step % val_loss_log_frequency) < 10:
                log_data["val_loss"] = val_loss[val_loss_index]

            wandb.log(log_data)


if __name__ == '__main__':
    wandb.init(project="SiameseViT")
    train()
    wandb.finish()
