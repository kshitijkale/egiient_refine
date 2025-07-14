import logging
import os
import torch
from torch.utils.data import DataLoader
from models.EGIInet import EGIInet

#import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from run.test import test_net
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import *
import os
import sys
from dotenv import load_dotenv
from run import hub_uploader
import wandb
import torch.nn as nn
load_dotenv()
shapenet_path = os.getenv("SHAPENET_DATASET_PATH")
dino_path = os.getenv("DINO_PROJECT_PATH")


def calculate_per_point_cd(p_pred, p_gt):
    """
    Calculates the one-way Chamfer Distance from each point in p_pred to p_gt.
    This serves as the ground truth error score for each predicted point.
    Args:
        p_pred (torch.Tensor): The predicted point cloud. Shape: (B, N, 3)
        p_gt (torch.Tensor): The ground truth point cloud. Shape: (B, M, 3)
    Returns:
        torch.Tensor: The per-point squared distance error. Shape: (B, N, 1)
    """
    p_pred_expanded = p_pred.unsqueeze(2)
    p_gt_expanded = p_gt.unsqueeze(1)
    dist_matrix_sq = torch.sum((p_pred_expanded - p_gt_expanded)**2, dim=-1)
    min_dist_sq, _ = torch.min(dist_matrix_sq, dim=2)
    return min_dist_sq.unsqueeze(-1)

def train_net(cfg):
    torch.backends.cudnn.benchmark = True
    # Start a new wandb run to track this script.
    
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="kshitijkale1212",
        # Set the wandb project where this run will be logged.
        project="egiinet_refine",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "egiinet_base",
            "dataset": "shapenet_vipc_subset",
            "epochs": 24,
        },
        name='egiinet_refine_end_to_end',
    )
    
    ViPC_train = ViPCDataLoader(os.path.join(dino_path,'train_list.txt'),
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                category=cfg.TRAIN.CATE)
    train_data_loader = DataLoader(ViPC_train,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.CONST.NUM_WORKERS,
                              shuffle=True,
                              drop_last=True,
                              prefetch_factor=cfg.CONST.DATA_perfetch)

    ViPC_test = ViPCDataLoader(os.path.join(dino_path,'sampled_20k.txt'),
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                 category=cfg.TRAIN.CATE)
    val_data_loader = DataLoader(ViPC_test,
                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=cfg.CONST.NUM_WORKERS,
                                   shuffle=True,
                                   drop_last=True,
                                   prefetch_factor=cfg.CONST.DATA_perfetch)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.TRAIN.CATE, '%s', datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    
    model = EGIInet()#.apply(weights_init_normal)
    model.cuda()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #steps = cfg.TRAIN.WARMUP_STEPS+1
        # lr_scheduler = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        # optimizer.param_groups[0]['lr']= cfg.TRAIN.LEARNING_RATE
        #logging.info('Recover complete.')

# # === MODIFIED WEIGHT LOADING LOGIC ===
#     if 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS:
#         logging.info('Loading partial weights from %s ...' % (cfg.CONST.WEIGHTS))
        
#         # 1. Load the old checkpoint's state_dict
#         old_checkpoint = torch.load(cfg.CONST.WEIGHTS)
#         old_state_dict = old_checkpoint['model']

#         # If the old model was saved with DataParallel, its keys will start with 'module.'
#         # We need to strip that prefix to match the keys in the new model.
#         if list(old_state_dict.keys())[0].startswith('module.'):
#             old_state_dict = {k[7:]: v for k, v in old_state_dict.items()}

#         # 2. Get the state_dict from the new, randomly initialized model
#         new_model_state_dict = model.module.state_dict() # Use .module to access the base model

#         # 3. Create a new state_dict to load, starting with the new model's state
#         state_dict_to_load = new_model_state_dict.copy()

#         # 4. Iterate through the OLD state_dict and copy matching weights
#         for name, param in old_state_dict.items():
#             # Check if the layer exists in the new model and has the same shape
#             if name in new_model_state_dict and param.shape == new_model_state_dict[name].shape:
#                 state_dict_to_load[name] = param
#                 # print(f"Loaded weight for: {name}") # Uncomment for debugging
        
#         # 5. Load the merged state_dict into the new model
#         model.module.load_state_dict(state_dict_to_load)
        
#         logging.info('Partial weight loading complete. New layers are randomly initialized.')
        
#         # Note: We are NOT loading the optimizer state, as it's tied to the old model structure.
#         # We will start training the optimizer from scratch.

    print('recon_points: ',cfg.DATASETS.SHAPENET.N_POINTS, 'Parameters: ', sum(p.numel() for p in model.parameters()))
    #exit()
    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

        model.train()

        total_cd_pc = 0
        total_style = 0
        total_loss = 0
        total_auxiliary = 0

        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        with tqdm(train_data_loader) as t:
            for batch_idx, (view,gt_pc,part_pc,) in enumerate(t):

                partial = part_pc.cuda()#[16,2048,3]
                gt = gt_pc.cuda()#[16,2048,3]
                png = view.cuda()
                # print(f"Train - Input shapes: partial {partial.shape}, gt {gt.shape}, png {png.shape}")
                
                partial = farthest_point_sample(partial,cfg.DATASETS.SHAPENET.N_POINTS)
                gt = farthest_point_sample(gt,cfg.DATASETS.SHAPENET.N_POINTS)
                # print(f"Train - After FPS: partial {partial.shape}, gt {gt.shape}")
                      
                coarse,confidence_scores,recon,style_loss=model(partial,png,gt)
                # print(f"Train - Model output: recon {recon.shape}, style_loss {style_loss}")
                
                cd = chamfer_sqrt(recon,gt)
                with torch.no_grad(): # Don't track gradients for the target calculation
                    per_point_error_sq = calculate_per_point_cd(coarse, gt)
                    max_error = 0.1                                                                          #NEEDS TUNING
                    gt_confidence = 1.0 - torch.clamp(torch.sqrt(per_point_error_sq) / max_error, 0, 1)
                # print(f"Train - Per-point error: {per_point_error_sq.shape}, gt_confidence: {gt_confidence.shape}")

                auxiliary_loss_fn = nn.L1Loss()
                # print(f"Confidence scores shape: {confidence_scores.shape}, gt_confidence shape: {gt_confidence.shape}")
                auxiliary_loss = auxiliary_loss_fn(confidence_scores, gt_confidence)
                beta = 0.1 # Weight for the auxiliary loss, tune this hyperparameter

                loss_total=cd+style_loss*1e-2+beta*auxiliary_loss
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                cd_pc_item = cd.item() * 1e3
                total_cd_pc += cd_pc_item
                style_item = style_loss.item() * 1e1
                total_style += style_item
                auxiliary_item = auxiliary_loss.item() * 1e1
                total_auxiliary += auxiliary_item
                loss_item = loss_total.item() * 1e3
                total_loss += loss_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/style', style_item, n_itr)
                train_writer.add_scalar('Loss/Batch/auxiliary', auxiliary_item, n_itr)
                train_writer.add_scalar('Loss/Batch/loss', loss_item, n_itr)
                run.log({"cdc":cd_pc_item,"style":style_item,"auxiliary":auxiliary_item,"loss":loss_item,"epoch":epoch_idx,"learning_rate":optimizer.param_groups[0]['lr']})

                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, style_item, auxiliary_item, loss_item]])
                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_style = total_style / n_batches
        avg_auxiliary = total_auxiliary / n_batches
        avg_loss = total_loss / n_batches

        lr_scheduler.step()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/style', avg_style, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/auxiliary', avg_auxiliary, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss', avg_loss, epoch_idx)
        logging.info(
                '[Epoch %d/%d]  Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS,
                 ['%.4f' % l for l in [avg_cdc, avg_style, avg_auxiliary, avg_loss]]))
        run.log({"avg_cdc":avg_cdc,"avg_style":avg_style,"avg_auxiliary":avg_auxiliary,"avg_loss":avg_loss,"epoch":epoch_idx,"learning_rate":optimizer.param_groups[0]['lr']})
        
        
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        run.log({"cd_eval":cd_eval,"epoch":epoch_idx,"learning_rate":optimizer.param_groups[0]['lr']})
        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
            file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)
            
            
            hub_uploader.upload_checkpoint_folder(
                local_dir=cfg.DIR.CHECKPOINTS, 
                repo_id="kshitij121212/shapenet_refine"
                )
            
            # artifact = wandb.Artifact(f"checkpoint-epoch-{epoch_idx:03d}", type="model")
            # artifact.add_file(output_path)
            # wandb.log_artifact(artifact)

            # hub_uploader.upload_checkpoint_folder(
            #     local_dir=cfg.DIR.LOGS, 
            #     repo_id="kshitij121212/shapenet_clean"
            #     )
            logging.info('Saved checkpoint to %s ...' % output_path)
        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))
    run.finish
    train_writer.close()
    val_writer.close()
