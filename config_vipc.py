from easydict import EasyDict as edict
import os
import sys
from dotenv import load_dotenv
load_dotenv()
shapenet_path = os.getenv("SHAPENET_DATASET_PATH")
dino_path = os.getenv("DINO_PROJECT_PATH")
__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.VIPC_PATH        = shapenet_path#path to dataset
#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.DATA_perfetch                          = 8
#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = os.path.join(dino_path,'project_logs')#path to save checkpoints and logs
__C.CONST.DEVICE                                 = '0,1'
#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.EGIInet                              = edict()
__C.NETWORK.EGIInet.embed_dim                    = 192
__C.NETWORK.EGIInet.depth                        = 6
__C.NETWORK.EGIInet.img_patch_size               = 14
__C.NETWORK.EGIInet.pc_sample_rate               = 0.125
__C.NETWORK.EGIInet.pc_sample_scale              = 2
__C.NETWORK.EGIInet.fuse_layer_num               = 2
__C.NETWORK.shared_encoder                       = edict()
__C.NETWORK.shared_encoder.block_head            = 12
__C.NETWORK.shared_encoder.pc_h_hidden_dim       = 192
#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 8
__C.TRAIN.N_EPOCHS                               = 160
__C.TRAIN.SAVE_FREQ                              = 1
__C.TRAIN.LEARNING_RATE                          = 0.0001
__C.TRAIN.LR_MILESTONES                          = [16,24,32,40]
__C.TRAIN.LR_DECAY_STEP                          = [16,24,32,40]
__C.TRAIN.WARMUP_STEPS                           = 100
__C.TRAIN.GAMMA                                  = 0.7
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.CATE                                   = 'all'
__C.TRAIN.d_size                                 = 1
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.CATE                                    = 'all'
__C.TEST.BATCH_SIZE                              = 64
#__C.CONST.WEIGHTS = #os.path.join('/home/kshitij/kshitij/egiinet_dino','checkpoints/all-eight-ckpt-best.pth') #path to pre-trained checkpoints
