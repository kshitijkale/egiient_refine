# import numpy as np
# #from deformable_attention import DeformableAttention1D
# from torch import nn
# from models.layers.subsample import furthest_point_sample, random_sample
# from models.dec_net import Decoder_Network
# import torch
# from torch.nn import Module
# from models.encoders import  transfer_loss_shared_encoder
# from config_vipc import cfg

# class EGIInet(Module):
#     def __init__(self, 
#                  embed_dim=cfg.NETWORK.EGIInet.embed_dim,
#                  depth=cfg.NETWORK.EGIInet.depth,
#                  img_patch_size=cfg.NETWORK.EGIInet.img_patch_size,
#                  pc_sample_rate=cfg.NETWORK.EGIInet.pc_sample_rate,
#                  pc_sample_scale=cfg.NETWORK.EGIInet.pc_sample_scale,
#                  fuse_layer_num=cfg.NETWORK.EGIInet.fuse_layer_num,
#                  ):
#         super().__init__()
#         self.encoder=transfer_loss_shared_encoder(embed_dim=embed_dim,
#                                                img_patch_size=img_patch_size,
#                                                sample_ratio=pc_sample_rate,
#                                                scale=pc_sample_scale,
#                                                block_head=cfg.NETWORK.shared_encoder.block_head,
#                                                depth=depth,
#                                                pc_h_hidden_dim=cfg.NETWORK.shared_encoder.pc_h_hidden_dim,
#                                                fuse_layer_num=fuse_layer_num,
#                                                )
#         self.decoder=Decoder_Network(K1=embed_dim,K2=embed_dim,N=embed_dim)
#         feature_channels = 192 # From your encoder
#         trans_dim = 384 # A common dimension for transformers
#         global_feature_dim =1024
#         self.num_query = 2048
#         self.increase_dim = nn.Sequential(
#             nn.Linear(192, 1024),
#             nn.GELU(),
#             nn.Linear(1024, global_feature_dim))
        
#         self.ranking = nn.Sequential(
#             nn.Linear(global_feature_dim+3, 512),
#             nn.GELU(),
#             nn.Linear(512, 256),
#             nn.GELU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#     def forward(self,pc,img,gt=None):
#         # print(f"EGIInet - Input shapes: pc {pc.shape}, img {img.shape}")
#         bs = pc.shape[0]
#         feature, _, _, style_loss = self.encoder(pc=pc, im=img)
#         # print("EGIInet - Encoder output - shape of the feature:", feature.shape)
#         # print(f"EGIInet - Encoder output - style_loss: {style_loss}")
    
#         final = self.decoder(feature, pc)
#         # print(f"EGIInet - Decoder output - final shape: {final.shape}")
#         #point refinement#############
#         coarse = final
#         global_feature = torch.max(feature, dim=1)[0]  # Shape: [B, 192]
#         global_feature = self.increase_dim(global_feature)
#         # print(f"EGIInet - Global feature shape: {global_feature.shape}")
#         # print(f"EGIInet - Coarse shape before ranking: {coarse.shape}")
#         corse_feat=torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),coarse],dim=-1)
#         confidence_score = self.ranking(corse_feat).reshape(bs, -1, 1)
#         idx = torch.argsort(confidence_score, dim=1, descending=True)  # b 512 1
#         coarse = torch.gather(coarse, 1, idx[:, :(self.num_query-self.num_query//4)].expand(-1, -1, 3))  # b 384 3

#         coarse_inp_idx = furthest_point_sample(pc, self.num_query // 4).long()  # B 128 3
#         coarse_inp = torch.gather(pc, 1, coarse_inp_idx.unsqueeze(-1).expand(-1, -1, 3))
#         # print(f"EGIInet - Coarse input shape: {coarse_inp.shape}")
#         # print(f"EGIInet - Coarse shape after gathering: {coarse.shape}")
#         coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 512 3
#         return coarse,style_loss

# if __name__ == '__main__':
#     import time
#     model = EGIInet().cuda()
#     pc = torch.rand([4, 2048, 3]).cuda()
#     img = torch.rand([4, 3, 224, 224]).cuda()
#     s=time.time()
#     fine = model(pc, img)
#     e=time.time()
#     print(e-s)
#     print(fine.shape)
#     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     parameters = sum([np.prod(p.size()) for p in model_parameters])
#     print(f"n parameters:{parameters}")


import numpy as np
import torch
from torch import nn
from torch.nn import Module

from models.dec_net import Decoder_Network
from models.encoders import transfer_loss_shared_encoder
from models.layers.subsample import furthest_point_sample
from config_vipc import cfg

# Helper function for K-Nearest Neighbors, essential for local context
def knn(x, k):
    """
    Find k nearest neighbors for each point in a batch of point clouds.
    
    Args:
        x (torch.Tensor): Input point clouds, shape (B, D, N).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of the k nearest neighbors, shape (B, N, k).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

class EGIInet(Module):
    def __init__(self, 
                 embed_dim=cfg.NETWORK.EGIInet.embed_dim,
                 depth=cfg.NETWORK.EGIInet.depth,
                 img_patch_size=cfg.NETWORK.EGIInet.img_patch_size,
                 pc_sample_rate=cfg.NETWORK.EGIInet.pc_sample_rate,
                 pc_sample_scale=cfg.NETWORK.EGIInet.pc_sample_scale,
                 fuse_layer_num=cfg.NETWORK.EGIInet.fuse_layer_num,
                 ):
        super().__init__()
        self.encoder = transfer_loss_shared_encoder(embed_dim=embed_dim,
                                               img_patch_size=img_patch_size,
                                               sample_ratio=pc_sample_rate,
                                               scale=pc_sample_scale,
                                               block_head=cfg.NETWORK.shared_encoder.block_head,
                                               depth=depth,
                                               pc_h_hidden_dim=cfg.NETWORK.shared_encoder.pc_h_hidden_dim,
                                               fuse_layer_num=fuse_layer_num,
                                               )
        self.decoder = Decoder_Network(K1=embed_dim, K2=embed_dim, N=embed_dim)
        
        # --- Point Refinement Module Initialization ---
        global_feature_dim = 1024
        self.num_query = 2048
        self.local_context_k = 16 # Number of neighbors for local context

        self.increase_dim = nn.Sequential(
            nn.Linear(192, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        
        # --- MODIFICATION 1: Update Ranking Network Input Dimension ---
        # The input now includes:
        # 1. Global feature (global_feature_dim)
        # 2. Point coordinates (3)
        # 3. Aggregated local context feature (3)
        ranking_input_dim = global_feature_dim + 3 + 3
        
        self.ranking = nn.Sequential(
            nn.Linear(ranking_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, pc, img, gt=None):
        bs = pc.shape[0]
        feature, _, _, style_loss = self.encoder(pc=pc, im=img)
    
        final = self.decoder(feature, pc)
        
        # --- Point Refinement Module ---
        coarse = final
        global_feature = torch.max(feature, dim=1)[0]  # Shape: [B, 192]
        global_feature = self.increase_dim(global_feature) # Shape: [B, 1024]

        # --- MODIFICATION 2: Calculate Local Context Feature ---
        # Find k-nearest neighbors for each point in the coarse cloud
        # Note: knn expects (B, D, N), so we transpose
        neighbor_idx = knn(coarse.transpose(1, 2).contiguous(), k=self.local_context_k)
        
        # Gather the coordinates of the neighbors
        # We need to manually index to gather neighbors efficiently
        B, N, D = coarse.shape
        # Create a flat index for gathering
        idx_base = torch.arange(0, B, device=coarse.device).view(-1, 1, 1) * N
        idx_flat = (neighbor_idx + idx_base).view(-1)
        coarse_flat = coarse.view(-1, D)
        # Gather neighbors using the flat index
        neighbors = coarse_flat[idx_flat, :].view(B, N, self.local_context_k, D)
        
        # Calculate relative coordinates (neighbors - center)
        center_points_expanded = coarse.unsqueeze(2)
        relative_neighbors = neighbors - center_points_expanded
        
        # Aggregate local features by max-pooling over the neighbors
        local_feature = torch.max(relative_neighbors, dim=2)[0] # Shape: [B, N, 3]
        # --- End of Local Context Calculation ---

        # Expand global feature to match the point dimension
        expanded_global_feature = global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1)

        # Create the new feature vector for the ranking network by concatenating all features
        # [global_feature, point_coordinates, local_context_feature]
        coarse_feat = torch.cat([expanded_global_feature, coarse, local_feature], dim=-1)
        
        # The rest of the logic remains the same
        confidence_score = self.ranking(coarse_feat).reshape(bs, -1, 1)
        idx = torch.argsort(confidence_score, dim=1, descending=True)
        
        # Select the top 75% of generated points based on the new confidence score
        num_to_keep_generated = self.num_query - (self.num_query // 4)
        coarse_best = torch.gather(coarse, 1, idx[:, :num_to_keep_generated].expand(-1, -1, 3))

        # Sample 25% of points from the original partial input
        num_to_keep_input = self.num_query // 4
        coarse_inp_idx = furthest_point_sample(pc, num_to_keep_input).long()
        coarse_inp = torch.gather(pc, 1, coarse_inp_idx.unsqueeze(-1).expand(-1, -1, 3))
        
        # Combine the selected points for the final output
        coarse_final = torch.cat([coarse_best, coarse_inp], dim=1)
        
        return coarse,confidence_score,coarse_final,style_loss

if __name__ == '__main__':
    import time
    model = EGIInet().cuda()
    pc = torch.rand([4, 2048, 3]).cuda()
    img = torch.rand([4, 3, 224, 224]).cuda()
    s=time.time()
    # The model now returns a tuple (coarse_points, style_loss)
    coarse_points, style_loss_val = model(pc, img)
    e=time.time()
    print(f"Execution time: {e-s:.4f}s")
    print(f"Final output shape: {coarse_points.shape}")
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {parameters}")
