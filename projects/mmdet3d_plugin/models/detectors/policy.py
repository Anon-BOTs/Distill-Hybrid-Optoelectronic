import torch
import cv2
import numpy as np
import torch.nn as nn
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, build_loss
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.core import bbox3d2result


import torch as th
import torch.nn as nn

class CustomCombinedExtractorWOCam(nn.Module):
    def __init__(self,  features_dim: int = 256, init_cfg=None):
        super().__init__()
        
        # 解析 observation_space
        self.hd_map_shape = [3, 256, 448]
        self.ego_state_dim = 2
        
        # HD Map 特征提取网络 (处理 BEV 地图)
        self.hd_map_net = nn.Sequential(
            nn.Conv2d(self.hd_map_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(458752, 256),
            nn.ReLU()
        )
        
        # Ego 状态特征提取网络
        self.ego_net = nn.Sequential(
            nn.Linear(self.ego_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.command_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(320 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, hd_map, ego_states, ego_commands) -> th.Tensor:
        # 处理 HD Map (形状调整为 (batch, C, H, W))
        hd_map_features = self.hd_map_net(hd_map)
        
        # 处理 Ego 状态
        ego_features = self.ego_net(ego_states.float())
        
        command_features = self.command_net(ego_commands)
        # 合并特征
        combined_features = th.cat([hd_map_features, ego_features, command_features], dim=1)
        
        # 最终特征映射
        features = self.fc(combined_features)
        return features

@DETECTORS.register_module()
class VectorPolicy(Base3DDetector):
    def __init__(self, 
                 train_cfg=None, 
                 action_loss=dict(type="L1Loss", loss_weight=10),
                 **kwargs):
        super(VectorPolicy, self).__init__()
        self.feature_extractor = CustomCombinedExtractorWOCam()
        self.action_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
        )
        self.action_loss = build_loss(action_loss)
        
    
    def forward_train(self, img, ego_states, ego_action, command, **kwargs):
        feature = self.feature_extractor(img, ego_states, command)
        action = self.action_head(feature)
        action_loss = self.action_loss(action, ego_action)
        
        losses = {}
        losses['action_loss'] = action_loss
        return losses
    
    def forward_test(self, img, ego_states, ego_action, command, **kwargs):
        feature = self.feature_extractor(img, ego_states, command)
        action = self.action_head(feature)
        # img = ((img[0].cpu().permute(1, 2, 0).numpy() * 0.5) + 0.5) * 255
        # img =np.ascontiguousarray(img[..., [2, 1, 0]].astype(np.uint8))
        # ego_action = ego_action.cpu().numpy().tolist()
        # ego_action = [round(ego_action[0][0], 3), round(ego_action[0][1], 3)]

        # action = action.cpu().numpy().tolist()
        # action = [round(action[0][0], 3), round(action[0][1], 3)]
        # cv2.putText(
        #     img,
        #     str(ego_action),
        #     # "collision with obj : {}".format(env.envs[0].coll_obj_type),
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2,
        # )

        # cv2.putText(
        #     img,
        #     str(action),
        #     # "collision with obj : {}".format(env.envs[0].coll_obj_type),
        #     (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 0, 0),
        #     1,
        # )
        # cv2.imwrite(f'/root/tmp/vis/{np.random.rand()}.jpg', img)
        return action

    def simple_test(self):
        pass
    def aug_test(self):
        pass

    def extract_feat(self):
        pass
