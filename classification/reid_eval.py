import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as T
import torchvision
from config import get_config
import random
from PIL import Image
import einops
from models import build_model
import tqdm
global config
import reid_trian

# scaler = torch.cuda.amp.GradScaler()
#
# parser = argparse.ArgumentParser('Get Intermediate Layer Output')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to config file')
# parser.add_argument('--resume', help='resume from checkpoint')
# args = parser.parse_args()
# config = get_config(args)

# emb_model = build_model(config)
# checkpoint = torch.load(config.MODEL.RESUME, map_location='cuda:0')
# emb_model.load_state_dict(checkpoint['model'])
# emb_model = IntermediateLayerGetter(emb_model, get_dense=True)

state_dict = torch.load("C:/Users/ricky/Downloads/ckpt.pth", map_location='cuda:0')
head = reid_trian.REIDHeadDense(2048)
head.load_state_dict(state_dict['head'])
head = head.cuda()

emb_model = torchvision.models.resnet50(pretrained=True)
emb_model = emb_model.cuda()
emb_model.fc = torch.nn.Identity()