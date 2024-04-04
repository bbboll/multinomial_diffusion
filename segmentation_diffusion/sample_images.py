import os
import math
import torch
import numpy as np
import pickle
import argparse
import torchvision.utils as vutils
from diffusion_utils.utils import add_parent_path
import torchvision
import imageio
from PIL import Image
from tqdm import tqdm

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args, get_plot_transform

# Model
from model import get_model, get_model_id, add_model_args
from diffusion_utils.base import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--samples', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

##################
## Specify data ##
##################

_, _, data_shape, _ = get_data(args)


###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)

if torch.cuda.is_available():
    checkpoint = torch.load(path_check)
else:
    checkpoint = torch.load(path_check, map_location='cpu')
model.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    model = model.cuda()

print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))


############
## Sample ##
############
plot_transform = get_plot_transform(args)

def batch_samples_to_rgb(batch):
    if len(batch.size()) == 3:
        batch = batch.unsqueeze(1)

    batch = plot_transform(batch).to(torch.uint8)
    return batch

savedir = f"{eval_args.model}/sample_img"
if not os.path.isdir(savedir):
    os.mkdir(savedir)

ind = 0
num_batches = (eval_args.samples // eval_args.batch_size) + 1


for batch_ind in tqdm(range(num_batches)):
    with torch.no_grad():
        samples = model.sample(eval_args.batch_size)

    samples = samples.squeeze(1)
    samples = (samples*255).to(torch.uint8).cpu()

    for x in samples:
        image = Image.fromarray(x.numpy(), 'L')
        image.save(os.path.join(savedir, f"{ind}_{eval_args.steps}steps.png"))
        ind += 1
