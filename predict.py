# Imports here
import numpy as np
import torch
from torch import nn
import torchvision.models as models
import argparse
import time
import os.path
import json

parser = argparse.ArgumentParser(
    description="Predict the flower name from a flower image")
parser.add_argument('img_file', action='store',
                    help='image filename, e.g. flowers/test/1/image_06743.jpg')
parser.add_argument('checkpoint', action='store',
                    help='checkpoint filename, e.g.flowers/checkpoints.pth')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default='cat_to_name.json',
                    help='default is cat_to_name.json')
parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    default=3,
                    type = int,
                    help='predict N most likely flowers. Default=3')
parser.add_argument('--gpu', action='store_true',
                    dest='gpu',
                    default=False,
                    help='switch to use cuda')

results = parser.parse_args()
img_file = results.img_file
checkpoint = results.checkpoint
category_names = results.category_names
top_k = results.top_k
if torch.cuda.is_available():
    gpu = results.gpu
else:
    print('No CUDA hardware. gpu must be False.')
    gpu = False
if os.path.isfile(img_file):
    print('img_file       = {!r}'.format(img_file))
else:
    print('img_file {!r} does not exist!'.format(img_file))
if os.path.isfile(checkpoint):
    print('checkpoint     = {!r}'.format(checkpoint))
else:
    print('checkpoint {!r} does not exist!'.format(checkpoint))
if os.path.isfile(category_names):
    print('category_names = {!r}'.format(category_names))
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    print('category_names {!r} does not exist!'.format(category_names))
print('top_k          = {!r}'.format(top_k))
print('gpu            = {!r}'.format(gpu))

go = input('\nReady to proceed? (Y/N)')
if (go == 'Y') or (go == 'y'):
    print("Let's go...")
    since = time.time()
else:
    exit()

# Load checkpoints
model = torch.load(checkpoint)
arch = model['structure']
hidden_layer1 = model['hidden_units']

# Reuild the trained network
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif arch == 'vgg19':
    model = models.vgg19(pretrained=True)
elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)
elif arch == 'alexnet':
    model = models.alexnet(pretrained = True)
else:
    print("{} is not a valid model. Try again.".format(structure))
    exit()

structures = {"vgg16":25088,
              "vgg19":25088,
              "densenet121":1024,
              "alexnet":9216}

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('dropout',nn.Dropout(0.5)),
    ('inputs', nn.Linear(structures[arch], hidden_layer1)),
    ('relu1', nn.ReLU()),
    ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
    ('relu2',nn.ReLU()),
    ('hidden_layer2',nn.Linear(90,80)),
    ('relu3',nn.ReLU()),
    ('hidden_layer3',nn.Linear(80,102)),
    ('output', nn.LogSoftmax(dim=1))
                      ]))

model.classifier = classifier
model.class_to_idx = model['class_to_idx']
model.load_state_dict(model['state_dict'])
if gpu:
    model.cuda()
model.eval()

# Process the image file
''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
'''
# Open the image
img = Image.open(image_path)
# Resize
if img.size[0] > img.size[1]:
    img.thumbnail((10000, 256))
else:
    img.thumbnail((256, 10000))
# Crop
left_margin = (img.width-224)/2
bottom_margin = (img.height-224)/2
right_margin = left_margin + 224
top_margin = bottom_margin + 224
img = image.crop((left_margin, bottom_margin, right_margin,
                  top_margin))
# Normalize
img = np.array(img)/255
mean = np.array([0.485, 0.456, 0.406]) #provided mean
std = np.array([0.229, 0.224, 0.225]) #provided std
img = (img - mean)/std
# Move color channels to first dimension as expected by PyTorch
img = img.transpose((2, 0, 1))
# Numpy -> Tensor
image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
# Add batch of size 1 to image
model_input = image_tensor.unsqueeze(0)
# Calculate the probability
probs = torch.exp(model.forward(model_input))
# Top N probs
top_probs, top_labs = probs.topk(topk)
top_probs = top_probs.detach().numpy().tolist()[0]
top_labs = top_labs.detach().numpy().tolist()[0]
# Convert indices to classes
idx_to_class = {val: key for key, val in
                                  model.class_to_idx.items()}
top_labels = [idx_to_class[lab] for lab in top_labs]
top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

flower_num = img_file.split('/')[2]
flower_name = cat_to_name[flower_num]

print('flower_num = {}'.format(flower_num))
print('flower_name = {}'.format(flower_name))
print('predict: top_labels {}'.format(top_labels))
print('predict: top_flowers {}'.format(top_flowers))
print('predict: top_probs {}'.format(top_probs))

print('The flower name of {!r} is {!r}.'.format(img_file, flower_name))
print('Followings are the prediction results:')
for i in range[topk]:
    print('The most likely flower name is {} with probability {}.format(top_flowers[i], top_probs[i])')

time_elapsed = time.time() - since
print('Prediction takes {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
