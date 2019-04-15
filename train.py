# Imports here
import argparse
import time
import os.path
import torch
from torch import nn, optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
from workspace_utils import keep_awake, active_session

parser = argparse.ArgumentParser(
    description="Train a CNN to recognize flower images")
parser.add_argument('data_dir', action='store',
                    help='root folder for train & validate image files')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='folder to save the checkpoints. Default=data_dir')
parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg16',
                    help='choose vgg16 (default), vgg19, densenet121 or alexnet')
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.01,
                    type = float,
                    help='learning rate = 0.01 (default)')
parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    default=512,
                    type = int,
                    help='hidden_units = 512 (default)')
parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    default=2,
                    type = int,
                    help='epoch = 2 (default)')
parser.add_argument('--gpu', action='store_true',
                    dest='gpu',
                    default=False,
                    help='switch to use cuda.')

results = parser.parse_args()
data_dir = results.data_dir
if results.save_dir == None:
    save_dir = results.data_dir
else:
    save_dir = results.save_dir
if results.arch == 'vgg19':
    arch = 'vgg19'
elif results.arch == 'densenet121':
    arch = 'densenet121'
elif results.arch == 'alexnet':
    arch = 'alexnet'
else:
    arch = 'vgg16'
learning_rate = results.learning_rate
hidden_units = results.hidden_units
num_epochs = results.epochs
if torch.cuda.is_available():
    gpu = results.gpu
else:
    print('No CUDA hardware. gpu must be False.')
    gpu = False
if os.path.isdir(data_dir):
    print('data_dir      = {!r}'.format(data_dir))
else:
    print('data_dir {!r} does not exist!'.format(data_dir))
if os.path.isdir(save_dir):
    print('save_dir      = {!r}'.format(save_dir))
else:
    print('save_dir {!r} does not exist!'.format(save_dir))
print('arch          = {!r}'.format(arch))
print('learning rate = {!r}'.format(learning_rate))
print('hidden_units  = {!r}'.format(hidden_units))
print('epochs        = {!r}'.format(num_epochs))
print('gpu           = {!r}'.format(gpu))

go = input('\nReady to proceed? (Y/N)')
if (go == 'Y') or (go == 'y'):
    print("Let's go...")
    since = time.time()
else:
    exit()

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms =transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
test_transforms =transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

structures = {"vgg16":25088,
              "vgg19":25088,
              "densenet121":1024,
              "alexnet":9216}

# Customize the network
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

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('dropout',nn.Dropout(0.5)),
    ('inputs', nn.Linear(structures[arch], hidden_units)),
    ('relu1', nn.ReLU()),
    ('hidden_layer1', nn.Linear(hidden_units, 90)),
    ('relu2',nn.ReLU()),
    ('hidden_layer2',nn.Linear(90,80)),
    ('relu3',nn.ReLU()),
    ('hidden_layer3',nn.Linear(80,102)),
    ('output', nn.LogSoftmax(dim=1))
                      ]))
model.classifier = classifier
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), learning_rate, momentum=0.9)
if gpu:
    model.cuda()

# Train the network
print_every = 10
for epoch in keep_awake(range(1, num_epochs+1)):
    t_loss = 0.0
    for i, (t_image, t_label) in enumerate(trainloader, 1):
        if gpu:
            t_image = t_image.cuda()
            t_label = t_label.cuda()
        #Reset gradients
        optimizer.zero_grad()
        #Forward
        output = model.forward(t_image)
        loss = criterion(output, t_label)
        #Backword
        loss.backward()
        #parameter update
        optimizer.step()
        t_loss += loss.item()
        #valiation
        if i % print_every == 0:
            v_loss=0.0
            accuracy=0.0
            model.eval()
            for i, (v_image, v_label) in enumerate(validloader, 1):
                if gpu:
                    v_image = v_image.cuda()
                    v_label = v_label.cuda()
                optimizer.zero_grad()
                with torch.no_grad():   # (PyTorch 4.0) operations below don't track history
                    output = model.forward(v_image)
                    v_loss = criterion(output, v_label)
                    ps = torch.exp(output).data
                    match = (v_label.data == ps.max(1)[1])
                    accuracy += match.type_as(torch.FloatTensor()).mean()

            v_loss = v_loss / len(validloader)
            accuracy = accuracy / len(validloader)

            print("Epoch: {}/{}... ".format(epoch, num_epochs),
                  "Train Loss: {:.4f}".format(t_loss/print_every),
                  "Valid Loss: {:.4f}".format(v_loss),
                  "Accuracy: {:.4f}".format(accuracy))

            t_loss = 0

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

# Save the checkpoints
model.class_to_idx = train_data.class_to_idx
torch.save({'structure':arch,
            'hidden_units':hidden_units,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            save_dir + '/' + 'checkpoints.pth')
