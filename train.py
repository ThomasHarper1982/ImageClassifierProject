
"""
train.py
@author Thomas Harper @Date 2018/07/29

Use case examples:
(1)create a new checkpoint, train according to model, optimisation params
The --overwrite flag is necessary to overwrite exisiting checkpoints AND to create new checkpoints
>> python train.py  --checkpoint CHECKPOINT_3HL_VGG13_1024_512_1.pt --data_dir C:/Users/thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data --arch vgg13 --learning_rate 0.00005 --epochs 2 --hl 1024,512 --dropout 0.05 --overwrite
>> >python train.py  --checkpoint CHECKPOINT_3HL_VGG16_1024_512_1.pt --data_dir C:/Users/thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data --arch vgg16 --learning_rate 0.00005 --batch 5 --epochs 2 --hl 1024,512 --dropout 0.05 --overwrite

(2)restore the model from saved checkpoint, restart training with new optimisation state
>> python train.py  --checkpoint CHECKPOINT_3HL_VGG13_1024_512_1.pt --data_dir C:/Users/thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data --learning_rate 0.00005

(3)restore the model from saved checkpoint, and restore optimisation state to resume training from last checkpoint, restarts at epoch 0 and learning rate must be provided
>>python train.py  --checkpoint CHECKPOINT_3HL_VGG13_1024_512_1.pt --data_dir C:/Users/thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data --resume_optimization -learning_rate 0.00005

Also try lower the batch (default is 10) to reduce computational load on GPU
>>python train.py  --checkpoint CHECKPOINT_3HL_VGG16_1024_512_1.pt --data_dir C:/Users/thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data --arch vgg16 --learning_rate 0.00005 --batch 5 --epochs 2 --hl 1024,512 --dropout 0.05 --overwrite

Only
"""


#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
import os.path
import sys
import argparse
import json
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict

#params

#  transforms for the training, validation, and testing sets


training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([ transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
# : Load the datasets with ImageFolder


#data_dir = 'flower_data'

def prepareData(data_dir, transform):
    data = datasets.ImageFolder(data_dir, transform=transform)
    return data

def prepareLoader(data, _batch_size):
    loader = torch.utils.data.DataLoader(data, batch_size=_batch_size,shuffle=True)
    return loader

#trainloader.class_to_idx
input_layers = {'vgg11':25088, 'vgg13':25088, 'vgg16':25088, 'vgg19':25088, 'densenet121':1024, 'densenet169':1024}


def get_pretrained_Network(x):
        return {
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'densenet121': models.densenet121,
            'densenet169': models.densenet169
        }[x]

def defineModel(hls, outputs, dropout, arch):
    f = get_pretrained_Network(arch)
    model = f(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        orderedDictParam = []
        #input
#    print(input_layers[arch])
    t1=('fc1', nn.Linear(input_layers[arch], hls[0]))
    orderedDictParam.append(t1)
    t2=('relu', nn.ReLU())
    orderedDictParam.append(t2)
    t3=('dropout', nn.Dropout(p=dropout))
    orderedDictParam.append(t3)
    #hidden layer
    inc = 2
    for i in range(0, len(hls)-1):
        t4=('fc{}'.format(i+2), nn.Linear(hls[i], hls[i+1]))
        orderedDictParam.append(t4)
        t5=('relu', nn.ReLU())
        orderedDictParam.append(t5)
        t6=('dropout', nn.Dropout(p=dropout))
        orderedDictParam.append(t6)
        inc+=1
    #output
    t4=('fc{}'.format(inc), nn.Linear(hls[len(hls)-1], outputs))
    orderedDictParam.append(t4)
    t5=('relu', nn.ReLU())
    orderedDictParam.append(t5)
    orderedDictParam.append(('output', nn.LogSoftmax(dim=1)))
#    print(orderedDictParam)
    classifier = nn.Sequential(OrderedDict(orderedDictParam))
    #classifier =
    model.classifier = classifier
    return model

def createSaveDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def saveCheckpoint(checkpoint, file_name):
    print("saving + : " + file_name)
    #torch.save(model.state_dict(), file_name)
    torch.save(checkpoint, file_name)

def loadCheckpoint(file_name):
    checkpoint = torch.load(file_name)
    return checkpoint

def updateCheckpoint(checkpoint, model=None, arch=None, epochs = None, optimizer = None):
  #  checkpoint = loadCheckpoint(file_name)
    print("updating checkpoint")
    if not model is None:
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['class_to_idx'] = model.class_to_idx
    if not arch is None:
        checkpoint['arch'] = arch
    if not epochs is None:
        checkpoint['epochs'] = epochs
    if not optimizer is None:
        checkpoint['optimizer'] = optimizer.state_dict()

def restoreModel(checkpoint):
    #f = getattr(torchvision.models,  "vgg13")
    f = getattr(torchvision.models,  checkpoint['arch'])
    model = f(pretrained=True)
    #print(f)
    #print(checkpoint['arch'])

    model.classifier = checkpoint['classifier']
#    print(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    print(model)
    return model

def verify_model(model, dataloader, header, processor = "cuda"):
    """Will verify the model using dataloader """
    correct = 0
    total = 0
    model.to(processor)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(processor), labels.to(processor)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("{} : {}".format(header, 'Accuracy of the network on the %d images: %d %%'%(total, 100 * correct / total)))

def train_model(model, epochs, trainloader,testloader,validloader, optimizer, file_path=None, processor="cuda", save_epoch = True):
    print("back_prop: start")
    #model = nn.model
    print_every = 40
    steps = 0
    # change to cuda
    print("processor " + processor)
    model.to(processor)
    #both optimizer and criterion can be separate modules
    criterion = nn.NLLLoss()
    for e in range(epochs):
        print("epoch: {}".format(e))
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(processor), labels.to(processor)
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0

        verify_model(model, validloader, "validation data")
        verify_model(model, testloader, "test data")
        if save_epoch and file_path != None:
            print("save model")
        #    print(file_path)
            updateCheckpoint(checkpoint, model = model, optimizer = optimizer)
            saveCheckpoint(checkpoint, file_path)
    verify_model(model, validloader, "validation data")
    verify_model(model, testloader, "test data")
    updateCheckpoint(checkpoint, model = model, optimizer = optimizer)
    saveCheckpoint(checkpoint, file_path)



if __name__ == "__main__":
    print("start")
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', action='store',
                        dest='checkpoint',
                        default='{}\\{}'.format(os.getcwd(), 'temp.pt'),
                        help='Enter the file name of checkpoint - DO NOT USE FULL PATH')
    parser.add_argument('--data_dir', action='store',
                        dest='data_dir',
                        default='{}\\{}'.format(os.getcwd(), 'flower_data'),
                        help='Full path to the data folder')
    parser.add_argument('--arch', action='store',
                        dest='arch',
                        default='vgg11',
                        help='Select architecture = {vgg11, vgg13, vgg11, vgg16, densenet121, densenet169}')
    parser.add_argument('--learning_rate', action='store',
                        type=float,
                        dest='learning_rate',
                        help='Set learning_rate')
    parser.add_argument('--epochs', action='store',
                        default=6,
                        type =  int,
                        dest='epochs',
                        help='Set the number of epochs')
    parser.add_argument('--batch', action='store',
                        dest='batch',
                        type=int,
                        default=10,
                        help='Set the batch size')
    parser.add_argument('--overwrite', action='store_true',
                        dest='overwrite',
                        help='Overwrite Checkpoint file')
    parser.add_argument('--hl', action='store',
                        dest='hl',
                        default='512,256',
                        help='Hidden Layer nodes : H1,H2, ... | H1 are ints, no spaces')
    parser.add_argument('--cpu', action='store_true',
                        dest='cpu',
						help='Select this to use CPU instead of Cuda')
    parser.add_argument('--dropout', action='store',
                        dest='dropout',
						type = float,
						default=0.0005,
						help='Select the dropout rate')
    parser.add_argument('--resume_optimization', action='store_true',
                        dest='resume_optimization',
                        default=False,
                        help='Resume optimization state from last checkpoint, therefore any learning rate option will be ignored')

    results = parser.parse_args()
    print('checkpoint     = {!r}'.format(results.checkpoint))
    print('learning_rate        = {!r}'.format(results.learning_rate))
    print('epochs       = {!r}'.format(results.epochs))
    print('batch = {!r}'.format(results.batch))
    print('overwrite = {!r}'.format(results.overwrite))
    print('data_dir = {!r}'.format(results.data_dir))
    print('cpu = {!r}'.format(results.cpu))
    print('optimizer = {!r}'.format(results.resume_optimization))
    print("finish args")
    _batch_size = results.batch
    file_path = results.checkpoint
    arch = results.arch
    batch = results.batch
    epochs = results.epochs
    hl_str = results.hl
    data_dir = results.data_dir
    learning_rate = results.learning_rate
    overwrite = results.overwrite
    cpu = results.cpu
    processor =  "cpu" if cpu else "cuda"
    dropout = results.dropout
    data_dir = results.data_dir
    resume_optimization = results.resume_optimization
    hl = [512, 256]
    outputs = 102 #problem output hardcoded
    try:
        hl =list(map(lambda x: int(x),hl_str.split(',')))
    except:
        print('Could not parse hidden layer inputs {}, please enter in form: E.G: 512,256 (Postive Integers, comma separated, no spaces)'.format(hl))
        sys.exit(0)

    #createSaveDir(save_dir)
    test_data = prepareData(data_dir+"/test", test_transforms)
    training_data = prepareData(data_dir+"/train", training_transforms)
    validation_data = prepareData(data_dir+"/valid", validation_transforms)
    test_loader = prepareLoader(test_data, _batch_size)
    train_loader = prepareLoader(training_data, _batch_size)
    validation_loader = prepareLoader(validation_data, _batch_size)
    #create class_to_idx.json in checkpoint folder

    checkpoint = {}
    model = None
    if overwrite: #overwrite model
        print('arch   = {!r}'.format(results.arch))
        print('dropout = {!r}'.format(results.dropout))
        print('hl = {!r}'.format(results.hl))
        model = defineModel(hl, outputs, dropout, arch)
        model.class_to_idx = test_data.class_to_idx
        updateCheckpoint(checkpoint, model = model, arch = arch, epochs = epochs)
    else:
        checkpoint = loadCheckpoint(file_path)
        model = restoreModel(checkpoint)
        verify_model(model , test_loader, "test data", processor = processor)
        verify_model(model , validation_loader, "validation data", processor = processor)
    #Here I have chosen to decouple the optimizer from the model to make it easier to re-run the same model with different optimizer parameters
    optimizer = None
    if resume_optimization:
        print("resuming training")
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        if learning_rate is None:
            learning_rate = 0.0001
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        updateCheckpoint(checkpoint, optimizer = optimizer)

    train_model(model, epochs,train_loader, test_loader, validation_loader, optimizer, file_path=file_path, processor=processor)
