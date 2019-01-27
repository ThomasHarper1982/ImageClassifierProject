"""
predict.py
@author Thomas Harper @Date 2018/07/29 predict.py

Examples:

python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_1024_512_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/94/image_07372.jpg
python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_1024_512_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/77/image_00061.jpg
python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_1024_512_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/48/image_04690.jpg
python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_1024_512_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/10/image_07094.jpg
python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_1024_512_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/1/image_06734.jpg
python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_1024_512_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/1/image_06739.jpg
python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_1024_512_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/1/image_06739.jpg --cpu --topK 10
python predict.py --checkpoint C:/Users/Thomas/Anaconda3/envs/classifier_project/CHECKPOINT_3HL_VGG13_512_256_2.pt --image C:/Users/Thomas/Documents/AIPythonCourse/aipnd-project-master/flower_data/valid/1/image_06739.jpg --cpu --topK 10
additional options: topK [int] param, and --cpu flag to use cpu instead of default CUDA

cat_to_name.json in same folder

"""

import io
import requests
import train
import argparse
import json
import os
import sys
from PIL import Image
from torchvision import models, transforms, datasets
import torch
from torch.autograd import Variable
import numpy as np


def feedforward(x, model):
    y = model.forward(x)
    return y

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
    img_pil = img_pil.resize((256,256))
    img_pil = img_pil.crop((16,16,240,240))
    np_image = np.array(img_pil)
#    sampleNpArrayAtPoints(np_image,[0,100],[0,100])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std

    np_image = np.transpose(np_image, (2,0,1))
    #img_pil_transformed_numpy = img_pil_transformed.unsqueeze_(0)
    return np_image

def invertDictionary(d):
    class_keys = d.keys()
    d_i = {}
    for key in class_keys:
        value = d[key]
        d_i[str(value)] = key
    return d_i

def predict(image_path, model, processor = "cuda", topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if model.class_to_idx is None:
        print("Classes not indexed")
        sys.exit(0)
    idx_to_class = invertDictionary(model.class_to_idx)
    inputs = process_image(image_path)
    inputs_tensor = torch.from_numpy(inputs)/255.0
    inputs_tensor.unsqueeze_(0)
    dtype = torch.cuda.FloatTensor if processor == "cuda" else torch.FloatTensor
    outputs = feedforward(inputs_tensor.type(dtype), model.type(dtype))
    outputs = outputs.type(torch.FloatTensor)
    top_results, idx = outputs.topk(topk)
    top_results_list =(np.power(10.0,top_results.detach().numpy())).squeeze().tolist()
    idx_list = idx.detach().numpy().squeeze().tolist()
    probs = top_results_list
    classes = []
    for index in idx_list:
        cat_num = idx_to_class[str(index)]
        classes += [str(cat_num)]
    return (probs, classes)

def convertJsonToDict(json_cat):
    with open(json_cat, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def presentResults(label, probs, cat_list, classes):
    print("\nResults:")
    print("Correct categtory: {}".format(label))
    for (clas, cat, prob) in zip(classes, cat_list, probs):
        print("{0:3}:{1:30}:{2:10}".format(clas, cat, "%0.4f" % prob))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store',
                        dest='checkpoint',
                        help='Enter the file name of checkpoint - DO NOT USE FULL PATH')
    parser.add_argument('--image', action='store',
                        dest='image',
                        help='Select the full path of the image')
    parser.add_argument('--cpu', action='store_true',
                        dest='cpu',
						help='Select this to use CPU instead of Cuda')
    parser.add_argument('--topK', action='store',
                        default=5,
                        type =  int,
                        dest='topK',
                        help='Set the tok <K> parameter')
    parser.add_argument('--json_cat', action='store',
                        default="{}\\cat_to_name.json".format(os.getcwd()),
                        type =  str,
                        dest='json_cat',
                        help='Set path to json mapping file')
    results = parser.parse_args()
    print('checkpoint     = {!r}'.format(results.checkpoint))
    print('image = {!r}'.format(results.image))
    print('cpu = {!r}'.format(results.cpu))
    print('topK = {!r}'.format(results.topK))

    file_path = results.checkpoint
    image = results.image
    cpu = results.cpu
    processor =  "cpu" if cpu else "cuda"
    json_cat = results.json_cat #gobal variable
    topK = results.topK

    file_path = results.checkpoint
    try:
        checkpoint = train.loadCheckpoint(file_path)
    except:
        print("falied to load check point file : '{}'".format(file_path))
    model = train.restoreModel(checkpoint)
    #model = train.determineModel(hl, outputs, dropout, arch, file_path, class_to_idx_json_path=class_to_idx_json_path, config_path=config_path, processor=processor, data =test_data)
    (probs, classes) = predict(image, model,  processor = processor, topk=topK)

    cat_to_name = convertJsonToDict(json_cat)
    label_index = image.split("/")[-2]
    label = cat_to_name[label_index]
    cat_list = list(map(lambda x:cat_to_name[x],classes))
    presentResults(label, probs, cat_list, classes)
