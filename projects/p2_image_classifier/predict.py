import argparse
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image
import numpy as np
from workspace_utils import active_session
import json 
def arg_parser():
  # parser arguments
  parser = argparse.ArgumentParser(description = 'Predict script with a trained model with pytorch with path for the trained model to be loaded') 
  parser.add_argument('image_path', type=str,help='image path')
  parser.add_argument('--checkpoint_dir',dest ='checkpoint_dir', help='data directory',type=str)
  parser.add_argument('--top_k', dest="topk", type=int,help='k most likely classes',default=3)
  parser.add_argument('--category_names', dest="category_names", type=str,help=' mapping of categories to real names', default ='cat_to_name.json')
  parser.add_argument('--gpu', dest="gpu", action='store_true',help='gpu', default=False)

  args = parser.parse_args()
  return args

def load_checkpoint(filepath):
    # load the saved file
    checkpoint = torch.load(filepath+'/checkpoint.pth')
    # download the pretrained model
    exec("model = models.{}(pretrained =True)".format(checkpoint['model']), globals())

    # freeze parameters 
    for param in model.parameters():
        param.requires_grad = False

        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        return model

def category_names_mapping(filename):
	"""
	Read a file that provides the name mapping for the category ids
	"""
	with open('cat_to_name.json', 'r') as f:
    	cat_to_name = json.load(f)
    return cat_to_name
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    test_image = PIL.Image.open(image)
    data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image_modified = data_transforms(test_image)
    return image_modified

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img_tensor = process_image(image_path)
    img_tensor.unsqueeze_(0)
    
    model.to("cpu")
    model.eval()
    
    logps = model.forward(img_tensor)
    ps = torch.exp(logps)
    
    # find 5 top results
    top_p, top_classes = ps.topk(topk, dim=1)
        
    # change the dictionary from class to id Dict
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    
    # convert the top_probability and top_class to numpy array
    top_prob = top_p.detach().numpy()[0]
    top_classes = top_classes.detach().numpy()[0]
    
    # get the labels and the classes for the predicated classes 
    top_labels = [idx_to_class[key] for key in top_classes]
    top_flowers = [cat_to_name[label] for label in top_labels]
    
    return top_prob, top_flowers

args = arg_parser()
model = load_checkpoint(args.checkpoint_dir)
image_path = args.image_path
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
cat_to_name = category_names_mapping(args.category_names)
top_prob, top_flowers = predict(image_path, model, topk=args.topk)

for i in range(len(top_flowers)):
    print('The rank {} of {} with a probability of {}'.format(i+1, top_flowers[i], top_prob[i]))


