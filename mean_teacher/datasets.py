import os
import warnings
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from . import data
from .utils import export

from skimage import io
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage.transform import resize

######################################################
######################################################
######################################################

@export
def cxr14():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                     )

    train_transformation = data.TransformTwice(transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                ]))

    eval_transformation = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                        ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        # 'datadir': '../data/cxr14/',
        # 'csvdir': '../data_csv/',
        # 'num_classes': None
    }


@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
#        transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        #'datadir': 'data-local/images/ilsvrc2012/',
        #'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }


### complete version similar to torchvision.datasets.ImageFolder / torchvision.datasets.DatasetFolder
class ChestXRayDataset(Dataset):
    """ CXR8 dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        df = pd.read_csv(csv_file)
        
        classes = df.columns[3:].values.tolist()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = dict(enumerate(classes))
        self.classes = classes
        
        samples = []
        for idx in range(len(df)):
            path = df.iloc[idx]['image_path']
            target = df.iloc[idx, 3:].as_matrix().astype('float32')       ### labels type: array
            item = (path, target)
            samples.append(item)
        assert(len(samples) == len(df))
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
         
        path, target = self.samples[index]
        
        ### load image
        img_name = os.path.join(self.root_dir, path)       ### get 'image_path'
        image = io.imread(img_name)
        if(len(image.shape) == 3):                          ### some samples have four channels
            image = image[:,:,0]
        h, w = image.shape
        c = 3
        images = np.zeros((h, w, c), dtype = np.uint8)      ### Set image channel dim = 3
        for i in range(c):
            images[:,:,i] = image 
        assert(images.shape == (1024,1024,3))
        images = Image.fromarray(images)

        if self.transform:
            images = self.transform(images)

        ### load labels
        labels = torch.from_numpy(target)
  
        ### return tuple
        return (images, labels)



class IVCdataset(Dataset):
    def __init__(self, csv_file, path, transform=None):
        """
        csv_file = csv where first column = image filenames and second column = classification
        path = directory to all iamges
        """
        self.path = path
        self.transform = transform

        df = pd.read_csv(csv_file, header=None)

        classes = df.iloc[:,1].values.tolist()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = dict(enumerate(classes))
        self.classes = classes
        print("> dataset size: ", df.shape[0])

        #load labels
        samples = []
        for i in range(len(df)):
            name = df.iloc[i,0]
            target = df.iloc[i,1].astype('int_')
            item = (name, target)
            samples.append(item)
        assert(len(samples) == len(df))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_name = os.path.join(self.path, path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = io.imread(img_name)

        if (len(image.shape)==3):
            image = image[:,:,0]

        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    image = resize(image, (224,224))
        image = image.astype('float32') 
        
        h, w  = image.shape
        c = 3
        images = np.zeros((h, w, c), dtype = np.uint8)
        for i in range(c):
            images[:,:,i] = image
        #assert(images.shape == (1024,1024,3))

        images = Image.fromarray(images)         

        #trans = transforms.ToTensor()
        #images = trans(images) 
        
        if self.transform:
            images = self.transform(images)
        
        labels = torch.from_numpy(np.array([target]))
        return (images, labels)



# ### simple original version
# class ChestDataset(Dataset):
#     """ CXR8 dataset."""
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.image_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_frame)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 1])       ### get 'image_path' 

#         ### load image
#         image = io.imread(img_name)
#         if(len(image.shape) == 3):                          ### some samples have four channels
#             image = image[:,:,0]
#         h, w = image.shape
#         c = 3
#         images = np.zeros((h, w, c), dtype = np.uint8)      ### Set image channel dim = 3
#         for i in range(c):
#             images[:,:,i] = image 
#         assert(images.shape == (1024,1024,3))
#         images = Image.fromarray(images)

#         if self.transform:
#             images = self.transform(images)

#         ### load labels
#         labels = self.image_frame.iloc[idx, 3:].as_matrix().astype('float32')         ### get all disease labels
#         assert(len(labels) == args.num_classes)
#         labels = torch.from_numpy(labels)

#         ### return tuple
#         return (images, labels)
