import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torchvision as tv

import os,sys
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# import spacy
# import scispacy
from tqdm import tqdm 
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer


# nlp = spacy.load('en_core_sci_md')
MAX_DIM = 2048

train_transform = tv.transforms.Compose([
    tv.transforms.RandomRotation(10)
    ,tv.transforms.RandomCrop(MAX_DIM)
    ,tv.transforms.ColorJitter(brightness=[0.5, 1.8]
                              , contrast=[0.5, 1.8]
                              , saturation=[0.5, 1.8])
    ,tv.transforms.ToTensor()
    ,tv.transforms.Normalize(0.5, 0.5)
])

val_transform = tv.transforms.Compose([
    tv.transforms.CenterCrop(MAX_DIM),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(0.5, 0.5)
])




# def remove_whitespace(line):
#     return str(" ".join(line.split()).strip())

# def list_to_string(sentence):
#     return " ".join(sentence)

# def normalize_report(row):
#     report = row
#     report_sentences = nlp(report)
#     new_report_sentences = []
#     for sentence in report_sentences.sents:
#         index_to_keep_dict = {} # index: {keep that token or not, replace_with}
#         for index in range(0, len(sentence)):
#             token = sentence[index]
#             if index < len(sentence) - 1:
#                 next_token = sentence[index + 1]
#                 if token.is_punct and next_token.is_punct and token.text.strip() == next_token.text.strip():
#                     # when it is the same type of punctuation
#                     index_to_keep_dict[index] = {'keep': False, 'replace_with': None}
#                     continue
#             if token.like_num:
#                 index_to_keep_dict[index] = {'keep': True, 'replace_with': 'NUMBER'}
#             else:
#                 index_to_keep_dict[index] = {'keep': True, 'replace_with': None}
#         # generate a new sentence based on this replacement
#         new_sentence = []
#         for index in range(0, len(sentence)):
#             token = sentence[index]
#             if not index_to_keep_dict[index]['keep']:
#                 continue # don't append when there is a double punctuation happening
#             if index_to_keep_dict[index]['replace_with'] is not None:
#                 new_sentence.append(index_to_keep_dict[index]['replace_with'])
#                 continue
#             new_sentence.append(token.text)
#         s = list_to_string(new_sentence).strip()
#         s = s.replace('DEID', '')
#         s = remove_whitespace(s)
#         new_report_sentences.append(s)
#     return {'sentences': ' '.join(new_report_sentences).replace(',','').replace('.','')}


# 2739_IM-1193-1001.dcm.png' not 2048
class IUDataset(Dataset):
    def __init__(self, root, dataset, max_length, transform=train_transform, mode='train'):
        super().__init__()
        
        self.root = root #save dir
        self.transform = transform
        self.mode = mode
#         vocab='allenai/scibert_scivocab_uncased'
#         self.tokenizer = BertTokenizer.from_pretrained(vocab, do_lower=True)
        self.classes = dataset['classes']
        self.datadict = dataset['data_dict'] # uid: {text:text, filenames:[filename]}
        if self.mode == 'train':
            self.keys = dataset['data_split']['train_uids'] # uid list
        elif self.mode == 'val':
            self.keys = dataset['data_split']['val_test_uids'] # uid list
#         elif self.mode == 'test':
#             self.keys = dataset['data_split']['test_uids'] # uid list
        
        self.idx2word = dataset['idx2word']
        self.word2idx = dataset['word2idx']
        self.__sep_id__ = dataset['word2idx']['[SEP]']
        self.vocab_size = len(dataset['word2idx'])
        
        self.max_length = max_length + 1
        self.class_to_idx = {'no finding':7
                             ,'atelectasis':0
                             ,'cardiomegaly':1
                             ,'effusion':2
                             ,'emphysema':3
                             ,'infiltrate':4
                             ,'nodule':5
                             ,'thickening':6}
        self.num_classes = 7

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        uid = self.keys[idx]
        
        image_id = np.random.choice(self.datadict[uid]['filenames'])# get one file name randomly
        image = Image.open(os.path.join(self.root, image_id))
        if image.size[0]<2048 or image.size[1]<2048:
            image = tv.transforms.Resize(MAX_DIM)(image)
            # print(image.size,':', self.datadict[uid]['filenames'])
            
        if self.transform:
            image = self.transform(image)
        
        classes = torch.tensor([self.class_to_idx[x] for x in self.classes[uid]])
#         print(classes)
        y_onehot = torch.FloatTensor(self.num_classes).zero_()
#         print(y_onehot)
        if 7 not in classes: # all zeros for No Finding, for others multi-1-hot
            y_onehot.scatter_(0, classes, 1)
#         print(y_onehot)
        return image, y_onehot

def build_dataset(mode='train', cfg=None):
    data_dir = '../data/'
    img_dir = os.path.join(data_dir, 'images', 'images_normalized')
    with open('../data/cleaned_dataset.pickle','rb') as f:
        dataset = pickle.load(f)
    if mode == 'train':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=train_transform)
        return data

    elif mode == 'val':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=val_transform)
        return data
    
#     elif mode == 'test':
#         data = IUDataset(img_dir, dataset, 
#                                max_length=cfg.max_length, mode=mode, 
#                                transform=val_transform)
#         return data

    else:
        raise NotImplementedError(f"{mode} not supported")