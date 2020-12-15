from __future__ import print_function
from six.moves import range
import os
import time
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
import datetime
import dateutil.tz
from misc.utils import mkdir_p
# from datasets import prepare_data
from model import ImageEncoder_Classification
from sklearn.metrics import roc_auc_score
# from InceptionScore import calculate_inception_score

# from misc.losses import sent_loss, words_loss, sent_triplet_loss, words_triplet_loss

from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

import math
from tqdm import tqdm
import timeit
# from catr.engine import train_one_epoch, evaluate
from misc.config import Config
# from transformers import BertConfig,BertTokenizer
# from nltk.tokenize import RegexpTokenizer


cfg = Config() 

            
# ################# Joint Image Text Representation (JoImTeR) learning task############################ #
class JoImTeR(object):
    def __init__(self, output_dir, data_loader,dataloader_val):
        if cfg.TRAIN:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True
        Q_c = np.array([46,53,86,103,137,265,290,2422])
        Q = 3166.0
        self.class_weight_vector = torch.tensor((Q - Q_c)/Q)        
        self.batch_size = data_loader.batch_size
        self.val_batch_size = dataloader_val.batch_size
        self.max_epoch = cfg.epochs
        self.snapshot_interval = cfg.snapshot_interval
        pos_weights = torch.tensor([3.255, 3.255, 3.255, 3.255, 3.255, 3.255, 3.255, 1.0])
        self.criterion = nn.BCEWithLogitsLoss(reduction='none',pos_weight = pos_weights)
        
        self.data_loader = data_loader
        self.dataloader_val = dataloader_val
        self.num_batches = len(self.data_loader)  
        if cfg.CUDA:
            pos_weights = pos_weights.cuda()
            self.class_weight_vector = self.class_weight_vector.cuda()
            self.criterion = self.criterion.cuda()
        
    def build_models(self):
        # ###################encoders######################################## #
        epoch = 0
        image_encoder = ImageEncoder_Classification(make_soft_attention = cfg.make_soft_attention
                                                    , freeze_backbone = cfg.freeze_backbone
                                                    , encoder_path=cfg.init_image_encoder_path
                                                    , pretrained=cfg.pretrained
                                                    , cfg = cfg)

        if cfg.text_encoder_path != '':
            img_encoder_path = cfg.text_encoder_path.replace('text_encoder', 'image_encoder')
            print('Load image encoder from:', img_encoder_path)
            state_dict = torch.load(img_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                image_encoder.load_state_dict(state_dict['model'])
                epoch = state_dict['epoch']
                epoch = int(epoch) + 1
            else:
                image_encoder.load_state_dict(state_dict)
                
        if cfg.freeze_backbone:
            for p in image_encoder.pretrained_encoder.parameters(): # make image encoder grad on
                p.requires_grad = False
            
            for n, p in image_encoder.named_parameters():
                if 'pretrained_encoder' not in n:
                    p.requires_grad = True
   
        
        # ########################################################### #
        if cfg.CUDA:
            image_encoder = image_encoder.cuda()
            
        return [image_encoder, epoch]
    
    
    def define_optimizers(self, image_encoder):
        ### change the learning rate in this function ###
        
        #################################
        classification_parameters = []
        for n, p in image_encoder.named_parameters():
            if 'pretrained_encoder' not in n:
                classification_parameters.append(p)
        optim_params = [
            {'params':image_encoder.pretrained_encoder.parameters(), 'lr':cfg.lr_backbone}, 
            {'params':classification_parameters, 'lr':cfg.lr}
        ]
#         print('\n\n CNN Encoder parameters that do not require grad are:')
        optimizerI = torch.optim.Adam(optim_params
                                           , weight_decay=cfg.weight_decay)
        
        lr_schedulerI = torch.optim.lr_scheduler.StepLR(optimizerI, cfg.lr_drop, gamma=cfg.lr_gamma)
        
        if cfg.text_encoder_path != '':
            img_encoder_path = cfg.text_encoder_path.replace('text_encoder', 'image_encoder')
            print('Load image encoder optimizer from:', img_encoder_path)
            state_dict = \
                torch.load(img_encoder_path, map_location='cpu')
            optimizerI.load_state_dict(state_dict['optimizer'])
            lr_schedulerI.load_state_dict(state_dict['lr_scheduler'])
        #################################
        
       
        ###############################################################

        return (optimizerI
                , lr_schedulerI)

    def prepare_labels(self):
        batch_size = self.batch_size
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            match_labels = match_labels.cuda()

        return match_labels

    def save_model(self, image_encoder, optimizerI, lr_schedulerI, epoch):
       
        
        # save image encoder model here
        torch.save({
            'model': image_encoder.state_dict(),
            'optimizer': optimizerI.state_dict(),
            'lr_scheduler': lr_schedulerI.state_dict(),
            'epoch':epoch
        }, '%s/image_encoder.pth' % (self.model_dir))

        
        
    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires
            
    def train(self):
        
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        #     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
        tb_dir = '../tensorboard/{0}_{1}_{2}'.format(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(tb_dir)
        tbw = SummaryWriter(log_dir=tb_dir) # Tensorboard logging

        
        ####### init models ########
        image_encoder, start_epoch = self.build_models()
        labels = Variable(torch.LongTensor(range(self.batch_size))) # used for matching loss
        
        image_encoder.train()
    
        ###############################################################
        
        ###### init optimizers #####
        optimizerI, lr_schedulerI = self.define_optimizers(image_encoder)
        ############################################
        
        ##### init data #############################
        
        match_labels = self.prepare_labels()

        batch_size = self.batch_size
        ##################################################################
        
        
        
        ###### init caption model criterion ############
        if cfg.CUDA:
            labels = labels.cuda()
        #################################################
        
        tensorboard_step = 0
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        
        #### print lambdas ###
#         print('LAMBDA_GEN:{0},LAMBDA_CAP:{1},LAMBDA_FT:{2},LAMBDA_FI:{3},LAMBDA_DAMSM:{4}'.format(cfg.TRAIN.SMOOTH.LAMBDA_GEN
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_CAP
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_FT
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_FI                                                                                                  
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_DAMSM))
        
        best_val_loss = 100000.0
        
        for epoch in range(start_epoch, self.max_epoch):
            
            ##### set everything to trainable ####
            image_encoder.train()
            total_bce_loss_epoch = 0.0
                      
            ####### print out lr of each optimizer before training starts, make sure lrs are correct #########
            print('Learning rates: lr_i %.7f'  % (optimizerI.param_groups[0]['lr']))
                     
            #########################################################################################
            
            start_t = time.time()

            data_iter = iter(self.data_loader)
#             step = 0
            pbar = tqdm(range(self.num_batches))
            
            for step in pbar:
                imgs, classes = data_iter.next()
                if cfg.CUDA:
                    imgs, classes = imgs.cuda(), classes.cuda()
                # add images, image masks, captions, caption masks for catr model
                
                ################## feedforward classification model ##################
                image_encoder.zero_grad() 
                
                y_pred = image_encoder(imgs) # input images to image encoder, feedforward
                bce_loss = self.criterion(y_pred,classes)
                bce_loss = bce_loss*self.class_weight_vector
                bce_loss = bce_loss.mean()
                total_bce_loss_epoch+=bce_loss.item()
                
                bce_loss.backward()
    
                torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), cfg.clip_max_norm)                    
                optimizerI.step()
                ##################### loss values for each step #########################################
                ## damsm ##
                tbw.add_scalar('Train_step/loss', float(total_bce_loss_epoch / (step+1)), step + epoch * self.num_batches)
                ## triplet ##
                ################################################################################################    
                
                ############ tqdm descriptions showing running average loss in terminal ##############################
#                 pbar.set_description('damsm %.5f' % ( float(total_damsm_loss) / (step+1)))
                pbar.set_description('loss %.5f' % ( float(total_bce_loss_epoch) / (step+1)))
                ######################################################################################################
                ##########################################################
            v_loss, auc_scores = self.evaluate(image_encoder, self.val_batch_size)
            print('[epoch: %d] val_loss: %.4f' % (epoch, v_loss))
            print('-'*80)
            ### val losses ###
            tbw.add_scalar('Val_step/loss', v_loss, epoch)
            for idx in range(len(auc_scores)):
                tbw.add_scalar('Val_step/{0}'.format(self.data_loader.dataset.idx_to_class[idx]), auc_scores[idx], epoch)
            
            lr_schedulerI.step()
            end_t = time.time()
            
            if v_loss<best_val_loss:
                best_val_loss=v_loss
                self.save_model(image_encoder, optimizerI, lr_schedulerI, epoch)                
                
            

#         self.save_model(image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch)                
                


    @torch.no_grad()
    def evaluate(self, cnn_model, batch_size):
        cnn_model.eval()
        ### add caption criterion here. #####
        labels = Variable(torch.LongTensor(range(batch_size))) # used for matching loss
        if cfg.CUDA:
            labels = labels.cuda()
        #####################################
        total_bce_loss_epoch=0.0
        val_data_iter = iter(self.dataloader_val)
        y_preds = []
        y_trues = []
        class_auc = []
        for step in tqdm(range(len(val_data_iter)),leave=False):
            real_imgs, classes = val_data_iter.next()
            if cfg.CUDA:
                real_imgs, classes = real_imgs.cuda(), classes.cuda()
                
            y_pred = cnn_model(real_imgs)
            y_pred_sigmoid = torch.sigmoid(y_pred)
            
            bce_loss = self.criterion(y_pred,classes)
            bce_loss = bce_loss*self.class_weight_vector
            bce_loss = bce_loss.mean()
            
            total_bce_loss_epoch+=bce_loss.item()
            y_preds.append(y_pred_sigmoid.detach().cpu().numpy())  
            y_trues.append(classes.detach().cpu().numpy())
        y_preds = np.concatenate(y_preds,axis=0)
        y_trues = np.concatenate(y_trues,axis=0)
        
        for i in range(y_preds.shape[-1]):
            class_auc.append(roc_auc_score(y_trues[:,i],y_preds[:,i]))
        
        v_cur_loss = total_bce_loss_epoch / (step+1)
        return v_cur_loss, class_auc