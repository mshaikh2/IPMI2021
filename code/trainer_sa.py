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
from model_sa import TextEncoder, ImageEncoder
# from InceptionScore import calculate_inception_score

from misc.losses import sent_loss, words_loss, sent_triplet_loss, words_triplet_loss

from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

import math
from tqdm import tqdm
import timeit
# from catr.engine import train_one_epoch, evaluate
from misc.config import Config
from transformers import BertConfig,BertTokenizer
from nltk.tokenize import RegexpTokenizer


cfg = Config() # initialize catr config here
# tokenizer = BertTokenizer.from_pretrained(cfg.vocab, do_lower=True)
# retokenizer = BertTokenizer.from_pretrained("catr/damsm_vocab.txt", do_lower=True)
# # reg_tokenizer = RegexpTokenizer(r'\w+')
# frozen_list_image_encoder = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3']

# @torch.no_grad()
# def evaluate(cnn_model, trx_model, cap_model, batch_size, cap_criterion, dataloader_val):
#     cnn_model.eval()
#     trx_model.eval()
#     cap_model.eval() ### 
#     s_total_loss = 0
#     w_total_loss = 0
#     c_total_loss = 0 ###
#     ### add caption criterion here. #####
# #     cap_criterion = torch.nn.CrossEntropyLoss() # add caption criterion here
#     labels = torch.LongTensor(range(batch_size)) # used for matching loss
#     if cfg.CUDA:
#         labels = labels.cuda()
# #         cap_criterion = cap_criterion.cuda() # add caption criterion here
# #     cap_criterion.eval()
#     #####################################

#     val_data_iter = iter(dataloader_val)
#     for step in tqdm(range(len(val_data_iter)),leave=False):
#         data = val_data_iter.next()

#         real_imgs, captions, cap_lens, class_ids, keys, cap_imgs, cap_img_masks, sentences, sent_masks = prepare_data(data)

#         words_features, sent_code = cnn_model(cap_imgs)

#         words_emb, sent_emb = trx_model(captions)

#         ##### add catr here #####
#         cap_preds = cap_model(words_features, cap_img_masks, sentences[:, :-1], sent_masks[:, :-1]) # caption model feedforward

#         cap_loss = caption_loss(cap_criterion, cap_preds, sentences)

#         c_total_loss += cap_loss.item()
#         #########################

#         w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
#                                             cap_lens, class_ids, batch_size)
#         w_total_loss += (w_loss0 + w_loss1).item()

#         s_loss0, s_loss1 = \
#             sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
#         s_total_loss += (s_loss0 + s_loss1).item()

# #             if step == 50:
# #                 break

#     s_cur_loss = s_total_loss / step
#     w_cur_loss = w_total_loss / step
#     c_cur_loss = c_total_loss / step

#     return s_cur_loss, w_cur_loss, c_cur_loss

            
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

        self.batch_size = data_loader.batch_size
        self.val_batch_size = dataloader_val.batch_size
        self.max_epoch = cfg.epochs
        self.snapshot_interval = cfg.snapshot_interval

        self.data_loader = data_loader
        self.dataloader_val = dataloader_val
        self.num_batches = len(self.data_loader)
        self.bert_config = BertConfig(vocab_size=data_loader.dataset.vocab_size, hidden_size=512, num_hidden_layers=3,
                    num_attention_heads=8, intermediate_size=2048, hidden_act='gelu',
                    hidden_dropout_prob=cfg.hidden_dropout_prob, attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
                    max_position_embeddings=512, layer_norm_eps=1e-12,
                    initializer_range=0.02, type_vocab_size=2, pad_token_id=0)

    def build_models(self):
        # ###################encoders######################################## #
      
        image_encoder = ImageEncoder(output_channels=cfg.hidden_dim)
        if cfg.text_encoder_path != '':
            img_encoder_path = cfg.text_encoder_path.replace('text_encoder', 'image_encoder')
            print('Load image encoder from:', img_encoder_path)
            state_dict = torch.load(img_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                image_encoder.load_state_dict(state_dict['model'])
            else:
                image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters(): # make image encoder grad on
            p.requires_grad = True
   
        
#         image_encoder.eval()
        epoch = 0
        
        ###################################################################
        text_encoder = TextEncoder(bert_config = self.bert_config)
        if cfg.text_encoder_path != '':
            epoch = cfg.text_encoder_path[istart:iend]
            epoch = int(epoch) + 1
            text_encoder_path = cfg.text_encoder_path
            print('Load text encoder from:', text_encoder_path)
            state_dict = torch.load(text_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                text_encoder.load_state_dict(state_dict['model'])
            else:
                text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters(): # make text encoder grad on
            p.requires_grad = True
           
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            
        return [text_encoder, image_encoder, epoch]
    
    
    def define_optimizers(self, image_encoder, text_encoder):
        ### change the learning rate in this function ###
        
        #################################
        img_encoder_path = cfg.text_encoder_path.replace('text_encoder', 'image_encoder')
#         print('\n\n CNN Encoder parameters that do not require grad are:')
        optimizerI = torch.optim.AdamW(image_encoder.parameters()
                                           , lr=cfg.lr
                                           , weight_decay=cfg.weight_decay)
        lr_schedulerI = torch.optim.lr_scheduler.StepLR(optimizerI, cfg.lr_drop, gamma=cfg.lr_gamma)
        
        if os.path.exists(img_encoder_path):
            print('Load image encoder optimizer from:', img_encoder_path)
            state_dict = \
                torch.load(img_encoder_path, map_location='cpu')
            optimizerI.load_state_dict(state_dict['optimizer'])
            lr_schedulerI.load_state_dict(state_dict['lr_scheduler'])
        #################################
        text_encoder_path = cfg.text_encoder_path
        optimizerT = torch.optim.AdamW(text_encoder.parameters()
                                           , lr=cfg.lr
                                           , weight_decay=cfg.weight_decay)
        lr_schedulerT = torch.optim.lr_scheduler.StepLR(optimizerT, cfg.lr_drop, gamma=cfg.lr_gamma)
        
        print('Load text encoder optimizer from:', cfg.text_encoder_path)        
        if os.path.exists(cfg.text_encoder_path):
            state_dict = torch.load(cfg.text_encoder_path,map_location='cpu')
            optimizerT.load_state_dict(state_dict['optimizer'])
            lr_schedulerT.load_state_dict(state_dict['lr_scheduler'])
        
       
        ###############################################################

        return (optimizerI
                , optimizerT
                , lr_schedulerI
                , lr_schedulerT)

    def prepare_labels(self):
        batch_size = self.batch_size
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            match_labels = match_labels.cuda()

        return match_labels

    def save_model(self, image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch):
       
        
        # save image encoder model here
        torch.save({
            'model': image_encoder.state_dict(),
            'optimizer': optimizerI.state_dict(),
            'lr_scheduler': lr_schedulerI.state_dict(),
        }, '%s/image_encoder%d.pth' % (self.model_dir, epoch))

        
        # save text encoder model here
        torch.save({
            'model': text_encoder.state_dict(),
            'optimizer': optimizerT.state_dict(),
            'lr_scheduler': lr_schedulerT.state_dict(),
        }, '%s/text_encoder%d.pth' % (self.model_dir, epoch))

        
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
        text_encoder, image_encoder, start_epoch, = self.build_models()
        labels = Variable(torch.LongTensor(range(self.batch_size))) # used for matching loss
        
        text_encoder.train()
        image_encoder.train()
    
        ###############################################################
        
        ###### init optimizers #####
        optimizerI, optimizerT, lr_schedulerI, lr_schedulerT = self.define_optimizers(image_encoder, text_encoder)
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
        
        for epoch in range(start_epoch, self.max_epoch):
            
            ##### set everything to trainable ####
            text_encoder.train()
            image_encoder.train()
            ####################################
            
            ####### init loss variables ############          
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            
            s_t_total_loss0 = 0
            s_t_total_loss1 = 0
            w_t_total_loss0 = 0
            w_t_total_loss1 = 0
            
            total_damsm_loss = 0
            total_t_loss = 0
                      
            ####### print out lr of each optimizer before training starts, make sure lrs are correct #########
            print('Learning rates: lr_i %.7f, lr_t %.7f' 
                 % (optimizerI.param_groups[0]['lr'], optimizerT.param_groups[0]['lr']))
                     
            #########################################################################################
            
            start_t = time.time()

            data_iter = iter(self.data_loader)
#             step = 0
            pbar = tqdm(range(self.num_batches))
            for step in pbar: 
#             while step < self.num_batches:
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                imgs, captions, masks, class_ids, cap_lens = data_iter.next()
                class_ids = class_ids.numpy()
                
                ids = np.array(list(range(batch_size)))
                neg_ids = Variable(torch.LongTensor([np.random.choice(ids[ids!=x]) for x in ids])) # used for matching loss
                
                if cfg.CUDA:
                    imgs, captions, masks, cap_lens = imgs.cuda(), captions.cuda(), masks.cuda(), cap_lens.cuda()
                    neg_ids = neg_ids.cuda()
                # add images, image masks, captions, caption masks for catr model
                
                ################## feedforward damsm model ##################
                image_encoder.zero_grad() # image/text encoders zero_grad here
                text_encoder.zero_grad()
                
                words_features, sent_code, attn_maps_l4, learnable_scalar_l4, attn_maps_l5, learnable_scalar_l5 = image_encoder(imgs) # input images to image encoder, feedforward
                nef, att_sze = words_features.size(1), words_features.size(2)
                # hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, masks) 
                
                #### damsm losses
                w_loss0, w_loss1, attn_maps = words_loss(words_features, words_embs[:,:,1:], labels, cap_lens-1, class_ids, batch_size)
                w_total_loss0 += w_loss0.item()
                w_total_loss1 += w_loss1.item()
                damsm_loss = w_loss0 + w_loss1
                
                s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
                s_total_loss0 += s_loss0.item()
                s_total_loss1 += s_loss1.item()
                damsm_loss += s_loss0 + s_loss1
                
                total_damsm_loss += damsm_loss.item()
                
#                 #### triplet loss
                s_t_loss0, s_t_loss1 = sent_triplet_loss(sent_code, sent_emb, labels, neg_ids, batch_size)
                s_t_total_loss0 += s_t_loss0.item()
                s_t_total_loss1 += s_t_loss1.item()
                t_loss = s_t_loss0 + s_t_loss1
                
                w_t_loss0, w_t_loss1, attn_maps = words_triplet_loss(words_features,words_embs[:,:,1:], labels, neg_ids, cap_lens-1, batch_size)
                w_t_total_loss0 += w_t_loss0.item()
                w_t_total_loss1 += w_t_loss1.item()
                t_loss += w_t_loss0 + w_t_loss1
                
                total_t_loss += t_loss.item()
                ############################################################################
                
                
                
#                 damsm_loss.backward()
#                 t_loss.backward()

                damsm_triplet_combo_loss = cfg.LAMBDA_DAMSM*damsm_loss + cfg.LAMBDA_TRIPLET*t_loss
                total_combo_loss+=damsm_triplet_combo_loss.item()
#                 damsm_loss.backward()
#                 t_loss.backward()
                damsm_triplet_combo_loss.backward()
    
                torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), cfg.clip_max_norm)                    
                optimizerI.step()
                
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), cfg.clip_max_norm)
                optimizerT.step()
                ##################### loss values for each step #########################################
                ## damsm ##
                tbw.add_scalar('Train_step/train_w_step_loss0', float(w_loss0.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_s_step_loss0', float(s_loss0.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_w_step_loss1', float(w_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_s_step_loss1', float(s_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_damsm_step_loss', float(damsm_loss.item()), step + epoch * self.num_batches)

                ## triplet ##
                tbw.add_scalar('Train_step/train_w_t_step_loss0', float(w_t_loss0.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_s_t_step_loss0', float(s_t_loss0.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_w_t_step_loss1', float(w_t_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_s_t_step_loss1', float(s_t_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_t_step_loss', float(t_loss.item()), step + epoch * self.num_batches)

                tbw.add_scalar('Train_step/attn_scalar_l4', float(learnable_scalar_l4.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/attn_scalar_l5', float(learnable_scalar_l5.item()), step + epoch * self.num_batches)
                
                ################################################################################################    
                
                ############ tqdm descriptions showing running average loss in terminal ##############################
#                 pbar.set_description('damsm %.5f' % ( float(total_damsm_loss) / (step+1)))
                pbar.set_description('triplet %.5f' % ( float(total_t_loss) / (step+1)))
                ######################################################################################################
                ##########################################################
            v_s_cur_loss, v_w_cur_loss = self.evaluate(image_encoder, text_encoder, self.val_batch_size)
            print('[epoch: %d] val_w_loss: %.4f, val_s_loss: %.4f' % (epoch, v_w_cur_loss, v_s_cur_loss))
            ### val losses ###
            tbw.add_scalar('Val_step/val_w_loss', float(v_w_cur_loss), epoch)
            tbw.add_scalar('Val_step/val_s_loss', float(v_s_cur_loss), epoch)

#             v_s_cur_loss, _ = self.evaluate(image_encoder, text_encoder, self.val_batch_size)
#             print('[epoch: %d] val_s_loss: %.4f' % (epoch, v_s_cur_loss))
#             ### val losses ###
#             tbw.add_scalar('Val_step/val_s_loss', float(v_s_cur_loss), epoch)
            
            lr_schedulerI.step()
            lr_schedulerT.step()
            
            end_t = time.time()
            
            if epoch % cfg.snapshot_interval == 0:
                self.save_model(image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch)                
                
            

        self.save_model(image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch)                
                

    @torch.no_grad()
    def evaluate(self, cnn_model, trx_model, batch_size):
        cnn_model.eval()
        trx_model.eval()
#         cap_model.eval() ### 
        s_t_total_loss = 0
        w_t_total_loss = 0
        ### add caption criterion here. #####
        labels = Variable(torch.LongTensor(range(batch_size))) # used for matching loss
        if cfg.CUDA:
            labels = labels.cuda()
        #####################################
        
        val_data_iter = iter(self.dataloader_val)
        for step in tqdm(range(len(val_data_iter)),leave=False):
            real_imgs, captions, masks, class_ids, cap_lens = val_data_iter.next()
            class_ids = class_ids.numpy()
            
            ids = np.array(list(range(batch_size)))
            neg_ids = Variable(torch.LongTensor([np.random.choice(ids[ids!=x]) for x in ids])) # used for matching loss
            
            if cfg.CUDA:
                real_imgs, captions, masks, cap_lens = real_imgs.cuda(), captions.cuda(), masks.cuda(), cap_lens.cuda()
                neg_ids = neg_ids.cuda()
            words_features, sent_code = cnn_model(real_imgs)
            words_emb, sent_emb = trx_model(captions, masks)
            
            
            w_loss0, w_loss1, attn = words_loss(words_features, words_emb[:,:,1:], labels, cap_lens-1, class_ids, batch_size)
            w_total_loss += (w_loss0 + w_loss1).data

            s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            s_total_loss += (s_loss0 + s_loss1).data
            
            w_t_loss0, w_t_loss1, _ = words_triplet_loss(words_features, words_emb[:,:,1:], labels, neg_ids, cap_lens-1, batch_size)
            w_t_total_loss += (w_t_loss0 + w_t_loss1).data

            s_t_loss0, s_t_loss1 = sent_triplet_loss(sent_code, sent_emb, labels, neg_ids, batch_size)
            s_t_total_loss += (s_t_loss0 + s_t_loss1).data

        s_t_cur_loss = s_t_total_loss / (step+1)
        w_t_cur_loss = w_t_total_loss / (step+1)
        return s_t_cur_loss, w_t_cur_loss