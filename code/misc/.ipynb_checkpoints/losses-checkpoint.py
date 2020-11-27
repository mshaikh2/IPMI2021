import random
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CosineSimilarity, MarginRankingLoss
from misc.config import Config
from GlobalAttention import func_attention

cfg = Config()

# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.bool)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.bool)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps

def ranking_loss(z_image, z_text, y, report_id, 
                 similarity_function='dot'):
    """
    A custom ranking-based loss function
    Args:
        z_image: a mini-batch of image embedding features
        z_text: a mini-batch of text embedding features
        y: a 1D mini-batch of image-text labels 
    """
    return imposter_img_loss(z_image, z_text, y, report_id, similarity_function) + \
           imposter_txt_loss(z_image, z_text, y, report_id, similarity_function)

def imposter_img_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the image is an imposter image chosen from the batch 
    """
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)

    for i in range(batch_size):
        if similarity_function == 'dot':
            paired_similarity = torch.dot(z_image[i], z_text[i])
        if similarity_function == 'cosine':
            paired_similarity = \
                torch.dot(z_image[i], z_text[i])/(torch.norm(z_image[i])*torch.norm(z_text[i]))
        if similarity_function == 'l2':
            paired_similarity = -1*torch.norm(z_image[i]-z_text[i])

        # Select an imposter image index and 
        # compute the maximum margin based on the image label difference
        j = i+1 if i < batch_size - 1 else 0
        if report_id[i] == report_id[j]: 
        # This means the imposter image comes from the same acquisition 
            margin = 0
        elif y[i].item() == -1 or y[j].item() == -1: # '-1' means unlabeled 
            margin = 0.5
        else:
            margin = max(0.5, (y[i] - y[j]).abs().item())

        if similarity_function == 'dot':
            imposter_similarity = torch.dot(z_image[j], z_text[i])
        if similarity_function == 'cosine':
            imposter_similarity = \
                torch.dot(z_image[j], z_text[i])/(torch.norm(z_image[j])*torch.norm(z_text[i]))
        if similarity_function == 'l2':
            imposter_similarity = -1*torch.norm(z_image[j]-z_text[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        if diff_similarity > 0:
            loss = loss + diff_similarity

    return loss / batch_size # 'mean' reduction

def imposter_txt_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the text is an imposter text chosen from the batch 
    """
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)

    for i in range(batch_size):
        if similarity_function == 'dot':
            paired_similarity = torch.dot(z_image[i], z_text[i])
        if similarity_function == 'cosine':
            paired_similarity = \
                torch.dot(z_image[i], z_text[i])/(torch.norm(z_image[i])*torch.norm(z_text[i]))
        if similarity_function == 'l2':
            paired_similarity = -1*torch.norm(z_image[i]-z_text[i])

        # Select an imposter text index and 
        # compute the maximum margin based on the image label difference
        j = i+1 if i < batch_size - 1 else 0
        if report_id[i] == report_id[j]: 
            # This means the imposter report comes from the same acquisition 
            margin = 0
        elif y[i].item() == -1 or y[j].item() == -1: # '-1' means unlabeled
            margin = 0.5
        else:
            margin = max(0.5, (y[i] - y[j]).abs().item())

        if similarity_function == 'dot':
            imposter_similarity = torch.dot(z_text[j], z_image[i])
        if similarity_function == 'cosine':
            imposter_similarity = \
                torch.dot(z_text[j], z_image[i])/(torch.norm(z_text[j])*torch.norm(z_image[i]))
        if similarity_function == 'l2':
            imposter_similarity = -1*torch.norm(z_text[j]-z_image[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        if diff_similarity > 0:
            loss = loss + diff_similarity

    return loss / batch_size # 'mean' reduction

def dot_product_loss(z_image, z_text):
    batch_size = z_image.size(0)
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    for i in range(batch_size):
        loss = loss - torch.dot(z_image[i], z_text[i])
    return loss / batch_size