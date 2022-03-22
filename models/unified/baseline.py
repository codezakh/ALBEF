'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from multiprocessing.sharedctypes import Value
from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertForMaskedLM
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

class MIM_Mode(Enum):
    unimodal = 'unimodal'
    multimodal = 'multimodal'


class VisionLanguageLearner(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=1, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(f'missing_keys={msg.missing_keys}\tunexpected_keys={msg.unexpected_keys}')         
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size

        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        # Hardcoded from DALL-E's D-VAE.
        vocab_size = 8192
        self.mim_head = nn.Linear(self.visual_encoder.embed_dim, vocab_size)
        self.mim_mode = MIM_Mode(config['mim_mode'])

        # create momentum models
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        
        self.model_pairs = [
                            [self.text_encoder,self.text_encoder_m],
                           ]
        
        self.copy_params()


    def forward(self, image, text, visual_token_ids, masked_visual_token_pos, masked_visual_tok_labels, alpha=0):
        # get momentum features
        with torch.no_grad():
            self._momentum_update()

        
        ##================= MLM ========================##                
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids, 
                                           attention_mask = text.attention_mask,
                                           return_dict = True,
                                           return_logits = True,
                                           mode='text'
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha,
                                       mode='text'
                                      )                           
        loss_mlm = mlm_output.loss        


        ##================= MIM ========================##
        post_mask_image_embeds = self.visual_encoder(image, masked_visual_token_pos)
        image_atts = torch.ones(post_mask_image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        post_mask_cross_embeds = self.text_encoder.bert(
                        inputs_embeds=post_mask_image_embeds, 
                        attention_mask=image_atts,
                        return_dict=True,
                        mode='text'
                    )
        # Drop the CLS token, because we don't mask it.
        post_mask_cross_embeds = post_mask_cross_embeds.last_hidden_state[:, 1:]
        predicted_visual_tokens = self.mim_head(post_mask_cross_embeds)
        loss_mim = F.cross_entropy(
            input=predicted_visual_tokens[masked_visual_token_pos], 
            target=masked_visual_tok_labels
        )

        return loss_mlm, loss_mim  

        

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

