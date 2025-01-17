from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
# from models.xbert import BertConfig, BertForMaskedLM
from models.xbert import BertConfig, BertModel, BertForMaskedLM
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder: str = None,
                 tokenizer = None,
                 config: Dict = None,    
                 temp: float = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        # vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        # self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      
        # for param in self.text_encoder.embeddings.word_embeddings.parameters():
        #     param.requires_grad = False

        text_width = self.text_encoder.config.hidden_size
        # self.vision_proj = nn.Linear(vision_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)         

        # self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)     

        # create momentum models
        # self.visual_encoder_m = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        # self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        # self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
        # self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        # self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
        #                     [self.vision_proj,self.vision_proj_m],
        #                     [self.text_encoder,self.text_encoder_m],
        #                     [self.text_proj,self.text_proj_m],
                        #    ]
        
        # self.copy_params()

        # create the queue
        # self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        # self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


    def make_sentence_pair(self, text_token_ids, text_attn_mask, image_embeds, image_atts, device):
        text_token_ids = text_token_ids.clone()
        with torch.no_grad():
            text_token_ids[:, 0] = self.tokenizer.sep_token_id 
        # Create the [CLS] prefix for the visual token. 
        # prefix = torch.zeros(image_embeds.shape[0], 1).to(image.device) * self.tokenizer.cls_token_id
        # prefix = prefix.long()
        # prefix_embeds = self.text_encoder.bert.embeddings.word_embeddings(prefix)
        # Get the word embeddings for language.
        word_embeddings = self.text_encoder.bert.embeddings.word_embeddings(text_token_ids)
        # Concatenate it all to make the input sentence.
        mm_model_input = torch.cat([image_embeds, word_embeddings], dim=1)
        # Create the attention mask for the combined sentence.
        imtext_attention_mask = torch.cat([image_atts, text_attn_mask], dim=1) 
        # Get the token_type_ids.
        # Following the BERT convention, the token_type_ids for the first sentence is 0,
        # and the second sentence is 1. To achieve this, we can simply concatenate the attention mask
        # of the text with a zero tensor.
        text_token_type_ids = text_attn_mask.clone()
        with torch.no_grad():
            text_token_type_ids[:, 0] = 0 # the [SEP] between the sentences is considered as sentence B.
        token_type_ids = torch.cat([torch.zeros_like(image_atts).to(device), text_token_type_ids], dim=1)
        return mm_model_input, imtext_attention_mask, token_type_ids


    def forward(self, image, text, alpha=0):
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        mm_pos_words, mm_pos_att_mask, mm_pos_token_type_ids = self.make_sentence_pair(
            text.input_ids,
            text.attention_mask,
            image_embeds,
            image_atts,
            image.device
        )
        output_pos = self.text_encoder.bert(
            inputs_embeds=mm_pos_words,
            attention_mask=mm_pos_att_mask, 
            token_type_ids=mm_pos_token_type_ids,
            return_dict = True,
            mode = 'text'
        )            

        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = torch.ones(bs, bs).to(image.device)
            weights_t2i = torch.ones(bs, bs).to(image.device)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_tokens_neg = []
        text_att_masks_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_tokens_neg.append(text.input_ids[neg_idx])
            text_att_masks_neg.append(text.attention_mask[neg_idx])
        text_tokens_neg = torch.stack(text_tokens_neg,dim=0)
        text_att_masks_neg = torch.stack(text_att_masks_neg,dim=0)

        text_tokens_all = torch.cat([text.input_ids, text_tokens_neg],dim=0)
        text_att_masks_all = torch.cat([text.attention_mask, text_att_masks_neg],dim=0)

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        mm_neg_words, mm_neg_att_mask, mm_neg_token_type_ids = self.make_sentence_pair(
            text_tokens_all,
            text_att_masks_all,
            image_embeds_all,
            image_atts_all,
            image.device
        )

        output_neg= self.text_encoder.bert(
            inputs_embeds=mm_neg_words,
            attention_mask=mm_neg_att_mask, 
            token_type_ids=mm_neg_token_type_ids,
            return_dict = True,
            mode = 'text'
        )            

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     

        ##================= MLM ========================##                
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        with torch.no_grad():
            logits_m = self.text_encoder(input_ids, 
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss

        return loss_itm, loss_mlm

        

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