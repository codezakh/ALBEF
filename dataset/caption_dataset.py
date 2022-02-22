import json
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
from dall_e.utils import map_pixels
from masking_generator import MaskingGenerator


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in tqdm(self.ann):
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(tqdm(self.ann)):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, visual_tokenizer_transform, max_words=30, image_resolution=256):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        # We do this following BEiT:
        # https://sourcegraph.com/github.com/microsoft/unilm/-/blob/beit/datasets.py?L64.
        self.max_words = max_words
        self.patch_transform = transform
        self.visual_token_transform = visual_tokenizer_transform 
        window_size = image_resolution  // 16, image_resolution // 16
        self.masked_position_generator = MaskingGenerator(
            window_size, num_masking_patches=75, max_num_patches=None, min_num_patches=16
        )
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        try: 
            image = Image.open(ann['image']).convert('RGB')   
            image_for_tokenization = self.visual_token_transform(image) 
            image_for_visual_encoder = self.patch_transform(image)
            masked_positions = self.masked_position_generator()
        except:
            return None
        else:
            return image_for_visual_encoder, caption, image_for_tokenization, masked_positions
            

    
