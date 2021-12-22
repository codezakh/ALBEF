from transformers import BertModel, AutoTokenizer
from models.vit import VisionTransformer
import torch

def merge_words(lm_words, vm_words):
    # The last dimension of each input should be the same.
    # There are a couple of details to describe here.
    # First, the positional embeddings are taken care
    # of by the language model. For this to work, a
    # critical step is to put the LM in sentence-pair
    # mode correctly. The last token in lm_words should
    # be the [SEP] token.

    bs, *, dim = lm_words.shape

    merged = torch.cat([lm_words, vm_words], axis=0)
    return merged

lm = BertModel('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') 

vm = VisionTransformer()

text = "this is a random image"
tokens = torch.Tensor(tokenizer.encode(text)).long()
# Will produce a sequence of shape B * L * D.
# B is the batch size, L is the sequence length (512 max)
# and D is the embedding dimension (768 in BERT).
lm_words = lm.embedding(tokens.unsqueeze(0))

image = torch.randn(3, 224, 224).unsqueeze(0)
# Will produce a sequence of shape B * L * D.
# B is the batch size, L is the sequence length (197 by default)
# and D is the embedding dimension (768 by default).
vm_words = vm(image) 



