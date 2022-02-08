import json
from pathlib import Path
from tqdm import tqdm

karpathy_root =  Path('/net/acadia10a/data/zkhan/karpathy-flickr30k/')
flickr_root = Path('/net/acadia10a/data/zkhan/flickr30k/')
# These were downloaded from Kaggle.
flickr_images = flickr_root / 'flickr30k-images'

if not flickr_root.exists():
    flickr_root.mkdir()

with open(karpathy_root / 'dataset.json', 'r') as f:
    karpathy = json.load(f)

pairs = []
for image in karpathy['images']:
    filename = image['filename']
    split = image['split']
    captions = [sentence['raw'] for sentence in image['sentences']]
    image_id = image['imgid']
    # We only need the test pairs for the evaluation, because Flickr30K
    # is used for zero-shot in the ALBEF paper.
    if split == 'test':
        pairs.append(dict(image=filename, caption=captions, image_id=image_id))

with open(flickr_root / 'flickr30-test-pairs.json', 'w') as f:
    json.dump(pairs, f)

for record in tqdm(pairs):
    assert (flickr_images / record['image']).exists()