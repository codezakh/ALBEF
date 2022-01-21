import json
import argparse

def convert_coco_to_albef_json(coco_json):
	image_filename_by_id = {_['id']: _['file_name'] for _ in coco_json['images']}
	val_pairs = []
	for annotation in coco_json['annotations']:
		val_pairs.append({
			'image': image_filename_by_id[annotation['image_id']],
			'image_id': annotation['image_id'],
			'caption': annotation['caption']
		})
	return val_pairs 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-json', type=str, required=True)
    parser.add_argument('--albef-json', type=str, required=True)
    args = parser.parse_args()

    with open(args.coco_json, 'r') as f:
        coco_json = json.load(f)

    albef_json = convert_coco_to_albef_json(coco_json)

    with open(args.albef_json, 'w') as f:
        json.dump(albef_json, f, indent=2)
