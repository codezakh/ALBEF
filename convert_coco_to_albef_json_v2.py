import json
import argparse

def convert_coco_to_albef_json(coco_json):
    records = {
        _['id']: {
            'image': _['file_name'],
            'image_id': _['id'],
            'caption': []
        } 
        for _ in coco_json['images']
    }
    for annotation in coco_json['annotations']:
        # Look up the image_id the annotation belongs to
        image_id = annotation['image_id']
        # Append the annotation to the list of annotations for the image.
        records[image_id]['caption'].append(annotation['caption'])
    return [_ for _ in records.values()]


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
