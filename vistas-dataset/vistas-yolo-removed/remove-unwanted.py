import os
import json
import yaml
from tqdm import tqdm
import yaml

original_path = '../vistas-yolo/labels'
target_path = 'labels'

with open('vistas_keep.json', 'r') as f:
    keep = json.load(f)

with open('../vistas-yolo/data-vistas.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    sid_to_name = data['names']

new_name_to_sid = {}
for cname in keep:
    if keep[cname]:
        new_name_to_sid[cname] = len(new_name_to_sid)

new_sid_to_name = {v: k for k, v in new_name_to_sid.items()}

print(new_name_to_sid)

def remove_unwanted(mode):
    global keep

    original_label_path = os.path.join(original_path, mode)
    target_label_path = os.path.join(target_path, mode)

    all_labels = os.listdir(original_label_path)

    for label_name in tqdm(all_labels):
        label_path = os.path.join(original_label_path, label_name)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        output_path = os.path.join(target_label_path, label_name)
        new_lines = []
        for line in lines:
            line = line.strip()
            label_id, *points = line.split()
            label_name = sid_to_name[int(label_id)]

            if keep[label_name]:
                new_label_id = new_name_to_sid[label_name]
                new_line = f'{new_label_id} {" ".join(points)}'
                new_lines.append(new_line)
            
        if new_lines:
            with open(output_path, 'w') as f:
                f.write('\n'.join(new_lines))

def dump_yaml():
    global new_sid_to_name
    todump = {
        'path': '../vistas-removed',
        'train': 'images/train',
        'val': 'images/val',
        'names': new_sid_to_name,
    }
    with open('data-vistas-removed.yaml', 'w') as f:
        yaml.dump(todump, f)


def main():
    for mode in ['train', 'val']:
        remove_unwanted(mode)
    dump_yaml()

if __name__ == '__main__':
    main()