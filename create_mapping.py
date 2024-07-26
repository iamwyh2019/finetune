import json
import yaml
from typing import Dict, Union, List
import os
from tqdm import tqdm

with open('config_v2.0.json') as f:
    metadata = json.load(f)

labels = metadata['labels']
vistas_name_to_id = {label['readable']: i for i, label in enumerate(labels)}
vistas_raw_name_to_id = {label['name']: i for i, label in enumerate(labels)}

with open('data-yolo.yaml') as f:
    # read in the existing file
    text = f.read()
    data_yaml = yaml.load(text, Loader=yaml.FullLoader)

with open('data-vistas.yaml') as f:
    # read in the existing file
    text = f.read()
    data_vistas = yaml.load(text, Loader=yaml.FullLoader)

coco_names: Dict[int, str] = data_yaml['names']
coco_names_inv: Dict[str, int] = {v: k for k, v in coco_names.items()}

with open("vistas_to_coco_mapping.json") as f:
    vistas_to_coco = json.load(f)

coco_to_vistas: Dict[int, Union[None, List[int]]] = {k: None for k in coco_names}

for vistas_name, coco_val in vistas_to_coco.items():
    vistas_id = vistas_name_to_id[vistas_name]
    # first case: coco_val is None
    if coco_val is None:
        continue

    def add(coco_name, vistas_id):
        coco_id = coco_names_inv[coco_name]
        if coco_to_vistas[coco_id] is None:
            coco_to_vistas[coco_id] = [vistas_id]
        else:
            coco_to_vistas[coco_id].append(vistas_id)

    # second case: coco_val is a str
    if isinstance(coco_val, str):
        add(coco_val, vistas_id)

    # third case: coco_val is a list of str
    elif isinstance(coco_val, list):
        for coco_name in coco_val:
            add(coco_name, vistas_id)

# remove all pairs with None value
coco_to_vistas = {k: v for k, v in coco_to_vistas.items() if v is not None}


with open('vistas_sim.json') as f:
    vistas_sim = json.load(f)
simplified_ids: Dict[int, int] = {}
simplified_name_to_id: Dict[str, int] = {}
for vistas_name, sim_name in vistas_sim.items():
    vistas_id = vistas_raw_name_to_id[vistas_name]
    if sim_name is None:
        simplified_ids[vistas_id] = None
        continue
    if sim_name not in simplified_name_to_id:
        simplified_name_to_id[sim_name] = len(simplified_name_to_id)
    simplified_id = simplified_name_to_id[sim_name]
    simplified_ids[vistas_id] = simplified_id
simplified_names: Dict[int, str] = {v: k for k, v in simplified_name_to_id.items()}


data_path = '/datastore1/visual-saliency-copy/Backend/yolo/vistas-simplified/labels'

def update_label(label_path, mode):
    true_path = os.path.join(label_path, mode)
    files = os.listdir(true_path)
    for file in tqdm(files):
        filepath = os.path.join(true_path, file)
        with open(filepath) as f:
            lines = f.readlines()

        goodlines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line == '':
                continue
            line = line.split()
            vid = int(line[0])
            
            # if it's None, skip this row
            if not simplified_ids[vid]:
                continue
            line[0] = str(simplified_ids[vid])
            lines[i] = ' '.join(line) + '\n'
            goodlines.append(lines[i])

        if len(goodlines) == 0:
            # delete the image and label
            os.remove(filepath)
            os.remove(filepath.replace('labels', 'images').replace('txt', 'jpg'))
            print(f"Deleted {filepath}")
            continue

        with open(filepath, 'w') as f:
            f.writelines(goodlines)

def update_yaml():
    output = {
        'path': '../vistas-simplified',
        'train': 'images/train',
        'val': 'images/val',
        'names': simplified_names,
    }
    with open('data-vistas-simplified.yaml', 'w') as f:
        yaml.dump(output, f)


def main():
    global labels, vistas_to_coco, coco_names_inv

    pth = '/datastore1/visual-saliency-copy/Backend/yolo/vistas-yolo/labels'
    src = 'val'
    dst = 'val-coco'

    src_path = os.path.join(pth, src)
    dst_path = os.path.join(pth, dst)

    src_all = os.listdir(src_path)
    for src_file in src_all:
        with open(os.path.join(src_path, src_file)) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line == '':
                continue
            line = line.split()
            vid = int(line[0])
            vname = labels[vid]['readable']
            cname = vistas_to_coco[vname]
            cid = coco_names_inv.get(cname, 80)
            line[0] = str(cid)
            lines[i] = ' '.join(line) + '\n'
        with open(os.path.join(dst_path, src_file), 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    # main()
    # print(simplified_ids)
    update_label(data_path, 'train')
    update_label(data_path, 'val')
    update_yaml()
    