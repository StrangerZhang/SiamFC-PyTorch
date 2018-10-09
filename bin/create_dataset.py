import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

from siamfc import config, get_instance_image

def worker(output_dir, video_dir):
    image_names = glob(os.path.join(video_dir, '*.JPEG'))
    image_names = sorted(image_names,
                        key=lambda x:int(x.split('/')[-1].split('.')[0]))
    video_name = video_dir.split('/')[-1]
    save_folder = os.path.join(output_dir, video_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    trajs = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        anno_name = image_name.replace('Data', 'Annotations')
        anno_name = anno_name.replace('JPEG', 'xml')
        tree = ET.parse(anno_name)
        root = tree.getroot()
        bboxes = []
        filename = root.find('filename').text
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)
            if trkid in trajs:
                trajs[trkid].append(filename)
            else:
                trajs[trkid] = [filename]
            instance_img, _, _ = get_instance_image(img, bbox,
                    config.exemplar_size, config.instance_size, config.context_amount, img_mean)
            instance_img_name = os.path.join(save_folder, filename+".{:02d}.x.jpg".format(trkid))
            cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs

def processing(data_dir, output_dir, num_threads):
    # get all 4417 videos
    video_dir = os.path.join(data_dir, 'Data/VID')
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))
    meta_data = []
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(
            functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            meta_data.append(ret)

    # save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    Fire(processing)

