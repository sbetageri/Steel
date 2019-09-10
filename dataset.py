import sys

import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_all_image_statistics(image_dir):
    ## Get the dimensions of all the images
    min_h, min_w, min_c = 1000000, 1000000, 1000000
    max_h, max_w, max_c = -1, -1, -1
    avg_h, avg_w, avg_c = 0, 0, 0
    img_dir = Path(image_dir)
    count = 0
    for idx, img_path in tqdm(enumerate(img_dir.glob('*.jpg'))):
        count += 1
        idx += 1
        img = Image.open(img_path)
        np_img = np.array(img)
        h, w, c = np_img.shape
        
        if h < min_h:
            min_h = h
        
        if w < min_w:
            min_w = w

        if c < min_c:
            min_c = c
        
        if h > max_h:
            max_h = h

        if w > max_w:
            max_w = w

        if c > max_c:
            max_c = c
        
        avg_h = (avg_h * (count - 1) + h) / count
        avg_w = (avg_w * (count - 1) + w) / count
        avg_c = (avg_c * (count - 1) + c) / count
        
    print('Number of images : ', idx)
    
    print('###############################')
    print('Average stats')
    print('Avg H : ', avg_h)
    print('Avg W : ', avg_w)
    print('Avg C : ', avg_c)
    
    
    print('###############################')
    print('Min stats')
    print('Min H : ', min_h)
    print('Min W : ', min_w)
    print('Min C : ', min_c)
        
    print('###############################')
    print('Max stats')
    print('Max H : ', max_h)
    print('Max W : ', max_w)
    print('Max C : ', max_c)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Incorrect Usage')
        print('Usage : $ python dataset.py stats <img_dir_path>')
        print('<img_dir_path> is the path to the directory containing the images')
    else:
        if sys.argv[1] == 'stats':
            path = sys.argv[2]
            get_all_image_statistics(path)