import sys

import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_pixel_coord(pix, num_rows):
    row = (pix % num_rows) 
    col = (pix // num_rows)
    return row, col

def generate_mask(rle, height=256, width=1600):
    mask = np.zeros((height, width), dtype=np.uint8)
    rle = [int(val) for val in rle.split(' ')]
    iter_rle = iter(rle)
    for val in iter_rle:
        start_pixel = val
        length = next(iter_rle)
        for i in range(length):
            pix = start_pixel - 1 + i
            h, w = get_pixel_coord(pix, height)
            mask[h, w] = 1
    return mask

def build_img2mask_map(old_df):
    img2mask = {}
    for i in tqdm(range(len(old_df))):
        point = old_df.iloc[i]
        img, label = point['ImageId_ClassId'].split('_')
        label = int(label) - 1
        if img in img2mask:
            val[label] = point['EncodedPixels']
        else:
            val = [-1] * 4
            val[label] = point['EncodedPixels']
            img2mask[img] = val
    return img2mask

def img2mask_to_df(img2mask):
    new_df = pd.DataFrame()
    values = []
    count = 0
    for img in img2mask:
        values.append([img, *img2mask[img]])
    new_df = new_df.append(values)
    new_df = new_df.rename(columns={0: 'img_id', 1: 'mask_1', 2: 'mask_2', 3: 'mask_3', 4: 'mask_4'})
    return new_df

def mask2rle(img):
    ## Taken from https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    ## Taken from https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

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