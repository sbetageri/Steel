import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_pixel_coord(pix, num_rows):
    '''Get coordinates of pixel
    
    :param pix: Pixel value in RLE format
    :type pix: Int
    :param num_rows: Height of image
    :type num_rows: Int
    :return: Co-ordinates of pixel
    :rtype: (Int, Int)
    '''
    row = (pix % num_rows) 
    col = (pix // num_rows)
    return row, col

def get_img_paths(dataframe, train_dir):
    '''Obtain path to all images
    
    :param dataframe: Dataframe containing image paths
    :type dataframe: pd.DataFrame
    :param train_dir: Directory with training images
    :type train_dir: String
    :return: List of image paths
    :rtype: List
    '''
    img_paths = train_dir + dataframe['img_id']
    return list(img_paths)

def get_img_rle(dataframe):
    masks = list(dataframe[['mask_1', 'mask_2', 'mask_3', 'mask_4']].values)
    return masks

def gen_masks():
    root_dir = '/Volumes/Transcend/Data/Steel/'
    csv_file = root_dir + 'train.csv'
    train_dir = root_dir + 'train/'
    mask_dir = root_dir + 'mask/'
    df = get_modified_df(csv_file)
    img_mask = get_img_paths(df, mask_dir)
    for path in img_mask:
        img = Image.open(path)
        img = np.array(img)
        yield tf.convert_to_tensor(img)

def gen_save_masks(img_paths, img_masks, mask_dir):
    for path, rle in tqdm(zip(img_paths, img_masks)):
        mask = get_img_masks(rle)
        path = path.split('/')
        img_id = path[-1]
        mask_path = mask_dir + img_id
        mask_img = Image.fromarray(mask, mode='CMYK')
        mask_img.save(mask_path)

def get_img_masks(masks):
    ## masks is a list of 4 masks
    img_mask = []
    for idx, mask in enumerate(masks):
        if mask != 'nan':
            mask = rle2mask(mask)
        else:
            mask = np.zeros((256, 1600))
        img_mask.append(mask)
    img_mask = np.stack(img_mask)
    img_mask = img_mask.reshape((256, 1600, 4))
    return img_mask

def get_modified_df(csv_file):
    o_df = pd.read_csv(csv_file)
    img_mask = build_img2mask_map(o_df)
    df = img2mask_to_df(img_mask)
    df = df.fillna('nan')
    return df
    
def load_process_gray_img(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [256, 1600])
    image /= 255.0  # normalize to [0,1] range
    return image    

def load_process_img(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 1600])
    image /= 255.0  # normalize to [0,1] range

    return image    

def load_process_mask(path):
    # image = Image.open(path)
    # image = np.array(image)
    # image = tf.convert_to_tensor(image)
    print(path)
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=4)
    image = tf.image.resize(image, [256, 1600])
    image /= 255.0
    return image

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

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    score = (2. * tf.keras.backend.sum(intersection)) 
    score = score / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return tf.keras.backend.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice(y_true, y_pred):
    smooth = 1
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

def gen_dataset():
    root_dir = '/home/srirakshith/Data/steel/'
    csv_file = root_dir + 'train.csv'
    train_dir = root_dir + 'train/'
    mask_dir = root_dir + 'mask/'
    df = get_modified_df(csv_file)
    img_masks = get_img_paths(df, mask_dir)
    img_paths = get_img_paths(df, train_dir)
    for img_path, img_mask in zip(img_paths, img_masks):
        img = Image.open(img_path)
        img = img.resize((400, 64))
        img = np.array(img)
        img = np.true_divide(img, 255.0)
        mask = Image.open(img_mask)
        mask = mask.resize((400, 64))
        mask = np.array(mask)
        # mask = np.true_divide(mask, 255.0)
        # yield img, mask
        yield tf.convert_to_tensor(img), tf.convert_to_tensor(mask)

def get_dataset(csv_file, train_dir, mask_dir):
    '''Build dataset of images and masks
    
    :param csv_file: Path to csv file with RLE labels
    :type csv_file: String
    :param train_dir: Path to train dir
    :type train_dir: String
    :param mask_dir: Path to mask dir
    :type mask_dir: String
    :return: Dataset of image and mask
    :rtype: tf.data.Dataset
    '''
    df = get_modified_df(csv_file)
    all_img_paths = get_img_paths(df, train_dir)
    all_img_mask_path = get_img_paths(df, mask_dir)
    image_ds = tf.data.Dataset.from_tensor_slices(all_img_paths)
    image_ds = image_ds.map(load_process_img)
    # mask_ds = tf.data.Dataset.from_tensor_slices(all_img_mask_path)
    # mask_ds = mask_ds.map(load_process_mask)
    mask_ds = tf.data.Dataset.from_generator(gen_masks, (tf.float32), tf.TensorShape([256, 1600, 4]))
    image_mask_ds = tf.data.Dataset.zip((image_ds, mask_ds))
    return image_mask_ds

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
        print('Usage : $ python dataset.py gen_masks <dataset_root_path>')
    else:
        if sys.argv[1] == 'stats':
            path = sys.argv[2]
            get_all_image_statistics(path)
            
        elif sys.argv[1] == 'gen_masks':
            root_dir = sys.argv[2] + '/'
            csv_file = root_dir + 'train.csv'
            train_dir = root_dir + 'train/'
            mask_dir = root_dir + 'mask/'
            
            print(root_dir)

            df = get_modified_df(csv_file)
            all_img_paths = get_img_paths(df, train_dir)
            all_img_masks = get_img_rle(df)
            gen_save_masks(all_img_paths, all_img_masks, mask_dir)