import tensorflow as tf
import pandas as pd
import numpy as np
import data

from tqdm import tqdm

from PIL import Image

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_file, img_dir, is_test=False, gen_masks=False):
        '''
        Initialises the dataset
        :param csv_file: Path to csv_file.
        :param img_dir: Image directory.
        :param batch_size: Size of each batch.
        :param is_test: Boolean value, is the dataset a test set.
        '''
        super(DataGenerator, self).__init__()
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_test = is_test
        self.orig_shape = (1600, 256)
        if gen_masks == True:
            for i in tqdm(range(len(self.df))):
                self._save_mask(i)


    def __len__(self):
        '''
        Length of the dataset
        :return: int
        '''
        return len(self.df)

    def get_img_path(self, idx):
        '''
        Obtain path to image from dataframe
        :param idx: Index to entry in csv
        :return: Image path
        '''
        img_id = self.df.iloc[idx]['img_id']
        img_path = self.img_dir + img_id
        return img_path

    def get_image(self, path):
        img = Image.open(path)
        np_img = np.array(img)
        np_img = np.true_divide(np_img, 255.0)
        return np_img

    def mask2rle(img):
        ## Taken from https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle2mask(self, mask_rle, shape=(1600, 256)):
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
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def split(self, array):
        '''
        Split an array into 4 equal parts
        :param array: np array to be split.
        :return: List of smaller equal sized np arrays.
        '''
        arr = np.array(array)
        arrays = np.split(arr, 4, axis=1)
        return arrays

    def get_img_masks(self, idx):
        data_point = self.df.iloc[idx]
        mask1 = data_point['mask_1']
        mask2 = data_point['mask_2']
        mask3 = data_point['mask_3']
        mask4 = data_point['mask_4']

        masks = []
        if type(mask1) == 'str':
            masks.append(self.rle2mask(mask1))
        else:
            masks.append(np.zeros(self.orig_shape))

        if type(mask2) == 'str':
                masks.append(self.rle2mask(mask2))
        else:
            masks.append(np.zeros(self.orig_shape))

        if type(mask3) == 'str':
                masks.append(self.rle2mask(mask3))
        else:
            masks.append(np.zeros(self.orig_shape))

        if type(mask4) == 'str':
            masks.append(self.rle2mask(mask4))
        else:
            masks.append(np.zeros(self.orig_shape))

        masks = np.array(masks)
        masks = masks.reshape(256, 1600, 4)
        return masks

    def _save_mask(self, idx):
        ## get img id
        img_id = self.df.iloc[idx]['img_id']
        img_id = ''.join(img_id.split('.')[:-1])
        masks = self.get_img_masks(idx)
        mask_path = data.train_mask_dir + img_id
        np.save(mask_path, masks)

    def __getitem__(self, idx):
        '''
        Obtain the split of images and masks
        :param idx: Index to image
        :return: Training images and labels
        '''

        assert False
        img_path = self.get_img_path(idx)

        ## img is an np array
        img = self.get_image(img_path)

        ## masks is an np array of shape = (1600, 256, 4)
        img_masks = self.get_img_masks(idx)

        split_imgs = self.split(img)
        split_masks = self.split(img_masks)

        # train_imgs = self.gen_train_data(split_imgs)
        # train_masks = self.get_train_data(split_masks)

        t_imgs = tf.convert_to_tensor(split_imgs)
        t_imgs = tf.image.per_image_standardization(t_imgs)

        t_masks = tf.convert_to_tensor(split_masks)

        return t_imgs, t_masks
