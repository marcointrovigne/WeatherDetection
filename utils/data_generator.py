import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

"""
Build the generator in order to train our model
"""


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, height, width, batch_size, root_path, df, img_generator=None):

        """
        :param height: resize image dimension
        :param width: resize image dimension
        :param batch_size: value of the batch
        :param root_path: directory to Algolux_allv3
        :param df: use train, validation or test split dataframe
        :param img_generator: if you want to use data_augmentation
        """

        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.root_path = root_path
        self.df = df
        self.img_generator = img_generator

    def generate_images(self, for_training=True):

        """
        Single image generator
        :param for_training: True for training and validation, set False for test
        """

        images, daytimes, precipitations, fogs, roadStates, sidewalkStates, infrastructures = [], [], [], [], \
                                                                                              [], [], []

        while True:
            for i in range(len(self.df)):
                r = self.df.iloc[i]

                daytime = r['daytime']
                precipitation = r["precipitation"]
                fog = r['fog']
                roadState = r['roadState']
                sidewalkState = r['sidewalkState']
                infrastructure = r["infrastructure"]

                filename = os.path.join(self.root_path, 'cam_stereo_left_lut', r['Filename'])
                im_arr = cv2.imread(filename)
                im_arr = cv2.resize(im_arr, (self.width, self.height))

                if self.img_generator is not None:
                    im_t = self.img_generator.get_random_transform(im_arr.shape)
                    im_arr = self.img_generator.apply_transform(im_arr, im_t)
                    im_arr = im_arr.astype(float) / 255
                else:
                    im_arr = im_arr.astype(float) / 255

                images.append(im_arr)
                daytimes.append(to_categorical(daytime, 2))
                precipitations.append(to_categorical(precipitation, 4))
                fogs.append(to_categorical(fog, 3))
                roadStates.append(to_categorical(roadState, 4))
                sidewalkStates.append(to_categorical(sidewalkState, 3))
                infrastructures.append(to_categorical(infrastructure, 3))

                if len(images) >= self.batch_size:
                    yield np.array(images), [np.array(daytimes),
                                             np.array(precipitations),
                                             np.array(fogs),
                                             np.array(roadStates),
                                             np.array(sidewalkStates),
                                             np.array(infrastructures)]
                    images, daytimes, precipitations, fogs, roadStates, sidewalkStates, infrastructures = [], [], [], [], [], [], []

            if not for_training:
                break

    def generate_images_concatenated(self, for_training=True, temporal_length=5):

        """
        oncatenated image generator
        :param for_training: True for training and validation, set False for test
        :param temporal_length: Set frame length
        """

        seq_images, daytimes, precipitations, fogs, roadStates, sidewalkStates, infrastructures = [], [], [], [], \
                                                                                                  [], [], []
        while True:
            for i in range(len(self.df)):
                r = self.df.iloc[i]
                samples = []

                daytime = r['daytime']
                precipitation = r["precipitation"]
                fog = r['fog']
                roadState = r['roadState']
                sidewalkState = r['sidewalkState']
                infrastructure = r["infrastructure"]

                # Change number inside this function in order to change length of sequence
                for p in [0, 1, 2, 3, 4]:
                    if p == 0:
                        filename = os.path.join(self.root_path, 'cam_stereo_left_lut', r['Filename'])
                    else:
                        folder = self.root_path + '/cam_stereo_left_lut_history_{}/'.format(p)
                        filename = folder + r["Filename"]

                    img = cv2.imread(filename)
                    img = cv2.resize(img, (self.width, self.height))
                    img = img.astype(float) / 255
                    samples.append(img)

                    if len(samples) == temporal_length:
                        samples_c = samples

                img = np.concatenate([samples_c[0], samples_c[1], samples_c[2], samples_c[3], samples_c[4]],
                                     axis=-1)
                seq_images.append(img)
                daytimes.append(to_categorical(daytime, 2))
                precipitations.append(to_categorical(precipitation, 4))
                fogs.append(to_categorical(fog, 3))
                roadStates.append(to_categorical(roadState, 4))
                sidewalkStates.append(to_categorical(sidewalkState, 3))
                infrastructures.append(to_categorical(infrastructure, 3))

                if len(seq_images) >= self.batch_size:
                    yield np.array(seq_images), [np.array(daytimes),
                                                 np.array(precipitations),
                                                 np.array(fogs),
                                                 np.array(roadStates),
                                                 np.array(sidewalkStates),
                                                 np.array(infrastructures)]
                    seq_images, daytimes, precipitations, fogs, roadStates, sidewalkStates, infrastructures = [], [], [], [], [], [], []

            if not for_training:
                break
