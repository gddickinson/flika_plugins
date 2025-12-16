#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:58:46 2024

@author: george
"""

# Import necessary modules
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from distutils.version import StrictVersion
import flika
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from skimage import filters
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.morphology import disk, ball

# Check Flika version for appropriate imports
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui

from flika.process.file_ import open_file
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

#to get tensorflow GPU Mac M1 compatibility pip install tensorflow-macos AND pip install tensorflow-metal
import os, shutil
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

import glob

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deleteFolderFiles(folder = None):
    ''''Helper function to delete contents of a folder'''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def custom_binary_crossentropy(y_true, y_pred):
    ''''Custom binary cross entrop loss function'''
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    return -K.mean(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))


class UNetClassifier:
    def __init__(self, img_height=128, img_width=128, img_channels=1, num_test_images=10, use_gpu=True):
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.IMG_CHANNELS = img_channels
        self.NUM_TEST_IMAGES = num_test_images
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.df_test = None

        # Set up GPU or CPU
        if use_gpu:
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("Using GPU for computations")
            else:
                print("No GPU found. Falling back to CPU")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Using CPU for computations")

    def load_data(self, input_dir, images_dir, masks_dir):
        os.chdir(input_dir)

        img_list = os.listdir(images_dir)
        mask_list = os.listdir(masks_dir)

        df_images = pd.DataFrame(img_list, columns=['image_id'])
        df_images = df_images[df_images['image_id'] != '.htaccess']

        df_images['num_cells'] = df_images['image_id'].apply(self.get_num_cells)
        df_images['has_mask'] = df_images['image_id'].apply(lambda x: 'yes' if x in mask_list else 'no')
        df_images['blur_amt'] = df_images['image_id'].apply(self.get_blur_amt)

        # Create a new DataFrame instead of a view
        df_masks = df_images[df_images['has_mask'] == 'yes'].copy()
        df_masks['mask_id'] = df_masks['image_id']

        self.df_test = df_masks.sample(self.NUM_TEST_IMAGES, random_state=101).reset_index(drop=True)
        test_images_list = list(self.df_test['image_id'])
        df_masks = df_masks[~df_masks['image_id'].isin(test_images_list)]

        self.prepare_data(df_masks, images_dir, masks_dir)

    def get_num_cells(self, x):
        return int(x.split('_')[2][1:])

    def get_blur_amt(self, x):
        return int(x.split('_')[3][1:])

    def prepare_data(self, df_masks, images_dir, masks_dir):
        image_id_list = list(df_masks['image_id'])
        mask_id_list = list(df_masks['mask_id'])
        test_id_list = list(self.df_test['image_id'])

        self.X_train = np.zeros((len(image_id_list), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        self.Y_train = np.zeros((len(image_id_list), self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=bool)
        self.X_test = np.zeros((self.NUM_TEST_IMAGES, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)

        for i, image_id in enumerate(image_id_list):
            self.X_train[i] = self.process_image(os.path.join(images_dir, image_id))

        for i, mask_id in enumerate(mask_id_list):
            self.Y_train[i] = self.process_image(os.path.join(masks_dir, mask_id))

        for i, image_id in enumerate(test_id_list):
            self.X_test[i] = self.process_image(os.path.join(images_dir, image_id))

    def process_image(self, image_path):
        image = imread(image_path)
        image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
        return np.expand_dims(image, axis=-1)

    def build_model(self):
        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = Lambda(lambda x: x / 255)(inputs)

        # Encoder (Contracting Path)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        # Decoder (Expanding Path)
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        #self.model.compile( optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),loss='binary_crossentropy')

    def train_model(self, epochs=50, batch_size=16, validation_split=0.1, save_weights_path=None):
        filepath = "model.keras"
        callbacks_list = [
            EarlyStopping(patience=5, verbose=1),
            ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        ]

        history = self.model.fit(
            self.X_train, self.Y_train,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list
        )

        if save_weights_path:
            self.save_weights(save_weights_path)

        return history

    def predict(self):
        self.model.load_weights('model.keras')
        test_preds = self.model.predict(self.X_test)
        return (test_preds >= 0.5).astype(np.uint8)

    def visualize_results(self, preds_test_thresh, masks_dir, num_samples=3):
        plt.figure(figsize=(10, 10))
        plt.axis('Off')

        for i in range(num_samples):
            # Display test image
            plt.subplot(3, 3, i*3 + 1)
            plt.imshow(self.X_test[i, :, :, 0])
            plt.title('Test Image', fontsize=14)
            plt.axis('off')

            # Display true mask
            plt.subplot(3, 3, i*3 + 2)
            mask_id = self.df_test.loc[i, 'mask_id']
            mask = self.process_image(os.path.join(masks_dir, mask_id))
            plt.imshow(mask[:,:,0], cmap='gray')
            plt.title('True Mask', fontsize=14)
            plt.axis('off')

            # Display predicted mask
            plt.subplot(3, 3, i*3 + 3)
            plt.imshow(preds_test_thresh[i, :, :, 0], cmap='gray')
            plt.title('Pred Mask', fontsize=14)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_training_loss(self, history):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")
        print(self.model.summary())

    def save_weights(self, filepath):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")

        try:
            self.model.save_weights(filepath)
            print(f"Weights saved successfully to {filepath}")
        except Exception as e:
            print(f"Failed to save weights to {filepath}. Error: {str(e)}")

    def load_pretrained_weights(self, weights_path):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")

        try:
            self.model.load_weights(weights_path)
            print(f"Weights loaded successfully from {weights_path}")
        except:
            print(f"Failed to load weights from {weights_path}. Check if the file exists and the architecture matches.")


    def get_intensity_range(self, input_folder):
        """
        Calculate the global minimum and maximum intensity values across all images.

        :param input_folder: Path to the folder containing input images
        :return: Tuple of (min_intensity, max_intensity)
        """
        min_intensity = float('inf')
        max_intensity = float('-inf')

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

        for image_file in tqdm(image_files, desc="Calculating intensity range"):
            try:
                image_path = os.path.join(input_folder, image_file)
                image = imread(image_path).astype(np.float32)

                min_intensity = min(min_intensity, np.min(image))
                max_intensity = max(max_intensity, np.max(image))

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

        return min_intensity, max_intensity

    def normalize_image(self, image, min_intensity, max_intensity):
        """
        Normalize image to 0-255 range based on global min and max intensities.

        :param image: Input image
        :param min_intensity: Global minimum intensity
        :param max_intensity: Global maximum intensity
        :return: Normalized image
        """
        normalized = (image - min_intensity) / (max_intensity - min_intensity)
        return (normalized * 255).astype(np.uint8)

    def predict_images_noNormalization(self, input_folder, output_folder):
        """
        Predict segmentation for 128x128 pixel TIFF images in a folder.

        :param input_folder: Path to the folder containing input images
        :param output_folder: Path to save the predicted segmentation masks
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

        if not image_files:
            print(f"No TIFF images found in {input_folder}")
            return

        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                image_path = os.path.join(input_folder, image_file)
                image = imread(image_path)

                if image.shape[:2] != (self.IMG_HEIGHT, self.IMG_WIDTH):
                    print(f"Warning: {image_file} is not 128x128. Resizing.")
                    image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)

                image = np.expand_dims(image, axis=-1)
                image = np.expand_dims(image, axis=0)

                prediction = self.model.predict(image)
                prediction = (prediction > 0.5).astype(np.uint8) * 255

                output_path = os.path.join(output_folder, f"pred_{image_file}")
                imsave(output_path, prediction[0, ..., 0])

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

    def predict_images(self, input_folder, output_folder):
        """
        Predict segmentation for 128x128 pixel TIFF images in a folder.

        :param input_folder: Path to the folder containing input images
        :param output_folder: Path to save the predicted segmentation masks
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

        if not image_files:
            print(f"No TIFF images found in {input_folder}")
            return

        min_intensity, max_intensity = self.get_intensity_range(input_folder)
        print(f"Global intensity range: [{min_intensity}, {max_intensity}]")

        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                image_path = os.path.join(input_folder, image_file)
                image = imread(image_path).astype(np.float32)

                image = self.normalize_image(image, min_intensity, max_intensity)

                if image.shape[:2] != (self.IMG_HEIGHT, self.IMG_WIDTH):
                    print(f"Warning: {image_file} is not {self.IMG_HEIGHT}x{self.IMG_WIDTH}. Resizing.")
                    image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                    #print(f"New size: {image.shape[:2]}.")

                image = np.expand_dims(image, axis=-1)
                image = np.expand_dims(image, axis=0)

                prediction = self.model.predict(image)
                #prediction = (prediction > 0.5).astype(np.uint8) * 255

                output_path = os.path.join(output_folder, f"pred_{image_file}")
                imsave(output_path, prediction[0, ..., 0])

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

    def predict_array(self, test_array, output_folder):
        """
        Predict segmentation for 128x128 pixel TIFF stack.

        :param test_array

        """

        min_intensity = np.min(test_array)
        max_intensity = np.max(test_array)
        frames, original_height, original_width = test_array.shape

        if test_array[0].shape[:2] != (self.IMG_HEIGHT, self.IMG_WIDTH):
            resizeImage = True
        else:
            resizeImage = False

        pred_array = np.zeros_like(test_array)

        print(f"Global intensity range: [{min_intensity}, {max_intensity}]")
        frame = 0

        for image in tqdm(test_array, desc="Processing images"):
            try:

                image = self.normalize_image(image, min_intensity, max_intensity)

                #if image.shape[:2] != (self.IMG_HEIGHT, self.IMG_WIDTH):
                if resizeImage:
                    print(f"Warning: {image} is not {self.IMG_HEIGHT}x{self.IMG_WIDTH}. Resizing.")
                    image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)

                image = np.expand_dims(image, axis=-1)
                image = np.expand_dims(image, axis=0)

                prediction = self.model.predict(image)
                #prediction = (prediction > 0.5).astype(np.uint8) * 255

                if resizeImage:
                    prediction = resize(prediction[0, ..., 0], (original_height, original_width), mode='constant', preserve_range=True)
                else:
                    prediction = prediction[0, ..., 0]

                output_path = os.path.join(output_folder, f"pred_{frame}.tif")
                imsave(output_path, prediction)

                pred_array[frame] = prediction

            except Exception as e:
                print(f"Error processing frame {frame}: {str(e)}")

            frame += 1

        return pred_array

    def predict_large_images(self, input_folder, output_folder, tile_size=128, overlap=16):
        """
        Predict segmentation for large TIFF images using tiling approach.

        :param input_folder: Path to the folder containing input images
        :param output_folder: Path to save the predicted segmentation masks
        :param tile_size: Size of tiles to use for prediction (default: 128)
        :param overlap: Overlap between tiles (default: 16)
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

        if not image_files:
            print(f"No TIFF images found in {input_folder}")
            return

        min_intensity, max_intensity = self.get_intensity_range(input_folder)
        print(f"Global intensity range: [{min_intensity}, {max_intensity}]")

        for image_file in tqdm(image_files, desc="Processing large images"):
            try:
                image_path = os.path.join(input_folder, image_file)
                image = imread(image_path).astype(np.float32)

                image = self.normalize_image(image, min_intensity, max_intensity)

                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)

                h, w, c = image.shape
                pad_h = (tile_size - h % tile_size) % tile_size
                pad_w = (tile_size - w % tile_size) % tile_size

                padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                prediction = np.zeros_like(padded_image)

                for y in range(0, padded_image.shape[0], tile_size - overlap):
                    for x in range(0, padded_image.shape[1], tile_size - overlap):
                        tile = padded_image[y:y+tile_size, x:x+tile_size]
                        tile = np.expand_dims(tile, axis=0)

                        tile_prediction = self.model.predict(tile)
                        tile_prediction = (tile_prediction > 0.5).astype(np.uint8)

                        prediction[y:y+tile_size, x:x+tile_size] = np.maximum(
                            prediction[y:y+tile_size, x:x+tile_size],
                            tile_prediction[0, ..., 0]
                        )

                prediction = prediction[:h, :w] * 255
                output_path = os.path.join(output_folder, f"pred_{image_file}")
                imsave(output_path, prediction[..., 0])

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

class ImageSegmenter:
    def __init__(self):
        super().__init__()
        print("ImageSegmentation initialized")
        self.classifier = None
        self.unet_classifier = UNetClassifier(img_height=128, img_width=128, img_channels=1, num_test_images=10, use_gpu=True)
        # Build the model
        self.unet_classifier.build_model()
        self.init_items()

    def init_items(self):
        print("Initializing items")

        self.train_window = ComboBox()
        self.test_window = ComboBox()
        self.mask_window = ComboBox()
        self.classifier_choice = ComboBox()
        self.classifier_choice.addItems(["Random Forest", "SVM", "Gradient Boosting", "k-NN", "U-Net"])
        self.svm_kernel_choice = ComboBox()
        self.svm_kernel_choice.addItems(["linear", "poly", "rbf", "sigmoid"])

        self.rf_n_estimators = SliderLabel()
        self.rf_n_estimators.setRange(10, 500)
        self.rf_n_estimators.setValue(100)

        self.rf_max_depth = SliderLabel()
        self.rf_max_depth.setRange(1, 50)
        self.rf_max_depth.setValue(10)

        self.gaussian_sigma = SliderLabel()
        self.gaussian_sigma.setRange(1, 100)
        self.gaussian_sigma.setValue(20)

        self.unet_epochs = SliderLabel()
        self.unet_epochs.setRange(1, 100)
        self.unet_epochs.setValue(20)

        # Feature selection checkboxes
        self.feature_original = CheckBox("Original Image")
        self.feature_original.setChecked(True)
        self.feature_gaussian = CheckBox("Gaussian Blur")
        self.feature_gaussian.setChecked(True)
        self.feature_sobel = CheckBox("Sobel Edge")
        self.feature_sobel.setChecked(True)
        self.feature_median = CheckBox("Median Filter")
        self.feature_median.setChecked(False)
        self.feature_gradient = CheckBox("Gradient Magnitude")
        self.feature_gradient.setChecked(False)

        self.feature_generators = {
            'original': lambda img: img,
            'gaussian': lambda img: filters.gaussian(img, sigma=self.gaussian_sigma.value() / 10),
            'sobel': filters.sobel,
            'median': filters.median,
            'gradient': lambda img: filters.rank.gradient(img.astype(np.uint8), disk(3))
        }

    def gui(self):
        print("Setting up GUI")
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(400, 400)  # Increased width to accommodate three docks
        self.win.setWindowTitle('Image Segmentation')

        # Main options dock
        self.d1 = Dock("Main Options", size=(200, 400))
        self.area.addDock(self.d1, 'left')

        # Traditional ML options dock
        self.d2 = Dock("Traditional ML Options", size=(200, 400))
        self.area.addDock(self.d2, 'right', self.d1)

        # U-Net options dock
        self.d3 = Dock("U-Net Options", size=(200, 400))
        self.area.addDock(self.d3, 'below', self.d2)

        self.w1 = pg.LayoutWidget()
        self.w2 = pg.LayoutWidget()
        self.w3 = pg.LayoutWidget()

        # Main options widgets
        self.updateGUI_button = QPushButton('Update')
        self.run_button = QPushButton('Run')
        self.export_images_button = QPushButton('Export Images')
        self.export_masks_button = QPushButton('Export Masks')

        self.w1.addWidget(QLabel('Training Window:'), 1, 0)
        self.w1.addWidget(self.train_window, 1, 1)
        self.w1.addWidget(QLabel('Testing Window:'), 2, 0)
        self.w1.addWidget(self.test_window, 2, 1)
        self.w1.addWidget(QLabel('Mask Window:'), 3, 0)
        self.w1.addWidget(self.mask_window, 3, 1)
        self.w1.addWidget(QLabel('Classifier:'), 4, 0)
        self.w1.addWidget(self.classifier_choice, 4, 1)
        self.w1.addWidget(self.updateGUI_button, 5, 0)
        self.w1.addWidget(self.run_button, 5, 1)
        self.w1.addWidget(self.export_images_button, 6, 0)
        self.w1.addWidget(self.export_masks_button, 6, 1)

        # Traditional ML options widgets
        self.w2.addWidget(QLabel('SVM Kernel:'), 1, 0)
        self.w2.addWidget(self.svm_kernel_choice, 1, 1)
        self.w2.addWidget(QLabel('RF N Estimators:'), 2, 0)
        self.w2.addWidget(self.rf_n_estimators, 2, 1)
        self.w2.addWidget(QLabel('RF Max Depth:'), 3, 0)
        self.w2.addWidget(self.rf_max_depth, 3, 1)
        self.w2.addWidget(QLabel('Gaussian Sigma:'), 4, 0)
        self.w2.addWidget(self.gaussian_sigma, 4, 1)
        self.w2.addWidget(QLabel('Features:'), 5, 0)
        self.w2.addWidget(self.feature_original, 5, 1)
        self.w2.addWidget(self.feature_gaussian, 6, 0)
        self.w2.addWidget(self.feature_sobel, 6, 1)
        self.w2.addWidget(self.feature_median, 7, 0)
        self.w2.addWidget(self.feature_gradient, 7, 1)

        # U-Net options widgets
        self.train_unet_button = QPushButton('Train U-Net')
        self.load_unet_weights_button = QPushButton('Load U-Net Weights')
        self.save_unet_weights_button = QPushButton('Save U-Net Weights')
        self.plot_history_button = QPushButton('Plot Training History')


        self.w3.addWidget(self.train_unet_button, 1, 0, 1, 2)
        self.w3.addWidget(self.load_unet_weights_button, 2, 0)
        self.w3.addWidget(self.save_unet_weights_button, 2, 1)
        self.w3.addWidget(self.plot_history_button, 3, 0, 1, 2)
        self.w3.addWidget(QLabel('U-Net Epochs:'), 4, 0)
        self.w3.addWidget(self.unet_epochs, 4, 1)

        self.d1.addWidget(self.w1)
        self.d2.addWidget(self.w2)
        self.d3.addWidget(self.w3)

        # Connect buttons to functions
        self.updateGUI_button.clicked.connect(self.update_comboboxes)
        self.run_button.clicked.connect(self.run_analysis)
        self.train_unet_button.clicked.connect(self.train_unet)
        self.load_unet_weights_button.clicked.connect(self.load_unet_weights)
        self.save_unet_weights_button.clicked.connect(self.save_unet_weights)
        self.export_images_button.clicked.connect(self.export_images)
        self.export_masks_button.clicked.connect(self.export_masks)
        self.plot_history_button.clicked.connect(self.plot_training_history)

        self.update_comboboxes()
        print("GUI setup complete")
        return self.win

    def update_comboboxes(self):
        windows = [w.name for w in g.windows if isinstance(w, Window)]
        self.test_window.clear()
        self.test_window.addItems(windows)
        self.train_window.clear()
        self.train_window.addItems(windows)
        self.mask_window.clear()
        self.mask_window.addItems(windows)

    def generate_feature_stack(self, image):
        feature_stack = []
        for feature_name, checkbox in [
            ('original', self.feature_original),
            ('gaussian', self.feature_gaussian),
            ('sobel', self.feature_sobel),
            ('median', self.feature_median),
            ('gradient', self.feature_gradient)
        ]:
            if checkbox.isChecked():
                feature = self.feature_generators[feature_name](image)
                feature_stack.append(feature.ravel())
        return np.asarray(feature_stack)

    def format_data(self, feature_stack, annotation):
        X = feature_stack.T
        y = annotation.ravel()
        mask = y > 0
        X = X[mask]
        y = y[mask]
        return X, y

    def make_annotation(self, win):
        annotation = np.zeros(win.image.shape)
        rois = win.rois

        for roi in rois:
            pt1, pt2 = roi.getPoints()
            color = roi.pen.color().name()
            if color == '#ffff00':
                annotation[pt1[0]:(pt1[0]+pt2[0]), pt1[1]:(pt1[1]+pt2[1])] = 1
            elif color == '#0000ff':
                annotation[pt1[0]:(pt1[0]+pt2[0]), pt1[1]:(pt1[1]+pt2[1])] = 2
            else:
                annotation[pt1[0]:(pt1[0]+pt2[0]), pt1[1]:(pt1[1]+pt2[1])] = 3

        return annotation


    def train_unet(self):
        #images_dir = QFileDialog.getExistingDirectory(caption="Select Training Images Directory")
        #masks_dir = QFileDialog.getExistingDirectory(caption="Select Masks Directory")

        input_dir = '/Users/george/Data/unet_flika/data'
        images_dir = 'input/bbbc005_v1_images/BBBC005_v1_images'
        masks_dir = 'input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth'


        if images_dir and masks_dir:

            self.unet_classifier.load_data(input_dir=input_dir,
                                images_dir=images_dir,
                                masks_dir=masks_dir)

            #display summary
            #self.unet_classifier.get_summary()

            # Train the model and save weights after training
            history = self.unet_classifier.train_model(epochs=50, batch_size=16, validation_split=0.1,
                                            save_weights_path='/Users/george/Data/unet_flika/data/save.weights.h5')

            # Plot the training loss
            self.unet_classifier.plot_training_loss(history)


    def load_unet_weights(self):
        weights_path, _ = QFileDialog.getOpenFileName(caption="Select U-Net Weights", filter="HDF5 files (*.h5)")
        if weights_path:
            self.unet_classifier.load_pretrained_weights(weights_path)

    def save_unet_weights(self):
        weights_path, _ = QFileDialog.getSaveFileName(caption="Save U-Net Weights", filter="HDF5 files (*.h5)")
        if weights_path:
            self.unet_classifier.save_weights(weights_path)

    def export_images(self):
        export_dir = QFileDialog.getExistingDirectory(caption="Select Directory to Export Images")
        if export_dir:
            deleteFolderFiles(export_dir)
            self.save_tiff_stack(g.windows[self.train_window.currentIndex()], export_dir, 'image')

    def export_masks(self):
        export_dir = QFileDialog.getExistingDirectory(caption="Select Directory to Export Masks")
        if export_dir:
            deleteFolderFiles(export_dir)
            self.save_tiff_stack(g.windows[self.train_window.currentIndex()], export_dir, 'image')

    def save_tiff_stack(self, win, export_dir, data_type):
        from tifffile import imsave
        image = win.image

        print(f"Window image shape: {image.shape}")

        min_intensity = np.min(image)
        max_intensity = np.max(image)

        # If the image is 2D (single slice), add a dimension to make it 3D
        if image.ndim == 2:
            image = image[np.newaxis, ...]

        for i in range(image.shape[0]):
            # Extract the 2D slice
            slice_2d = image[i]

            print(f"Slice {i} shape: {slice_2d.shape}")

            # If the slice is 1D, reshape it to 2D
            if slice_2d.ndim == 1:
                slice_2d = slice_2d.reshape((image.shape[1], -1))  # Reshape using known dimensions

            # Ensure the image is in the correct data type and range
            if data_type == 'image':
                normalized = (slice_2d - min_intensity) / (max_intensity - min_intensity)
                save_image = (normalized * 255).astype(np.uint8)

            else:  # mask
                save_image = (slice_2d > 0).astype(np.uint8) * 255

            print(f"Saving image with shape: {save_image.shape}")

            filename = f"{export_dir}/{data_type}_{i:04d}.tif"
            imsave(filename, save_image)
            print(f"Saved {filename}")

    def run_analysis(self):
        logger.info('Starting analysis')
        print('Start Analysis')
        try:
            train_window = self.train_window.currentIndex()
            test_window = self.test_window.currentIndex()
            mask_window = self.mask_window.currentIndex()
            classifier_type = self.classifier_choice.currentText()

            train_win = g.windows[train_window]
            test_win = g.windows[test_window]
            mask_win = g.windows[mask_window]

            train_array = train_win.image
            test_array = test_win.image
            mask_array = mask_win.image

            if classifier_type != "U-Net":
                self.process_classifier(train_array, test_array)
            else:
                self.process_unet(train_array, test_array)

            print('Analysis finished')
        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}", exc_info=True)
            print(f"An error occurred during analysis: {e}")
            import traceback
            traceback.print_exc()

    def process_classifier(self, train_array, test_array):
        classifier_type = self.classifier_choice.currentText()
        feature_stack = self.generate_feature_stack(train_array)
        annotation = self.make_annotation(g.windows[self.train_window.currentIndex()])
        X, y = self.format_data(feature_stack, annotation)

        print(f"Training {classifier_type}...")

        if classifier_type == "Random Forest":
            n_estimators = self.rf_n_estimators.value()
            max_depth = self.rf_max_depth.value()
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif classifier_type == "SVM":
            kernel = self.svm_kernel_choice.currentText()
            classifier = SVC(kernel=kernel, random_state=42)
        elif classifier_type == "Gradient Boosting":
            classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
        elif classifier_type == "k-NN":
            classifier = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        classifier.fit(X, y)
        self.apply_classifier(classifier, test_array)


    def process_unet(self, train_array, test_array):
        print("Applying U-Net...")
        print(f"Original test_array shape: {test_array.shape}")

        # Process each slice separately
        result_stack = self.unet_classifier.predict_array(test_array, '/Users/george/Data/unet_flika/results')

        print(f"Final result shape: {result_stack.shape}")

        self.display_results(result_stack, test_array)


    def apply_classifier(self, classifier, test_array):
        feature_stack = self.generate_feature_stack(test_array)
        result = classifier.predict(feature_stack.T) - 1
        self.display_results(result, test_array)

    def display_results(self, res, image):
        res_image = res.reshape(image.shape)
        self.result_win = Window(res_image, self.classifier_choice.currentText())

    def plot_training_history(self):
        if hasattr(self.unet_classifier, 'history'):
            self.unet_classifier.plot_training_history(self.unet_classifier.history)
        else:
            print("No training history available. Please train the model first.")

def gui():
    try:
        print("Starting FLIKA")
        flika.start_flika()
        print("FLIKA started")
        app = QApplication.instance()
        plugin = ImageSegmenter()
        plugin_gui = plugin.gui()
        plugin_gui.show()
        print("Plugin GUI displayed")
        app.exec_()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gui()
