import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import tqdm

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
        df_images = df_images[df_images['image_id'] != '.DS_Store']

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
        #set default value
        value = 1
        try:
            value = int(x.split('_')[2][1:])
        except:
            pass
        return value

    def get_blur_amt(self, x):
        #set default value
        value = 1
        try:
            value = int(x.split('_')[3][1:])
        except:
            pass
        return value

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

                image = np.expand_dims(image, axis=-1)
                image = np.expand_dims(image, axis=0)

                prediction = self.model.predict(image)
                #prediction = (prediction > 0.5).astype(np.uint8) * 255

                output_path = os.path.join(output_folder, f"pred_{image_file}")
                imsave(output_path, prediction[0, ..., 0])

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

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

#######################################################################
input_dir='/Users/george/Data/unet_flika/data'
#images_dir='input/bbbc005_v1_images/BBBC005_v1_images'
#masks_dir='input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth'

images_dir='combined_input/BBBC005_v1_images'
masks_dir='combined_input/BBBC005_v1_ground_truth'

weights_path = '/Users/george/Data/unet_flika/data/save.weights.h5'
#######################################################################


#from cell_segmentation import UNetClassifier

# Initialize the UNetClassifier object
segmenter = UNetClassifier(img_height=128, img_width=128, img_channels=1, num_test_images=10, use_gpu=True)

# Load and prepare data
segmenter.load_data(input_dir=input_dir,
                    images_dir=images_dir,
                    masks_dir=masks_dir)

# Build the model
segmenter.build_model()

#display summary
segmenter.get_summary()

# Load pre-trained weights (optional)
#segmenter.load_pretrained_weights(weights_path)

# Train the model and save weights after training
history = segmenter.train_model(epochs=50, batch_size=16, validation_split=0.1,
                                save_weights_path= weights_path)

# Plot the training loss
segmenter.plot_training_loss(history)

# Make predictions
predictions = segmenter.predict()

# Visualize results
segmenter.visualize_results(predictions, masks_dir=masks_dir, num_samples=3)

# You can also save weights manually at any point
#segmenter.save_weights(weights_path)






# =============================================================================
# #Initialize the UNetClassifier object
# segmenter = UNetClassifier(img_height=128, img_width=128, img_channels=1, num_test_images=10, use_gpu=False)
#
# #Build the model
# segmenter.build_model()
# # Initialize and set up the model as before
# segmenter.load_pretrained_weights(weights_path)
#
# # Predict on 128x128 images
# #testFolder = '/Users/george/Data/unet_flika/data/input/bbbc005_v1_images/BBBC005_v1_images'
# testFolder = '/Users/george/Data/unet_flika/training_images'
#
# segmenter.predict_images(input_folder= testFolder, output_folder= '/Users/george/Data/unet_flika/results')
#
# # Predict on large images
# #segmenter.predict_large_images(input_folder='path/to/large_images', output_folder='path/to/large_predictions', tile_size=128, overlap=16)
#
# =============================================================================


