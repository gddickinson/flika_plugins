#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:55:50 2024

@author: george
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bridge
        self.bridge = self.conv_block(512, 1024)

        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.functional.max_pool2d(enc1, 2))
        enc3 = self.enc3(nn.functional.max_pool2d(enc2, 2))
        enc4 = self.enc4(nn.functional.max_pool2d(enc3, 2))

        # Bridge
        bridge = self.bridge(nn.functional.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(bridge)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final(dec1))

class CellSegmentationDataset(Dataset):
    def __init__(self, images, masks=None, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.masks is not None:
            mask = self.masks[idx]
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask
        else:
            if self.transform:
                image = self.transform(image)
            return image

class CellSegmentation:
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

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using {self.device} for computations")

    def load_data(self, input_dir, images_dir, masks_dir):
        os.chdir(input_dir)

        img_list = os.listdir(images_dir)
        mask_list = os.listdir(masks_dir)

        df_images = pd.DataFrame(img_list, columns=['image_id'])
        df_images = df_images[df_images['image_id'] != '.htaccess']

        df_images['num_cells'] = df_images['image_id'].apply(self.get_num_cells)
        df_images['has_mask'] = df_images['image_id'].apply(lambda x: 'yes' if x in mask_list else 'no')
        df_images['blur_amt'] = df_images['image_id'].apply(self.get_blur_amt)

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

        self.X_train = np.zeros((len(image_id_list), self.IMG_CHANNELS, self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.float32)
        self.Y_train = np.zeros((len(image_id_list), 1, self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.float32)
        self.X_test = np.zeros((self.NUM_TEST_IMAGES, self.IMG_CHANNELS, self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.float32)

        for i, image_id in enumerate(image_id_list):
            self.X_train[i] = self.process_image(os.path.join(images_dir, image_id)).transpose(2, 0, 1)

        for i, mask_id in enumerate(mask_id_list):
            mask = self.process_image(os.path.join(masks_dir, mask_id))
            self.Y_train[i] = (mask > 0.5).astype(np.float32).transpose(2, 0, 1)  # Ensure binary mask

        for i, image_id in enumerate(test_id_list):
            self.X_test[i] = self.process_image(os.path.join(images_dir, image_id)).transpose(2, 0, 1)

    def process_image(self, image_path):
        image = imread(image_path)
        image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
        if image.ndim == 2:  # If it's a grayscale image
            image = np.expand_dims(image, axis=-1)
        return image.astype(np.float32) / 255.0  # Normalize to [0, 1]

    def build_model(self):
        self.model = UNet(in_channels=self.IMG_CHANNELS, out_channels=1).to(self.device)

    def train_model(self, epochs=50, batch_size=16, validation_split=0.1, save_weights_path=None):
        X_train_tensor = torch.from_numpy(self.X_train)
        Y_train_tensor = torch.from_numpy(self.Y_train)

        # Print shapes and value ranges to verify
        print(f"X_train shape: {X_train_tensor.shape}, range: [{X_train_tensor.min()}, {X_train_tensor.max()}]")
        print(f"Y_train shape: {Y_train_tensor.shape}, range: [{Y_train_tensor.min()}, {Y_train_tensor.max()}]")

        dataset = CellSegmentationDataset(X_train_tensor, Y_train_tensor)
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())

        # Report split sizes
        print('Training set has {} instances'.format(len(train_dataset)))
        print('Validation set has {} instances'.format(len(val_dataset)))

        best_val_loss = float('inf')
        for epoch in tqdm(range(epochs)):
            self.model.train()
            train_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                if torch.any(masks < 0) or torch.any(masks > 1):
                    raise ValueError(f"Mask values out of range: min={masks.min().item()}, max={masks.max().item()}")

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_weights_path:
                    torch.save(self.model.state_dict(), save_weights_path)
                    print(f"Model saved to {save_weights_path}")

        return {"train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader)}

    def predict(self):
        self.model.eval()
        X_test_tensor = torch.from_numpy(self.X_test).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_test_tensor)

        return (predictions.cpu().numpy() > 0.5).astype(np.uint8)

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
            plt.imshow(preds_test_thresh[i, 0, :, :], cmap='gray')
            plt.title('Pred Mask', fontsize=14)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_training_loss(self, history):
        plt.figure(figsize=(8, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")
        print(self.model)

    def save_weights(self, filepath):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")

        try:
            torch.save(self.model.state_dict(), filepath)
            print(f"Weights saved successfully to {filepath}")
        except Exception as e:
            print(f"Failed to save weights to {filepath}. Error: {str(e)}")

    def load_pretrained_weights(self, weights_path):
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build_model() first.")

        try:
            self.model.load_state_dict(torch.load(weights_path))
            print(f"Weights loaded successfully from {weights_path}")
        except:
            print(f"Failed to load weights from {weights_path}. Check if the file exists and the architecture matches.")

    def get_intensity_range(self, input_folder):
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
        normalized = (image - min_intensity) / (max_intensity - min_intensity)
        return (normalized * 255).astype(np.uint8)

    def predict_images(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

        if not image_files:
            print(f"No TIFF images found in {input_folder}")
            return

        min_intensity, max_intensity = self.get_intensity_range(input_folder)
        print(f"Global intensity range: [{min_intensity}, {max_intensity}]")

        self.model.eval()
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
                image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).to(self.device)

                with torch.no_grad():
                    prediction = self.model(image_tensor)

                prediction = prediction.cpu().numpy()[0, 0, :, :]
                output_path = os.path.join(output_folder, f"pred_{image_file}")
                imsave(output_path, prediction)

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

    def predict_large_images(self, input_folder, output_folder, tile_size=128, overlap=16):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

        if not image_files:
            print(f"No TIFF images found in {input_folder}")
            return

        min_intensity, max_intensity = self.get_intensity_range(input_folder)
        print(f"Global intensity range: [{min_intensity}, {max_intensity}]")

        self.model.eval()
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
                        tile_tensor = torch.from_numpy(tile).permute(0, 3, 1, 2).to(self.device)

                        with torch.no_grad():
                            tile_prediction = self.model(tile_tensor)

                        tile_prediction = tile_prediction.cpu().numpy()[0, 0, :, :]
                        prediction[y:y+tile_size, x:x+tile_size] = np.maximum(
                            prediction[y:y+tile_size, x:x+tile_size],
                            tile_prediction
                        )

                prediction = prediction[:h, :w] * 255
                output_path = os.path.join(output_folder, f"pred_{image_file}")
                imsave(output_path, prediction[..., 0])

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

# Usage example:
if __name__ == "__main__":
    segmenter = CellSegmentation(img_height=128, img_width=128, img_channels=1, num_test_images=10, use_gpu=True)

    segmenter.load_data(input_dir='/Users/george/Data/unet_flika/data',
                        images_dir='input/bbbc005_v1_images/BBBC005_v1_images',
                        masks_dir='input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth')

    segmenter.build_model()

    segmenter.get_summary()


    # Uncomment to load pre-trained weights
    # segmenter.load_pretrained_weights('/Users/george/Data/unet_flika/data/save.weights.h5')

    history = segmenter.train_model(epochs=50, batch_size=16, validation_split=0.1,
                                    save_weights_path='/Users/george/Data/unet_flika/data/save.weights.h5')

    segmenter.plot_training_loss(history)

    predictions = segmenter.predict()

    segmenter.visualize_results(predictions, masks_dir='input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth', num_samples=3)

# =============================================================================
#     # Predict on new images
#     # #testFolder = '/Users/george/Data/unet_flika/data/input/bbbc005_v1_images/BBBC005_v1_images'
#     testFolder = '/Users/george/Data/unet_flika/training_images'
#     segmenter.predict_images(input_folder=testFolder, output_folder='/Users/george/Data/unet_flika/results')
#
#     # Predict on large images
#     segmenter.predict_large_images(input_folder='/path/to/large_images', output_folder='/path/to/large_predictions', tile_size=128, overlap=16)
#
# =============================================================================
