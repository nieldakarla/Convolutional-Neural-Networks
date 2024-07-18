import tensorflow as tf
import os
import numpy as np

def load_images(directory, img_size=(150, 150)):
    data = []
    labels = []
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(directory, split)
        for folder_name in ['NORMAL', 'PNEUMONIA']:
            folder_path = os.path.join(split_path, folder_name)
            if not os.path.exists(folder_path):
                print(f'Folder does not exist: {folder_path}')
                continue
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    img = tf.keras.preprocessing.image.load_img(file_path, target_size=img_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0
                    data.append(img_array)
                    labels.append(0 if folder_name == 'NORMAL' else 1)
                except (IOError, SyntaxError) as e:
                    print(f'Skipping bad file: {file_path}')
    return np.array(data), np.array(labels)

dataset_path = '/Users/nieldamac/Documents/Convolutional-Neural-Networks/chest_xray'
data, labels = load_images(dataset_path)
print(f'Loaded {len(data)} images.')

# Check the shape of the data and labels
print(f'Data shape: {data.shape}')
print(f'Labels shape: {labels.shape}')
