import os
from PIL import Image

def verify_images(directory):
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
                    img = Image.open(file_path)
                    img.verify()
                except (IOError, SyntaxError) as e:
                    print(f'Bad file: {file_path}')

dataset_path = '/Users/nieldamac/Documents/Convolutional-Neural-Networks/chest_xray'
verify_images(dataset_path)
