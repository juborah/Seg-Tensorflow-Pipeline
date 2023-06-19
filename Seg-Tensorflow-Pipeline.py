import os
from glob import glob
import cv2
import numpy as np
import random
import tensorflow as tf


size = 256
batch_size = 2
buffer_size = 100

AUTOTUNE = tf.data.experimental.AUTOTUNE

path = 'E:/repositories/medical_research/Data/Polyp_Colonoscopy/PNG/'

def load_data(path):
    img_dir_path = glob(os.path.join(path, 'Original/*'))
    mask_dir_path = glob(os.path.join(path, 'Ground Truth/*'))

    return img_dir_path, mask_dir_path

#img_path, mask_path = load_data(path='E:\repositories\medical_research\Data\Polyp_Colonoscopy')

img_dir_path, mask_dir_path = load_data(path)
print(f"Images: {len(img_dir_path)}, Masks: {len(mask_dir_path)}")



def read_img(img_dir_path):
    imgs = []
    for path in img_dir_path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32)

        imgs.append(img)

    return imgs

def read_mask(mask_dir_path):
    masks = []
    for path in mask_dir_path:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {path}")

        mask = cv2.resize(mask, (size, size))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(np.float32)

        masks.append(mask)

    return masks

imgs = read_img(img_dir_path)
masks = read_mask(mask_dir_path)

dataset = tf.data.Dataset.from_tensor_slices((imgs, masks))

for (img, mask) in dataset:
    print(img.numpy(), mask.numpy())
    print(img.numpy().shape, mask.numpy().shape)
    break


def normalize_img_mask(img, mask):
    """Normalize image and mask within range 0-1."""
    image = tf.cast(img, tf.float32)
    image = image / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = mask / 255.0

    return image, mask

dataset = dataset.map(normalize_img_mask, num_parallel_calls=AUTOTUNE)
print(dataset)


for (img, mask) in dataset:
    print(img.numpy(), mask.numpy())
    print(img.numpy().shape, mask.numpy().shape)
    break


def data_split(dataset, train_ratio = 0.8, val_ratio = 0.1):
    data_count = len(dataset)
    print(data_count)

    train_size = int(data_count * train_ratio)
    train_data = dataset.take(train_size)
    data_rem = dataset.skip(train_size)

    val_size = int(data_count * val_ratio)
    val_data = data_rem.take(val_size)

    test_data = data_rem.skip(val_size)

    return train_data, val_data, test_data

train_data, val_data, test_data = data_split(dataset, train_ratio = 0.8, val_ratio = 0.1)

print(f'Train_data: {len(train_data)}, Val_data: {len(val_data)}, Test _data: {len(test_data)}')

def _datasets(train_data, val_data, test_data):
    print("Train Data:")
    for img, mask in train_data:
        print(img.numpy(), mask.numpy())
        print(img.numpy().shape, mask.numpy().shape)

    print("Validation Data:")
    for img, mask in val_data:
        print(img.numpy(), mask.numpy())
        print(img.numpy().shape, mask.numpy().shape)

    print("Test Data:")
    for img, mask in test_data:
        print(img.numpy(), mask.numpy())
        print(img.numpy().shape, mask.numpy().shape) 

_datasets(train_data, val_data, test_data)

train_data = train_data.shuffle(len(train_data))
val_data = val_data.shuffle(len(val_data))

train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

print(train_data), print(val_data)


#def preprocess(img, masks):
    #def _parse_data(img, mask):
        #images = tf.image.decode_image(images, channels=3, dtype=tf.float32)
        #masks = tf.image.decode_image(masks, channels=1, dtype=tf.float32)
        #img = read_img(path)
        #masks = read_msk(path)

        #return img, masks

    #image_x, masks_y = tf.numpy_function(_parse_data, [img, masks], [tf.float32, tf.float32])
        
    #return image_x, masks_y

#def dataset(img_path, mask_path, batch_size):
    #data = tf.data.Dataset.from_tensor_slices((img_path, mask_path))
    #data = data.map(_parse_data)
    #data = data.shuffle(buffer_size)
    
    #data = data.map(_map_function, num_parallel_calls=AUTOTUNE)
    #data = data.batch(batch_size)

    #return data

#if __name__ == '__main__':
    #path = 'E:/repositories/medical_research/Data/Polyp_Colonoscopy/PNG/'
    #img_dir_path, mask_dir_path = load_data(path)
    #print(f"Images: {len(img_dir_path)}, Masks: {len(mask_dir_path)}")

    #img = read_img(path) 
    #masks = read_msk(path)


     




  