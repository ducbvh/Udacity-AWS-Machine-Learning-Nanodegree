import os
import boto3
import logging
import tensorflow as tf
# from .settings import SecrectAWS



class CustomDataLoader:
    def __init__(self, bucket_name, data_dir, image_size=(256,256), 
                 batch_size=16, buffer_size=500, images_train=0):
        self.s3 = self.client('s3')
        self.bucket_name = bucket_name
        self.data_dir = data_dir  # This will be the S3 prefix
        self.dir_train, self.dir_val, self.dir_test = self.dir_train_test_validation()
        
        self.image_size = image_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.images_train = images_train
        # self.images_valid_test = int(self.images_train*10%) if self.images_train>10 else
        self.trainsets = None
        self.valsets = None
        self.testsets = None
    
    def dir_train_test_validation(self):
        dir_train = os.path.join(self.data_dir, "train")
        dir_val = os.path.join(self.data_dir, "validation")
        dir_test = os.path.join(self.data_dir, "test")
        return dir_train, dir_val, dir_test
        
    def dir_folder_mask_unmask(self, dir_datasets):
        dir_folder_unmask = os.path.join(dir_datasets, "un_mask")
        dir_folder_masked = os.path.join(dir_datasets, "mask")
        print(dir_folder_unmask, dir_folder_masked)
        return dir_folder_unmask, dir_folder_masked
    
    def list_s3_files(self, prefix):
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [content['Key'] for content in response['Contents']]
        return []
    
    def get_images(self, dir_mask, dir_unmask, data_test=False):
        maskes = []
        unmaskes = []

        # Retrieve the list of images from S3
        mask_files = self.list_s3_files(dir_mask)
        unmask_files = self.list_s3_files(dir_unmask)

        # Match files based on your logic
        for path_img_unmask in unmask_files:
            base_name_unmask = os.path.basename(path_img_unmask).split('.')[0]
            for path_img_mask in mask_files:
                # print(base_name_unmask, os.path.basename(name_mask).split(".")[0])
                if os.path.basename(path_img_mask).split(".")[0].split('_')[0] == base_name_unmask:
                    mask_imge, unmask_image = self.load_image(mask_path=path_img_mask, 
                                                              unmask_path=path_img_unmask)
                    maskes.append(mask_imge)
                    unmaskes.append(unmask_image)
            if not data_test and self.images_train>0 and len(maskes)==self.images_train:
                break
            elif data_test and self.images_train>0 and len(maskes)==int(self.images_train*10/100):
                break

        logger.info(f"Number image mask: {len(maskes)} - Num image unmask: {len(unmaskes)}")
        return maskes, unmaskes
    
    def resize_image(self, image_mask, image_unmask):
        image_mask = tf.image.resize(image_mask, self.image_size, method= 'bilinear')
        image_unmask = tf.image.resize(image_unmask, self.image_size, method= 'bilinear')
        return image_mask, image_unmask
    
    def normalize_image(self, image_mask, image_unmask):
        image_mask = tf.cast(image_mask, tf.float32)
        image_unmask = tf.cast(image_unmask, tf.float32)
        image_mask = (image_mask / 127.5) - 1
        image_unmask = (image_unmask / 127.5) - 1

        return image_mask, image_unmask
    
    def load_image(self, mask_path, unmask_path):
        image_mask = self.read_image_from_s3(object_key=mask_path)
        image_unmask = self.read_image_from_s3(object_key=unmask_path)
        image_mask, image_unmask = self.normalize_image(image_mask=image_mask, image_unmask=image_unmask)
        image_mask, image_unmask = self.resize_image(image_mask=image_mask, image_unmask=image_unmask)
        return image_mask, image_unmask
    
    def read_image_from_s3(self, object_key):
        # Download the image as a stream of bytes
        response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
        img_data = response['Body'].read()
        # Decode the image using TensorFlow
        img = tf.io.decode_image(img_data, channels=3)
        return img

    def load_trainsets(self):
        dir_train_unmask, dir_train_mask = self.dir_folder_mask_unmask(dir_datasets=self.dir_train)
        
        mask_images, unmask_images = self.get_images(dir_mask=dir_train_mask, dir_unmask=dir_train_unmask, data_test=False)
        print(f"Number train image: {len(mask_images)}")
        self.trainsets = tf.data.Dataset.from_tensor_slices((mask_images, unmask_images)).prefetch(tf.data.AUTOTUNE) \
                        .shuffle(buffer_size=self.buffer_size).batch(self.batch_size)
    
    def load_valsets(self):
        dir_val_unmask, dir_val_mask = self.dir_folder_mask_unmask(dir_datasets=self.dir_val)
        mask_images, unmask_images = self.get_images(dir_mask=dir_val_mask, dir_unmask=dir_val_unmask, data_test=True)

        print(f"Number validation image: {len(mask_images)}")

       
        self.valsets = tf.data.Dataset.from_tensor_slices((mask_images, unmask_images)).prefetch(tf.data.AUTOTUNE) \
                        .shuffle(buffer_size=self.buffer_size).batch(self.batch_size)
    def load_testsets(self):
        dir_test_unmask, dir_test_mask = self.dir_folder_mask_unmask(dir_datasets=self.dir_test)
        mask_images, unmask_images = self.get_images(dir_mask=dir_test_mask, dir_unmask=dir_test_unmask, data_test=True)

        print(f"Number test image: {len(mask_images)}")

       
        self.testsets = tf.data.Dataset.from_tensor_slices((mask_images, unmask_images)). \
            prefetch(tf.data.AUTOTUNE).batch(self.batch_size)

    def load_data_set(self, folder_data="test"):
        if folder_data.lower() == "train":
            if self.trainsets is None:
                self.load_trainsets()
            return self.trainsets
        elif folder_data == "validation":
            if self.valsets is None:
                self.load_valsets()
            return self.trainsets
        else:
            if self.testsets is None:
                self.load_testsets()
            return self.testsets

    def load(self):
        self.load_trainsets()
        self.load_valsets()
        self.load_testsets()
        return self.trainsets, self.valsets, self.testsets