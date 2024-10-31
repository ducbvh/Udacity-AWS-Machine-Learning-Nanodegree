from utils.models import Generator
from utils.helper import generate_images
import tensorflow as tf
import numpy as np
import cv2
import argparse
from datetime import datetime

def load_img(image_path):
    input_image = tf.io.read_file(image_path)
    input_image = tf.image.decode_png(input_image, channels= 3)
    #normalize
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1
    #resize
    input_image = tf.image.resize(input_image, (256, 256), method= 'bilinear')
    return input_image

def inference_sample(args):
    image_input = load_img(image_path=args.image_path)
    my_generator = Generator(train_attention=False)
    path_model = "weights/generator_20241028-173931.h5"
    status = my_generator.load_weights(args.model_dir, by_name=True)

    # status = my_generator.load_weights("weights/model_3.h5", by_name=True)

    input_image = tf.expand_dims(image_input, axis=0)

    generate_images(model=my_generator, test_input=input_image, tar=input_image, save_folder=args.output_dir, epoch="4")
    output_image = my_generator(input_image, training=True)

    output_image = output_image[0].numpy()*0.5+0.5
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    now = datetime.datetime.now()
    exp_time = now.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(f"{args.output_dir}/test_img_{exp_time}.png",output_image)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="./weights/generator_base.h5")
    parser.add_argument('--output_dir', type=str, default="./inference_output")
    parser.add_argument('--image_path', type=bool, default=False)

    args=parser.parse_args()
    inference_sample(args)



