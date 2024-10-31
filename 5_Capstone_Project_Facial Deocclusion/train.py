import os
import sys
import time
import datetime
import argparse
import logging
import numpy as np
import tensorflow as tf

from utils.data_loader import CustomDataLoader
from utils.models import Generator, Discriminator
from utils.losses import generator_loss, discriminator_loss
from utils.helper import generate_images, plot_history_train

##########################
##### Define Logger ######
##########################

# Get the current timestamp for the log file name
now = datetime.datetime.now()
exp_time = now.strftime("%Y%m%d-%H%M%S")

# Create a custom logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)  # Set the log level (DEBUG, INFO, WARNING, etc.)

# Create handlers
# 1. File handler to write logs to a file
file_handler = logging.FileHandler(filename=f"logs/log_{exp_time}.log")
file_handler.setLevel(logging.DEBUG)

# 2. Stream handler to print logs to the terminal (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatters and add them to the handlers
formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)



##########################
####### Model Fit ########
##########################

def evaluate_val(valid_dataset, model_information, args):
    loss_total_each_batch=[]
    loss_ssim_each_batch=[]
    loss_rec_each_batch=[]
    disc_loss_each_batch=[]
    
    generator = model_information["generator"]
    discriminator = model_information["discriminator"]
    
    lambda_rec = args.lambda_rec
    lambda_adv = args.lambda_adv
    lambda_ssim = args.lambda_ssim
    
    for step, (batch_test_image, batch_test_target) in enumerate(valid_dataset):
        valid_out = generator(batch_test_image, training=True)
        val_ssim_loss,  val_rec_loss = generator_loss(batch_test_target, valid_out)
        loss_ssim_each_batch.append(val_ssim_loss)
        loss_rec_each_batch.append(val_rec_loss)

        disc_real_output  = discriminator([batch_test_image, batch_test_target], training=True)
        disc_generated_output  = discriminator([batch_test_image, valid_out], training=True)

        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        loss_total = lambda_rec*val_rec_loss + lambda_adv*disc_loss + lambda_ssim*val_ssim_loss

        loss_total_each_batch.append(loss_total)
        disc_loss_each_batch.append(disc_loss)
    eval_results = {"loss_total_each_batch": np.mean(loss_total_each_batch),
                    "loss_ssim_each_batch": np.mean(loss_ssim_each_batch),
                    "loss_rec_each_batch": np.mean(loss_rec_each_batch),
                    "disc_loss_each_batch": np.mean(disc_loss_each_batch),}
    return eval_results


def train_step(train_datasets, update_D, model_information, args):
    loss_total_each_batch=[]
    loss_ssim_each_batch=[]
    loss_rec_each_batch=[]
    disc_loss_each_batch=[]
    
    lambda_rec = args.lambda_rec
    lambda_adv = args.lambda_adv
    lambda_ssim = args.lambda_ssim
    
    generator = model_information["generator"]
    generator_optimizer = model_information["generator_optimizer"]
    discriminator = model_information["discriminator"]
    discriminator_optimizer = model_information["discriminator_optimizer"]
    
    for step, (batch_train_image, batch_train_target) in enumerate(train_datasets):
        print(f"Batch: {step+1}...", end='\r')
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_out = generator(batch_train_image, training=True)
            gen_ssim_loss,  gen_rec_loss = generator_loss(batch_train_target, gen_out)
            loss_ssim_each_batch.append(gen_ssim_loss)
            loss_rec_each_batch.append(gen_rec_loss)

            disc_real_output = discriminator([batch_train_image, batch_train_target], training=True)
            disc_generated_output = discriminator([batch_train_image, gen_out], training=True)

            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            loss_total = lambda_rec*gen_rec_loss + lambda_adv*disc_loss + lambda_ssim*gen_ssim_loss

            loss_total_each_batch.append(loss_total)
            disc_loss_each_batch.append(disc_loss)

        generator_gradients = gen_tape.gradient(loss_total,
                                    generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                        generator.trainable_variables))
    
        if update_D:
            discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    train_results = {"loss_total_each_batch": np.mean(loss_total_each_batch),
                    "loss_ssim_each_batch": np.mean(loss_ssim_each_batch),
                    "loss_rec_each_batch": np.mean(loss_rec_each_batch),
                    "disc_loss_each_batch": np.mean(disc_loss_each_batch),}
    
    model_info = {"generator": generator,
                 "generator_optimizer": generator_optimizer,
                 "discriminator": discriminator,
                 "discriminator_optimizer":discriminator_optimizer}
            
    
    return model_info, train_results

def model_fit(train_datasets, valid_datasets, test_datasets, epochs, num_epoch_update_D, args):
    history_train = {"loss_total": [],
                      "loss_ssim": [],
                      "loss_rec": [], 
                      "loss_disc":[]}
    
    history_val = {"loss_total": [],
                   "loss_ssim": [],
                   "loss_rec": [], 
                   "loss_disc":[]}
    
    logger.info(f"Train attention: {args.train_attention}")
    generator = Generator(train_attention=args.train_attention)
    
    if args.load_retrain:
        logger.info(f"Load Model: {args.model_dir}")
        status = generator.load_weights(args.model_dir, by_name=True)
    # generator.load_weights("weights/generator_20241028-173931.h5", by_name=True)
   
    discriminator = Discriminator()
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(9e-6, beta_1=0.5)
    model_info = {"generator": generator,
                 "generator_optimizer": generator_optimizer,
                 "discriminator": discriminator,
                 "discriminator_optimizer":discriminator_optimizer}
    now = datetime.datetime.now()
    exp_time = now.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f"{args.output_dir}/path_to_save_checkpoint/exp_{exp_time}"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=model_info["generator_optimizer"],
                                     discriminator_optimizer=model_info["discriminator_optimizer"],
                                     generator=model_info["generator"],
                                     discriminator=model_info["discriminator"])
    log_dir="logs_tf/"

    summary_writer = tf.summary.create_file_writer(
      log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    path_save_image = os.path.join(checkpoint_dir, "save_images_result")
    if not os.path.isdir(path_save_image):
        os.makedirs(path_save_image, exist_ok=True)

    # for input, target in valid_datasets.take(3):
    for input, target in test_datasets.take(3):
        generate_images(generator, input, target, 
                path_save_image, "base_test", show_plt=False)

    for input, target in valid_datasets.take(3):
        generate_images(generator, input, target, 
                path_save_image, "base_valid", show_plt=False)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        #Training
        if epoch%num_epoch_update_D==0:
            update_D = True
            
        if epoch==0:
            update_D=False
            logger.info(f"\nEpoch {epoch+1}/{epochs}\nOny Update G!")

        else:
            if epoch%num_epoch_update_D==0:
                update_D = True
                logger.info(f"\nEpoch {epoch+1}/{epochs}:\nUpdate D!")

            else:
                update_D = False
                logger.info(f"\nEpoch {epoch+1}/{epochs}\nOny Update G!")
        
        model_info, train_results = train_step(train_datasets, update_D, model_info, args)

        history_train["loss_total"].append(train_results["loss_total_each_batch"]) 
        history_train["loss_ssim"].append(train_results["loss_ssim_each_batch"])
        history_train["loss_rec"].append(train_results["loss_rec_each_batch"]) 
        history_train["loss_disc"].append(train_results["disc_loss_each_batch"]) 

        #Evaluation on testsets
        eval_results = evaluate_val(valid_datasets, model_info, args)
        
        history_val["loss_total"].append(eval_results["loss_total_each_batch"]) 
        history_val["loss_ssim"].append(eval_results["loss_ssim_each_batch"])
        history_val["loss_rec"].append(eval_results["loss_rec_each_batch"])
        history_val["loss_disc"].append(eval_results["disc_loss_each_batch"])

        logger.info(f'  loss_total: {train_results["loss_total_each_batch"]} - loss_ssim: {train_results["loss_ssim_each_batch"]} - loss_rec: {train_results["loss_rec_each_batch"]} - loss_discriminator: {train_results["disc_loss_each_batch"]}')
        logger.info(f'  val_loss: {eval_results["loss_total_each_batch"]} - val_ssim: {eval_results["loss_ssim_each_batch"]} - val_rec: {eval_results["loss_rec_each_batch"]} - val_Disc: {eval_results["disc_loss_each_batch"]}')
        logger.info("  Time taken: %.2fs" % (time.time() - start_time))
        
        if args.save_cp_epoch and epoch%30==0:
            # if history_val["loss_total"][epoch] <= min(history_val["loss_total"]) :
                checkpoint.save(file_prefix=checkpoint_prefix)
            
        if epoch>0 and (epoch-1)%args.epoch_save_results==0:
            for input, target in valid_datasets.take(3):
                generate_images(model_info["generator"], input, target, path_save_image, f"valid_sample_{epoch}", show_plt=False)
            for input, target in test_datasets.take(3):
                generate_images(generator, input, target, 
                        path_save_image, f"test_sample_{epoch}", show_plt=False)

    plot_history_train(history_train=history_train['loss_total'], history_val=history_val['loss_total'], path_save=checkpoint_dir)
    plot_history_train(history_train=history_train['loss_ssim'], history_val=history_val['loss_ssim'], path_save=checkpoint_dir, file_name="ssim", total_history=False)
    plot_history_train(history_train=history_train['loss_rec'], history_val=history_val['loss_rec'], path_save=checkpoint_dir, file_name="rec", total_history=False)
    plot_history_train(history_train=history_train['loss_disc'], history_val=history_val['loss_disc'], path_save=checkpoint_dir, file_name="disc", total_history=False)
    
    
    return model_info["generator"], model_info["discriminator"], checkpoint_dir



#################
def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'S3 Bucket: {args.s3_bucket} - Data Paths: {args.data}')

    logger.info("Start load datasets...")
    mydataset = CustomDataLoader(bucket_name=args.s3_bucket, data_dir=args.data,
                                 image_size=(256,256), batch_size=args.batch_size, 
                                 buffer_size=500, images_train=args.num_img_trains, 
                                 )
    
    data_train, data_valid, data_test = mydataset.load()
    logger.info(f"Num Batch in trains: {len(data_train)}")

    logger.info("Training in processing....")
    model_gen, model_dis, checkpoint_dir = model_fit(train_datasets=data_train, valid_datasets=data_valid, 
                                     test_datasets=data_test, epochs=args.num_epochs, 
                                     num_epoch_update_D=args.epoch_updateD, args=args)
    
    logger.info("Training done, Saving best accuracy epoch into format .h5")
    now = datetime.datetime.now()
    exp_time = now.strftime("%Y%m%d-%H%M%S")
    model_gen.save(os.path.join(checkpoint_dir, f'generator_{exp_time}.h5'))
    model_dis.save(os.path.join(checkpoint_dir, f'discriminator_{exp_time}.h5'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--s3_bucket', type=str, default="aws-ml-mycapstone-project")
    parser.add_argument('--data', type=str, default="dataset-try")
    parser.add_argument('--lambda_rec', type=float, default=1.2)
    parser.add_argument('--lambda_adv', type=float, default=0.5)
    parser.add_argument('--lambda_ssim', type=float, default=80)
    parser.add_argument('--batch_size', type=float, default=8)
    parser.add_argument('--num_epochs', type=int, default=210)
    parser.add_argument('--epoch_updateD', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--model_dir', type=str, default="./weights/generator_base.h5")
    parser.add_argument('--output_dir', type=str, default="./exp_train")
    parser.add_argument('--epoch_save_results', type=int, default=5)
    parser.add_argument('--num_img_trains', type=int, default=0)
    parser.add_argument('--load_retrain', type=bool, default=False)
    parser.add_argument('--save_cp_epoch', type=bool, default=True)
    parser.add_argument('--train_attention', type=bool, default=False)

    args=parser.parse_args()
    logger.info(args)
    main(args)
