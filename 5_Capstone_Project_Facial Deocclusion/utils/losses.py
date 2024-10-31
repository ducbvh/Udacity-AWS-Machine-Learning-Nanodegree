import tensorflow as tf

def discriminator_loss(real_image, fake_image):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    
    real_loss = cross_entropy(tf.ones_like(real_image), real_image)
    fake_loss = cross_entropy(tf.zeros_like(fake_image), fake_image)
    total_loss = real_loss + fake_loss
    return total_loss

def loss_SSIM(target_image, gen_image):
    ssim = tf.image.ssim(target_image, gen_image, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return (1-ssim)


def generator_loss(target_image, gen_image):
    mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    #Loss SSIM
    l_ssim = tf.reduce_mean(loss_SSIM(target_image, gen_image))
    
    #Loss Recontruction
    l_rec = mae(target_image, gen_image)
    return l_ssim, l_rec