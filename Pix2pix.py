# -*- coding: UTF-8 -*-
"""
    Name: Qingneng Li (Eng: Lional)
    Time: 2020/05/07
    Place: SIAT, Shenzhen
    Item: sAC-PET <-- NAC-PET --> pCT

"""

import os, shutil
import numpy as np
import datetime
from utils import *
from models import *
from config import FLAGS

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

"""===================== Configure ========================="""
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
GPUs = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(GPUs[0], True)
# tf.debugging.set_log_device_placement(True)
VGG = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
"""======================= Main ============================"""
class Pix2Pix():
    def __init__(self):
        # load dataset
        self.x_train = tf.keras.utils.HDF5Matrix('./data/train_data.hdf5', 'NAC')
        self.y_train = tf.keras.utils.HDF5Matrix('./data/train_data.hdf5', 'AC')
        self.datagen = keras_generator(self.x_train[:FLAGS.split_data],
                                       self.y_train[:FLAGS.split_data])
        # Build G and D network
        self.generator, self.discriminator = Unet(), D_Pix2Pix()
        # self.generator.load_weights('../PETCT/logs/CT_clean_unet/model.h5', by_name=True)
        # Hyper-paramenters in train phrase
        self.step_per_epoch = FLAGS.split_data // FLAGS.batch_size
        self.train_steps = FLAGS.num_epoch * self.step_per_epoch
        # Build each G and D optimizers
        self.decay_steps, self.decay_rate = self.step_per_epoch*10, 0.64
        gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            FLAGS.gen_lr, decay_steps= self.decay_steps,
                            decay_rate=self.decay_rate, staircase=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(gen_lr_schedule, 0.5)
        dir_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            FLAGS.dis_lr, decay_steps = self.decay_steps,
                            decay_rate = self.decay_rate, staircase=True)
        self.dis_optimizer = tf.keras.optimizers.Adam(dir_lr_schedule, 0.5)
        # from_logit=True: desire a linear tensor without any activation in last
        # layer, you have to remove the sigmoid, since the loss itself applies
        # the softmax to your network output and compute cross entropy.
        self.LAMBDA = 100
        self.BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.MAE = tf.keras.losses.MeanAbsoluteError()
        self.MSE = tf.keras.losses.MeanSquaredError()
        self.PCP = PCP(VGG).Content
        # Summary in a tensorboard
        self.summary_writer = tf.summary.create_file_writer(FLAGS.logdir)

    def generator_loss(self, D_fake, y_pred, y_true):
        gan_loss = self.BCE(tf.ones_like(D_fake), D_fake)
        # Pixel loss
        pixel_loss = tf.reduce_sum([BME(y_true, y_pred),
                                    self.PCP(y_true, y_pred),
                                    ])
        return gan_loss + pixel_loss, gan_loss, pixel_loss

    def discriminator_loss(self, D_real, D_fake):
        real_loss = self.BCE(tf.ones_like(D_real), D_real)
        fake_loss = self.BCE(tf.zeros_like(D_fake), D_fake)
        return real_loss + fake_loss, real_loss, fake_loss

    def train(self):
        start_time = datetime.datetime.now()
        for batch_i, (input, label) in enumerate(self.datagen):

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                fake_img = self.generator(input, training=True)
                D_real = self.discriminator([input, label], training=True)
                D_fake = self.discriminator([input, fake_img], training=True)
                # Compute generator loss (gan loss + pixel loss)
                G_loss = self.generator_loss(D_fake, fake_img, label)
                # Compute discriminator loss (fake loss + real loss)
                D_loss = self.discriminator_loss(D_real, D_fake)
                # Compute Metrics: PSNR, SSIM , PCC
                psnr = PSNR(fake_img, label)
                ssim = SSIM(fake_img, label)
                pcc = PCC(fake_img, label)

            generator_gradients = gen_tape.gradient(G_loss[0],
                                                    self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(D_loss[0],
                                                         self.discriminator.trainable_variables)

            self.gen_optimizer.apply_gradients(zip(generator_gradients,
                                                   self.generator.trainable_variables))
            self.dis_optimizer.apply_gradients(zip(discriminator_gradients,
                                                   self.discriminator.trainable_variables))

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print ("[Step %d/%d] [D loss: %.4g/%.4g] [G loss: %.4g/%.4g]"
                   " [PSNR/SSIM/PCC: %.2f/%.4f/%.4f] time: %s"
                   % (batch_i, self.train_steps, D_loss[1], D_loss[2], G_loss[1],
                      G_loss[2], psnr, ssim, pcc, elapsed_time))

            if batch_i % self.step_per_epoch ==0:
                epoch = batch_i // self.step_per_epoch
                gen_lr = FLAGS.gen_lr * tf.pow(self.decay_rate, batch_i//self.decay_steps)
                dis_lr = FLAGS.dis_lr * tf.pow(self.decay_rate, batch_i//self.decay_steps)
                with self.summary_writer.as_default():
                    tf.summary.scalar('Learning_rate/G', gen_lr, epoch)
                    tf.summary.scalar('Learning_rate/D', dis_lr, epoch)
                    tf.summary.scalar('Gen/total_loss', G_loss[0], epoch)
                    tf.summary.scalar('Gen/gan_loss', G_loss[1], epoch)
                    tf.summary.scalar('Gen/l1_loss', G_loss[2], epoch)
                    tf.summary.scalar('Dis/total_loss', D_loss[0], epoch)
                    tf.summary.scalar('Dis/real_loss', D_loss[1], epoch)
                    tf.summary.scalar('Dis/fake_loss', D_loss[2], epoch)
                    tf.summary.scalar('Metrics/PSNR', psnr, epoch)
                    tf.summary.scalar('Metrics/SSIM', ssim, epoch)
                    tf.summary.scalar('Metrics/PCC', pcc, epoch)
                    tf.summary.image('input', input, epoch, 4)
                    tf.summary.image('label', label, epoch, 4)
                    tf.summary.image('logit', fake_img, epoch, 4)
                # save the model in each 10 epoches
                if (epoch + 1) % 10 ==0:
                    self.generator.save(FLAGS.logdir + '/model_%02d.h5'%epoch)


            if batch_i == self.train_steps:
                self.generator.save(FLAGS.logdir + '/model.h5')
                break

    """============================ inference =============================="""
    @staticmethod
    def test():
        save_dir = os.path.join('./results', FLAGS.logdir[7:])
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        # test data: images and labels
        images = tf.keras.utils.HDF5Matrix('./data/test_data.hdf5', 'NAC')
        labels = tf.keras.utils.HDF5Matrix('./data/test_data.hdf5', 'AC')
        # load generator from workspace or files in PC
        generator = tf.keras.models.load_model(FLAGS.logdir + '/model.h5',
                                               compile=False)
        # visualization
        logits = generator.predict(images)
        # logits = scale_to_minmax(logits)
        combined = tf.concat((images, logits, labels), 2)
        for i in range(len(logits)):
            print(i)
            tf.keras.preprocessing.image.save_img(
                save_dir + '/%05d.jpg' % i, combined[i])

        nme = NME(combined[:, :, 256:512, :], combined[:, :, 512:, :], False)
        nmse = NMSE(combined[:, :, 256:512, :], combined[:, :, 512:, :], False)
        psnr = PSNR(combined[:, :, 256:512, :], combined[:, :, 512:, :], False)
        ssim = SSIM(combined[:, :, 256:512, :], combined[:, :, 512:, :], False)
        pcc = PCC(combined[:, :, 256:512, :], combined[:, :, 512:, :], False)
        np.savez(FLAGS.logdir + '/metrics.npz', NME=nme, NMSE=nmse, PSNR=psnr, SSIM=ssim, PCC=pcc)

        print(FLAGS.logdir[7:])
        # MAOWUWEI
        nme_1_mean, nme_1_std = tf.reduce_mean(nme[:263]), tf.math.reduce_std(nme[:263])
        nmse_1_mean, nmse_1_std = tf.reduce_mean(nmse[:263]), tf.math.reduce_std(nmse[:263])
        psnr_1_mean, psnr_1_std = tf.reduce_mean(psnr[:263]), tf.math.reduce_std(psnr[:263])
        ssim_1_mean, ssim_1_std = tf.reduce_mean(ssim[:263]), tf.math.reduce_std(ssim[:263])
        pcc_1_mean, pcc_1_std = tf.reduce_mean(pcc[:263]), tf.math.reduce_std(pcc[:263])
        print('ID-1, NME:%.4f(%.4f), NMSE: %.4f(%.4f), PSNR:%.2f(%.2f), SSIM:%.4f(%.4f), '
              'PCC:%.4f(%.4f)' % (nme_1_mean, nme_1_std, nmse_1_mean, nmse_1_std, psnr_1_mean,
                                  psnr_1_std, ssim_1_mean, ssim_1_std, pcc_1_mean, pcc_1_std))
        # HEHUA
        nme_2_mean, nme_2_std = tf.reduce_mean(nme[263:526]), tf.math.reduce_std(nme[263:526])
        nmse_2_mean, nmse_2_std = tf.reduce_mean(nmse[263:526]), tf.math.reduce_std(nmse[263:526])
        psnr_2_mean, psnr_2_std = tf.reduce_mean(psnr[263:526]), tf.math.reduce_std(psnr[263:526])
        ssim_2_mean, ssim_2_std = tf.reduce_mean(ssim[263:526]), tf.math.reduce_std(ssim[263:526])
        pcc_2_mean, pcc_2_std = tf.reduce_mean(pcc[263:526]), tf.math.reduce_std(pcc[263:526])
        print('ID-2, NME:%.4f(%.4f), NMSE: %.4f(%.4f), PSNR:%.2f(%.2f), SSIM:%.4f(%.4f), '
              'PCC:%.4f(%.4f)' % (nme_2_mean, nme_2_std, nmse_2_mean, nmse_2_std, psnr_2_mean,
                                  psnr_2_std, ssim_2_mean, ssim_2_std, pcc_2_mean, pcc_2_std))
        # KONGBO
        nme_3_mean, nme_3_std = tf.reduce_mean(nme[526:825]), tf.math.reduce_std(nme[526:825])
        nmse_3_mean, nmse_3_std = tf.reduce_mean(nmse[526:825]), tf.math.reduce_std(nmse[526:825])
        psnr_3_mean, psnr_3_std = tf.reduce_mean(psnr[526:825]), tf.math.reduce_std(psnr[526:825])
        ssim_3_mean, ssim_3_std = tf.reduce_mean(ssim[526:825]), tf.math.reduce_std(ssim[526:825])
        pcc_3_mean, pcc_3_std = tf.reduce_mean(pcc[526:825]), tf.math.reduce_std(pcc[526:825])
        print('ID-3, NME:%.4f(%.4f), NMSE: %.4f(%.4f), PSNR:%.2f(%.2f), SSIM:%.4f(%.4f), '
              'PCC:%.4f(%.4f)' % (nme_3_mean, nme_3_std, nmse_3_mean, nmse_3_std, psnr_3_mean,
                                  psnr_3_std, ssim_3_mean, ssim_3_std, pcc_3_mean, pcc_3_std))
        # FUCAIXIA
        nme_4_mean, nme_4_std = tf.reduce_mean(nme[825:1124]), tf.math.reduce_std(nme[825:1124])
        nmse_4_mean, nmse_4_std = tf.reduce_mean(nmse[825:1124]), tf.math.reduce_std(nmse[825:1124])
        psnr_4_mean, psnr_4_std = tf.reduce_mean(psnr[825:1124]), tf.math.reduce_std(psnr[825:1124])
        ssim_4_mean, ssim_4_std = tf.reduce_mean(ssim[825:1124]), tf.math.reduce_std(ssim[825:1124])
        pcc_4_mean, pcc_4_std = tf.reduce_mean(pcc[:263]), tf.math.reduce_std(pcc[:263])
        print('ID-4, NME:%.4f(%.4f), NMSE: %.4f(%.4f), PSNR:%.2f(%.2f), SSIM:%.4f(%.4f), '
              'PCC:%.4f(%.4f)' % (nme_4_mean, nme_4_std, nmse_4_mean, nmse_4_std, psnr_4_mean,
                                  psnr_4_std, ssim_4_mean, ssim_4_std, pcc_4_mean, pcc_4_std))
        # CAIZHUAN
        nme_5_mean, nme_5_std = tf.reduce_mean(nme[1124:]), tf.math.reduce_std(nme[1124:])
        nmse_5_mean, nmse_5_std = tf.reduce_mean(nmse[1124:]), tf.math.reduce_std(nmse[1124:])
        psnr_5_mean, psnr_5_std = tf.reduce_mean(psnr[1124:]), tf.math.reduce_std(psnr[1124:])
        ssim_5_mean, ssim_5_std = tf.reduce_mean(ssim[1124:]), tf.math.reduce_std(ssim[1124:])
        pcc_5_mean, pcc_5_std = tf.reduce_mean(pcc[1124:]), tf.math.reduce_std(pcc[1124:])
        print('ID-5, NME:%.4f(%.4f), NMSE: %.4f(%.4f), PSNR:%.2f(%.2f), SSIM:%.4f(%.4f), '
              'PCC:%.4f(%.4f)' % (nme_5_mean, nme_5_std, nmse_5_mean, nmse_5_std, psnr_5_mean,
                                  psnr_5_std, ssim_5_mean, ssim_5_std, pcc_5_mean, pcc_5_std))
        # AVERAGE
        nme_mean, nme_std = tf.reduce_mean(nme), tf.math.reduce_std(nme)
        nmse_mean, nmse_std = tf.reduce_mean(nmse), tf.math.reduce_std(nmse)
        psnr_mean, psnr_std = tf.reduce_mean(psnr), tf.math.reduce_std(psnr)
        ssim_mean, ssim_std = tf.reduce_mean(ssim), tf.math.reduce_std(ssim)
        pcc_mean, pcc_std = tf.reduce_mean(pcc), tf.math.reduce_std(pcc)
        print('Average, NME:%.4f(%.4f), NMSE: %.4f(%.4f), PSNR:%.2f(%.2f), SSIM:%.4f(%.4f), '
              'PCC:%.4f(%.4f)' % (nme_mean, nme_std, nmse_mean, nmse_std, psnr_mean,
                                  psnr_std, ssim_mean, ssim_std, pcc_mean, pcc_std))


if __name__ == '__main__':
    if os.path.exists(FLAGS.logdir):
        shutil.rmtree(FLAGS.logdir)
    gan = Pix2Pix()
    gan.train()
    gan.test()