#-*- coding: UTF-8 -*-
"""
    Name: Qingneng Li (Eng: Lional)
    Time: 2020/10/07
    Place: SIAT, Shenzhen
    Item: Deep Multi-modality
"""
from datetime import datetime
from copy import deepcopy
import tensorflow as tf
import numpy as np

from config import FLAGS
from utils import *
from models import *

"""================================= Main ==================================="""
class CNN():
    def __init__(self):
        # Data Loader
        if FLAGS.phase == 'test':
            data = np.load('./data/' + FLAGS.logdir[7:] + '.npz')
            train_paths = list(data['train'])
            valid_paths = list(data['valid'])
            infer_paths = list(data['infer'])
        else:
            x_path = glob_image_from_folder(FLAGS.data_dir + FLAGS.input_name)
            y_path = glob_image_from_folder(FLAGS.data_dir + FLAGS.output_name)
            train_paths, valid_paths, infer_paths = \
                split_dataset(list(zip(x_path, y_path)))
            np.savez('./data/' + FLAGS.logdir[7:] + '.npz',
                     train=train_paths, valid=valid_paths, infer=infer_paths)
        x_train, y_train = list(zip(*train_paths))
        x_valid, y_valid = list(zip(*valid_paths))
        train_dataset = dataset_loader((list(x_train), list(y_train)),
                                       read_pairs_with_augment)
        valid_dataset = dataset_loader((list(x_valid), list(y_valid)),
                                       read_pairs_with_augment)
        self.train_iter = iter(train_dataset)
        self.valid_iter = iter(valid_dataset)
        self.test_paths = infer_paths

        self.step_per_epoch = len(train_paths) // FLAGS.batch_size
        self.iteration = self.step_per_epoch * FLAGS.num_epoch

        # Build Model
        self.G = UNet_AE(FLAGS.img_size, FLAGS.img_ch, False, hidden_dims=512)
        self.G_ema = deepcopy(self.G)
        self.G.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        self.G_ema.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        G_params = self.G.count_params()

        # Optimizer
        g_lr_schedule = LR_schedule(self.iteration, FLAGS.gen_lr, 'Piecewise')
        self.G_optimizer = tf.keras.optimizers.RMSprop(g_lr_schedule)
        self.gen_tv = self.G.trainable_variables

        # PCP set
        self.VGG19 = tf.keras.applications.VGG19(include_top=False,
                                                 weights='imagenet')
        self.VGG_layers = ['block1_pool','block2_pool','block3_pool',
                           'block4_pool','block5_pool']

        # Checkpoints
        self.ckpt, self.start_iteration = tf.train.Checkpoint(G_ema=self.G_ema), 0
        self.manager = tf.train.CheckpointManager(self.ckpt, FLAGS.logdir, max_to_keep=1)
        if FLAGS.finetune:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
            print('Latest checkpoint restored! start iteration: ', self.start_iteration)

        # Save dirs
        self.save_dir = os.path.join('./outs', FLAGS.logdir[7:])
        self.tmp_dir = os.path.join(self.save_dir, 'tmp')

        # Logging information
        print('*****************************************************************')
        print("Train Dataset number: ", len(train_paths))
        print("Valid Dataset number: ", len(valid_paths))
        print('Train %d Epoches (%d Iterations), %d Iteration per epoch'
              %(FLAGS.num_epoch, self.iteration, self.step_per_epoch))
        print('*****************************************************************')
        print("G network parameters: ", G_params)
        print('*****************************************************************')
        print('Initial G learning rate: %5g'% FLAGS.gen_lr)

    """================================= Loss ==================================="""
    def gen_mae_loss(self, real, fake):
        mae_loss = tf.reduce_mean(tf.abs(real-fake))
        return 10.0 * mae_loss

    def gen_pcp_loss(self, real, fake):
        pcp_loss = PCP(self.VGG19, self.VGG_layers).Loss(real, fake)
        return 1.0 * pcp_loss

    def gen_reg_loss(self):
        reg_G = tf.nn.scale_regularization_loss(self.G.losses)
        return 1.0 * reg_G

    """=========================== Every Train Step ================================"""
    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape(persistent=True) as gen_tape:
            # input: [PET, MRI], B, H, W, 2*C
            # logit: [MRI, PET], B, H, W, 2*C
            logit = self.G(image)
            logit = tf.tanh(logit)

            g_mae_loss = self.gen_mae_loss(label, logit)
            g_pcp_loss = self.gen_pcp_loss(label, logit)
            g_reg_loss = self.gen_reg_loss()
            g_loss = g_mae_loss + g_reg_loss + g_pcp_loss

            # Metrics
            norm_image = tf.clip_by_value(0.5 * (image + 1), 0, 1)
            norm_logit = tf.clip_by_value(0.5 * (logit + 1), 0, 1)
            norm_label = tf.clip_by_value(0.5 * (label + 1), 0, 1)
            psnr = tf.reduce_mean(tf.image.psnr(norm_logit, norm_label, 1.0))
            ssim = tf.reduce_mean(tf.image.ssim(norm_logit, norm_label, 1.0))
            pcc = PCC(norm_logit, norm_label, True)
            mae = tf.reduce_mean(tf.abs(norm_logit-norm_label))

        g_gradient = gen_tape.gradient(g_loss, self.gen_tv)
        self.G_optimizer.apply_gradients(zip(g_gradient, self.gen_tv))

        return g_loss, g_mae_loss, g_pcp_loss, g_reg_loss,\
               psnr, ssim, mae, pcc, \
               norm_image, norm_label, norm_logit

    @tf.function
    def valid_step(self, image, label):
        logit = self.G_ema(image)
        logit = tf.tanh(logit)
        # Loss
        mae_loss = self.gen_mae_loss(label, logit)
        pcp_loss = self.gen_pcp_loss(label, logit)
        reg_loss = self.gen_reg_loss()
        loss = mae_loss + reg_loss + pcp_loss
        # Indicates (Metrics)
        norm_image = tf.clip_by_value(0.5 * (image + 1), 0, 1)
        norm_logit = tf.clip_by_value(0.5 * (logit + 1), 0, 1)
        norm_label = tf.clip_by_value(0.5 * (label + 1), 0, 1)
        psnr = tf.reduce_mean(tf.image.psnr(norm_logit, norm_label, 1.0))
        ssim = tf.reduce_mean(tf.image.ssim(norm_logit, norm_label, 1.0))
        mae = tf.reduce_mean(tf.abs(norm_logit - norm_label))
        pcc = PCC(norm_logit, norm_label, True)
        return norm_image, norm_label, norm_logit, \
               loss, mae_loss, pcp_loss, reg_loss, \
               psnr, ssim, mae, pcc

    @tf.function
    def moving_average(self, model, model_test, beta=0.999):
        update_weight = model.trainable_weights
        previous_weight = model_test.trainable_weights
        for new_param, pre_param in zip(update_weight, previous_weight):
            average_param = beta * pre_param + (1 - beta) * new_param
            pre_param.assign(average_param)

    """============================ Update Training ==============================="""
    def train(self):
        self.summary_1 = tf.summary.create_file_writer(FLAGS.logdir + '/train')
        self.summary_2 = tf.summary.create_file_writer(FLAGS.logdir + '/valid')
        start_time = datetime.now()
        for idx in range(self.start_iteration, self.iteration):
            image, label = next(self.train_iter)
            # PET-MRI
            g_loss, g_mae_loss, g_pcp_loss, g_reg_loss, \
            psnr, ssim, mae, pcc, I, L, F =  self.train_step(image, label)

            self.moving_average(self.G, self.G_ema)

            epoch = idx // self.step_per_epoch
            info = "[Iter %6d/%6d(%d)] [G: %.4f]" \
                   "[PSNR/SSIM/MAE/PCC: %.4g/%.4g/%.4g/%.4g]" \
                   "[time: %s]" % (idx, self.iteration, epoch, g_loss, psnr,
                     ssim*100, mae*100,pcc*100, datetime.now() - start_time)
            fprint(FLAGS.logdir + '/log.txt', info)

            if idx % self.step_per_epoch ==0:
                IMAGE, LABEL = next(self.valid_iter)
                IMAGE, LABEL, LOGIT, ALL_LOSS, MAE_LOSS, PCP_LOSS, REG_LOSS, \
                Psnr, Ssim, Mae, Pcc = self.valid_step(IMAGE, LABEL)

                with self.summary_1.as_default():
                    tf.summary.scalar('Lr/G', self.G_optimizer.lr(idx), epoch)
                    tf.summary.scalar('G/All_loss', g_loss, step=epoch)
                    tf.summary.scalar('G/Mae_loss', g_mae_loss, step=epoch)
                    tf.summary.scalar('G/PCP_loss', g_pcp_loss, step=epoch)
                    tf.summary.scalar('G/Reg_loss', g_reg_loss, step=epoch)
                    tf.summary.scalar('Indicator/PSNR', psnr, step=epoch)
                    tf.summary.scalar('Indicator/SSIM', ssim, step=epoch)
                    tf.summary.scalar('Indicator/MAE',   mae, step=epoch)
                    tf.summary.scalar('Indicator/PCC',   pcc, step=epoch)
                    tf.summary.image('Image', I, epoch, 4)
                    tf.summary.image('Label', L, epoch, 4)
                    tf.summary.image('Logit', F, epoch, 4)

                with self.summary_2.as_default():
                    tf.summary.scalar('G/All_loss', ALL_LOSS, step=epoch)
                    tf.summary.scalar('G/Mae_loss', MAE_LOSS, step=epoch)
                    tf.summary.scalar('G/PCP_loss', PCP_LOSS, step=epoch)
                    tf.summary.scalar('G/Reg_loss', REG_LOSS, step=epoch)
                    tf.summary.scalar('Indicator/PSNR', Psnr, epoch)
                    tf.summary.scalar('Indicator/SSIM', Ssim, epoch)
                    tf.summary.scalar('Indicator/MAE', Mae, epoch)
                    tf.summary.scalar('Indicator/PCC', Pcc, epoch)
                    tf.summary.image('Image', IMAGE, epoch, 4)
                    tf.summary.image('Label', LABEL, epoch, 4)
                    tf.summary.image('Logit', LOGIT, epoch, 4)

                # save every self.save_freq
                if epoch % 10 == 0:
                    self.manager.save(checkpoint_number=idx)

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    """============================ Testing and Results ==============================="""
    def test(self):
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

        psnr, ssim, mae, rmse, pcc = [], [], [], [], []
        for i, (input_path, label_path) in enumerate(self.test_paths):
            print(i)
            print(input_path, label_path)
            image, label = read_pairs_without_augment(input_path,
                                                      label_path)

            logit = tf.tanh(self.G_ema(image[None])[0])

            # normalize to [0, 1]
            image = tf.clip_by_value(0.5 * image + 0.5, 0, 1)
            label = tf.clip_by_value(0.5 * label + 0.5, 0, 1)
            logit = tf.clip_by_value(0.5 * logit + 0.5, 0, 1)

            psnr.append(tf.reduce_mean(tf.image.psnr(label, logit, 1.0)).numpy())
            ssim.append(tf.reduce_mean(tf.image.ssim(label, logit, 1.0)).numpy())
            mae.append(tf.reduce_mean(tf.abs(label - logit)).numpy())
            rmse.append(RMSE(label, logit).numpy())
            pcc.append(PCC(logit, label, True).numpy())

            comb = tf.concat((image, label, logit), 1)
            save_name = self.save_dir + '/%04d.jpg'%i
            tf.keras.preprocessing.image.save_img(save_name, comb)

        # np.savez(self.save_dir + '/metrics.npz',
        #          psnr=psnr, ssim=ssim, rmse=rmse, mae=mae, pcc=pcc)
        save_csv(self.save_dir + '/metrics.csv',
                 {'psnr': psnr, 'ssim':ssim, 'rmse':rmse, 'mae':mae, 'pcc':pcc})

        save_file = self.save_dir + '/results.txt'
        fprint(save_file, 'Save_dir, %s -> %s, %s'
               % (FLAGS.input_name, FLAGS.output_name, str(datetime.now())))
        fprint(save_file, '==================================================')
        fprint(save_file, 'PSNR = %.4f + %.4fdB' % list_mean_std(psnr, 1))
        fprint(save_file, 'SSIM = %.4f + %.4f%%' % list_mean_std(ssim, 100))
        fprint(save_file, 'RMSE = %.4f + %.4f%%' % list_mean_std(rmse, 100))
        fprint(save_file, 'MAE = %.4f + %.4f%%' % list_mean_std(mae, 100))
        fprint(save_file, 'PCC = %.4f + %.4f%%' % list_mean_std(pcc, 100))
        fprint(save_file, '==================================================')


    def test_once(self):
        if not os.path.exists(self.tmp_dir): os.makedirs(self.tmp_dir)
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

        psnr, ssim, mae, rmse, pcc = [], [], [], [], []
        input_dir = os.path.join(FLAGS.data_dir, FLAGS.input_name, 'test')
        output_dir = os.path.join(FLAGS.data_dir, FLAGS.output_name, 'test')
        files = os.listdir(input_dir)
        for f in files:
            input_path = os.path.join(input_dir, f)
            output_path = os.path.join(output_dir, f)
            image, label = read_pairs_without_augment(input_path, output_path)
            logit = tf.tanh(self.G_ema(image[None])[0])
            logit = tf.clip_by_value(0.5 * logit + 0.5, 0, 1)
            image = tf.clip_by_value(0.5 * image + 0.5, 0, 1)
            label = tf.clip_by_value(0.5 * label + 0.5, 0, 1)
            comb = tf.concat((image, label, logit), 1)
            tf.keras.preprocessing.image.save_img(os.path.join(self.tmp_dir, f), comb)
            print('Finish saving ', f)
            psnr.append(tf.reduce_mean(tf.image.psnr(label, logit, 1.0)).numpy())
            ssim.append(tf.reduce_mean(tf.image.ssim(label, logit, 1.0)).numpy())
            mae.append(tf.reduce_mean(tf.abs(label - logit)).numpy())
            rmse.append(RMSE(label, logit).numpy())
            pcc.append(PCC(logit, label, True).numpy())

        save_csv(self.tmp_dir + '/metrics.csv',
                 {'psnr': psnr, 'ssim':ssim, 'rmse':rmse, 'mae':mae, 'pcc':pcc})

        save_file = self.save_dir + '/results.txt'
        fprint(save_file, 'Tmp_dir, %s -> %s, %s'
               % (FLAGS.input_name, FLAGS.output_name, str(datetime.now())))
        fprint(save_file, '==================================================')
        fprint(save_file, 'PSNR = %.4f + %.4fdB' % list_mean_std(psnr, 1))
        fprint(save_file, 'SSIM = %.4f + %.4f%%' % list_mean_std(ssim, 100))
        fprint(save_file, 'RMSE = %.4f + %.4f%%' % list_mean_std(rmse, 100))
        fprint(save_file, 'MAE = %.4f + %.4f%%' % list_mean_std(mae, 100))
        fprint(save_file, 'PCC = %.4f + %.4f%%' % list_mean_std(pcc, 100))
        fprint(save_file, '==================================================')


"""============================ RUN ==============================="""

if __name__ == '__main__':
    run_on_gpu()
    check_logdir()
    cnn = CNN()
    if FLAGS.phase == 'train':
        cnn.train()
    cnn.test()
    cnn.test_once()

