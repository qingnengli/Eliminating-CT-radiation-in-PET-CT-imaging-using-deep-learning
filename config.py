#-*- coding: UTF-8 -*-
"""
    Name: Qingneng Li (Eng: Lional)
    Time: 2019/12/27
    Place: SIAT, Shenzhen
    Item: NAC-PET --> AC-PET --> CT

"""

import tensorflow.compat.v1 as tfv
tfv.app.flags.DEFINE_string(
    'GPU', '1', 'the order of using GPU')

tfv.app.flags.DEFINE_string(
    'logdir', './logs/baseline',
    'the dir for tensorboard log files')

tfv.app.flags.DEFINE_string(
    'train_file', './data/train_data.hdf5',
    'the file of train dataset')

tfv.app.flags.DEFINE_string(
    'test_file', './data/test_data.hdf5',
    'the file of test dataset')

tfv.app.flags.DEFINE_string(
    'data_dir', './data/',
    'the dir of train/valid/test dataset')

tfv.app.flags.DEFINE_string(
    'input_name', 'UMAP',
    'the name of input name, one of ["NAC","umap"]')

tfv.app.flags.DEFINE_string(
    'output_name', 'CT',
    'the name of output name, one of ["AC","CT"]')

tfv.app.flags.DEFINE_integer(
    'batch_size', 4, 'The training batch size.')

tfv.app.flags.DEFINE_integer(
    'img_size', 256, 'The size of images.')

tfv.app.flags.DEFINE_integer(
    'img_ch', 3, 'The channels of images.')

tfv.app.flags.DEFINE_integer(
    'split_data', 5760,
    'The number of train data spitted'
    ' from whole train_valid data.')

tfv.app.flags.DEFINE_integer(
    'num_epoch', 200, 'The number of epoach in train.')

tfv.app.flags.DEFINE_float(
    'gen_lr', 1e-4, 'Initial learning rate in generator.')

tfv.app.flags.DEFINE_float(
    'dis_lr', 1e-4, 'Initial learning rate in discriminator.')

tfv.app.flags.DEFINE_bool(
    'finetune', False, 'whether fine tuning or not')

tfv.app.flags.DEFINE_string(
    'phase', 'test', 'whether train phase or test phase')

tfv.app.flags.DEFINE_integer(
    'memory_limit', None, 'whether train phase or test phase')

FLAGS = tfv.app.flags.FLAGS
