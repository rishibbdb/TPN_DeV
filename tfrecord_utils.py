import tensorflow as tf
import numpy as np
import glob
import os
import copy
from sklearn.utils import shuffle

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def serialize_example(x, y):
    data = {
            "input_features": _bytes_feature(serialize_array(x)),
            "labels":  _bytes_feature(serialize_array(y)),
          }
    example = tf.train.Example(features=tf.train.Features(feature=data))
    return example.SerializeToString()


def parse_tfr_element(element, n_features=7, n_labels=105):
    data = {
        'labels': tf.io.FixedLenFeature([], tf.string),
        'input_features': tf.io.FixedLenFeature([], tf.string),
      }

    content = tf.io.parse_single_example(element, data)
    labels = content['labels']
    features = content['input_features']

    feature = tf.io.parse_tensor(features, out_type=tf.float32)
    feature = tf.ensure_shape(feature, (n_features,))

    label = tf.io.parse_tensor(labels, out_type=tf.float32)
    label = tf.ensure_shape(label, (n_labels,))

    return (feature, label)


def tfrecords_reader_dataset(infiles, n_readers=16, block_length=1, batch_size=8, shuffle_buffer=16):
    dataset = tf.data.Dataset.list_files(infiles, shuffle=True)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(325)
    dataset = dataset.interleave(
       lambda filename: tf.data.TFRecordDataset(filename, compression_type=''),
        cycle_length=n_readers,
        block_length=block_length,
        num_parallel_calls=tf.data.AUTOTUNE, # opt
        deterministic=False) # opt

    dataset = dataset.map(parse_tfr_element,
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)

def parse_numpy(indir = "/home/storage/hans/photondata/fullazisupport/npy", batch_size=8, shuffle_buffer=16, cartesian=False):
    x_train = []
    y_train = []

    infiles_x = sorted(glob.glob(os.path.join(indir, "x_train*.npy")))
    infiles_y = sorted(glob.glob(os.path.join(indir, "y_train*.npy")))

    for inf_x, inf_y in zip(infiles_x, infiles_y):
        x = np.load(inf_x)
        y = np.load(inf_y)

        # scale inputs fall within [-1, 1]
        km_scale = 1000
        if cartesian:
            km_scale = 1000
            x[:, 0] /= km_scale # x relative to track
            x[:, 3] /= km_scale # z relative to track
            # no need to scale
            # unit vector direction of track
            # since unit vector components
            # are scaled to [-1, 1]

        else:
            x[:, 0] /= km_scale
            x[:, 1] /= km_scale
            # normalize radians
            angle_scale = np.pi
            x[:, 2] /= angle_scale
            x[:, 4] /= angle_scale
            # note: cos(zenith) at column index 3 is already within [-1, 1]

        idx = np.isfinite(y.sum(axis=1))

        # and select reasonable parameter range for now.
        if cartesian:
            idx1 = np.logical_and(x[:, 0] > 1 / km_scale, x[:, 0] < 500 / km_scale)
            idx2 = np.logical_and(x[:, 3] > -800 / km_scale, x[:, 3] < 800 / km_scale)
            idx3 = np.logical_and(idx1, idx2)

        else:
            idx1 = np.logical_and(x[:, 0] > 1 / km_scale, x[:, 0] < 400 / km_scale)
            idx2 = np.logical_and(x[:, 1] > -800 / km_scale, x[:, 1] < 800 / km_scale)
            idx3 = np.logical_and(idx1, idx2)

        idx = np.logical_and(idx, idx3)

        x_train.append(copy.copy(x[idx]))
        y_train.append(copy.copy(y[idx]))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_train, y_train = shuffle(x_train, y_train)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.repeat()
    dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)