#####
# Original Sources: 
# https://github.com/MarcCoru/MTLCC/blob/master/S2parser.py
# https://github.com/michaeltrs/DeepSatModels/blob/782f7363e04b9e9d3dfc18a1ce6609157613b5cf/data/MTLCC/S2parser.py
# Adaptation: Joseph Metcalfe
# Purpose: Used to read the .tfrecords by tfrecord2npy
#####


import pickle
import sys
import numpy as np
import tensorflow as tf


class S2parser:
    """Defines the Sentinel-2 .tfrecord format for MTLCC data."""

    def __init__(self):
        self.feature_format = {
            'x10/data':    tf.io.FixedLenFeature([], tf.string),
            'x10/shape':   tf.io.FixedLenFeature([4], tf.int64),
            'x20/data':    tf.io.FixedLenFeature([], tf.string),
            'x20/shape':   tf.io.FixedLenFeature([4], tf.int64),
            'x60/data':    tf.io.FixedLenFeature([], tf.string),
            'x60/shape':   tf.io.FixedLenFeature([4], tf.int64),
            'dates/doy':   tf.io.FixedLenFeature([], tf.string),
            'dates/year':  tf.io.FixedLenFeature([], tf.string),
            'dates/shape': tf.io.FixedLenFeature([1], tf.int64),
            'labels/data': tf.io.FixedLenFeature([], tf.string),
            'labels/shape':tf.io.FixedLenFeature([3], tf.int64),
        }

    def write(self, filename, x10, x20, x60, doy, year, labels):
        """Serialise arrays to a .tfrecord file."""
        x10    = x10.astype(np.int64)
        x20    = x20.astype(np.int64)
        x60    = x60.astype(np.int64)
        doy    = doy.astype(np.int64)
        year   = year.astype(np.int64)
        labels = labels.astype(np.int64)

        feature = {
            'x10/data':    tf.train.Feature(bytes_list=tf.train.BytesList(value=[x10.tobytes()])),
            'x10/shape':   tf.train.Feature(int64_list=tf.train.Int64List(value=x10.shape)),
            'x20/data':    tf.train.Feature(bytes_list=tf.train.BytesList(value=[x20.tobytes()])),
            'x20/shape':   tf.train.Feature(int64_list=tf.train.Int64List(value=x20.shape)),
            'x60/data':    tf.train.Feature(bytes_list=tf.train.BytesList(value=[x60.tobytes()])),
            'x60/shape':   tf.train.Feature(int64_list=tf.train.Int64List(value=x60.shape)),
            'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
            'labels/shape':tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
            'dates/doy':   tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
            'dates/year':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
            'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape)),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        with tf.io.TFRecordWriter(filename) as writer:
            writer.write(example.SerializeToString())

        sys.stdout.flush()

    def parse_example(self, serialized_example):
        """Parse a single serialised tf.train.Example into tensors."""
        feature = tf.io.parse_single_example(serialized_example, self.feature_format)

        x10    = tf.reshape(tf.io.decode_raw(feature['x10/data'],    tf.int64), tf.cast(feature['x10/shape'],    tf.int32))
        x20    = tf.reshape(tf.io.decode_raw(feature['x20/data'],    tf.int64), tf.cast(feature['x20/shape'],    tf.int32))
        x60    = tf.reshape(tf.io.decode_raw(feature['x60/data'],    tf.int64), tf.cast(feature['x60/shape'],    tf.int32))
        doy    = tf.reshape(tf.io.decode_raw(feature['dates/doy'],   tf.int64), tf.cast(feature['dates/shape'],  tf.int32))
        year   = tf.reshape(tf.io.decode_raw(feature['dates/year'],  tf.int64), tf.cast(feature['dates/shape'],  tf.int32))
        labels = tf.reshape(tf.io.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

        return x10, x20, x60, doy, year, labels

    def read_and_return(self, filename):
        """Read a single .tfrecord file and return numpy arrays."""
        dataset = tf.data.TFRecordDataset([filename])
        dataset = dataset.map(self.parse_example)
        example = next(iter(dataset))
        return tuple(t.numpy() for t in example)

    def get_shapes(self, sample):
        """Print and return the shapes of all tensors in a sample file."""
        print(f"Reading shapes from: {sample}")
        data = self.read_and_return(sample)
        shapes = [t.shape for t in data]
        for name, shape in zip(['x10', 'x20', 'x60', 'doy', 'year', 'labels'], shapes):
            print(f"  {name}: {shape}")
        return shapes

    def tfrecord_to_pickle(self, tfrecord_name, pickle_name):
        """Convert a .tfrecord file to a pickle file."""
        data = self.read_and_return(tfrecord_name)
        with open(pickle_name, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved pickle to {pickle_name}")


if __name__ == '__main__':
    parser = S2parser()
    parser.tfrecord_to_pickle("1.tfrecord", "1.pkl")
