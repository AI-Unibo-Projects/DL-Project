from __future__ import print_function

import os
import os.path
import matplotlib.image as mpimg
import tensorflow as tf

tf.compat.v1.flags.DEFINE_string("fn_root", "", "Name of root file path.")
tf.compat.v1.flags.DEFINE_string("partition_fn", "", "Partition file path.")
tf.compat.v1.flags.DEFINE_string("number", "", "Number of files.")

FLAGS = tf.compat.v1.flags.FLAGS


def map_to_boolean(n):
    if n == "1":
        return "True"
    else:
        return "False"


def main():
    """Main converter function."""
    if not os.path.exists('tfrecs'):
        os.makedirs('tfrecs')

    # Celeb A
    with open(FLAGS.partition_fn, "r") as infile:
        img_fn_list = infile.readlines()

    attributes_name = img_fn_list[1].split()

    img_fn_list = img_fn_list[2:]
    img_fn_list = [elem.strip().split() for elem in img_fn_list]

    fn_root = FLAGS.fn_root
    num_examples = len(img_fn_list)
    num_exaple_per_file = len(img_fn_list) // int(FLAGS.number)
    i = 0
    file_out = "tfrecord_{}.tfrec".format(i)
    print(file_out)
    writer = tf.io.TFRecordWriter("tfrecs/" + file_out)

    for example_idx, img_fn in enumerate(img_fn_list):
        if example_idx % 1000 == 0:
            print(example_idx, "/", num_examples)
        if example_idx != 0 and example_idx % num_exaple_per_file == 0:
            writer.close()
            i += 1
            file_out = "tfrecord_{}.tfrec".format(i)
            print(file_out)
            writer = tf.io.TFRecordWriter("tfrecs/" + file_out)
        img_path = os.path.join(fn_root, img_fn[0])
        img_shape = mpimg.imread(img_path).shape
        filename = os.path.basename(img_path)

        # Read image data in terms of bytes
        with tf.io.gfile.GFile(img_path, 'rb') as fid:
            image_data = fid.read()

        image_attributes = list(map(map_to_boolean, img_fn[1:]))

        attributes = []
        for i in range(len(attributes_name)):
            attributes.append(bytes(attributes_name[i], 'utf-8'))
            attributes.append(bytes(image_attributes[i], 'utf-8'))

        # attributes = [i for j in zip(attributes_name, image_attributes) for i in j]

        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[0]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[1]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[2]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            'attributes': tf.train.Feature(bytes_list=tf.train.BytesList(value=attributes))
        }))

        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    main()
