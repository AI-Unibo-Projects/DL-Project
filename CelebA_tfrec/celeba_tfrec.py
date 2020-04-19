from __future__ import print_function

import os
import os.path
import imageio
import scipy.io
import scipy.io.wavfile
import scipy.ndimage
import tensorflow as tf



tf.compat.v1.flags.DEFINE_string("fn_root", "", "Name of root file path.")
tf.compat.v1.flags.DEFINE_string("partition_fn", "", "Partition file path.")
tf.compat.v1.flags.DEFINE_string("number", "", "Number of files.")


FLAGS = tf.compat.v1.flags.FLAGS


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
	"""Main converter function."""
	# Celeb A
	with open(FLAGS.partition_fn, "r") as infile:
		img_fn_list = infile.readlines()
		
	img_fn_list = [elem.strip() for elem in img_fn_list]
	
	fn_root = FLAGS.fn_root
	num_examples = len(img_fn_list)
	num_exaple_per_file = len(img_fn_list)//int(FLAGS.number)	
	i = 0
	file_out = "tfrecord_{}.tfrec".format(i)
	print(file_out)
	writer = tf.io.TFRecordWriter(file_out)
	
	
	for example_idx, img_fn in enumerate(img_fn_list):
		if example_idx % 1000 == 0:
			print(example_idx, "/", num_examples)
		if example_idx!=0 and example_idx % num_exaple_per_file == 0:
			writer.close()
			i += 1
			file_out = "tfrecord_{}.tfrec".format(i)
			print(file_out)
			writer = tf.io.TFRecordWriter(file_out)			
		image_raw = imageio.imread(os.path.join(fn_root, img_fn))
		rows = image_raw.shape[0]
		cols = image_raw.shape[1]
		depth = image_raw.shape[2]
		image_raw = image_raw.tostring()
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					"height": _int64_feature(rows),
					"width": _int64_feature(cols),
					"depth": _int64_feature(depth),
					"image_raw": _bytes_feature(image_raw)
				}
			)
		)
		writer.write(example.SerializeToString())
	writer.close()


if __name__ == "__main__":
	main()