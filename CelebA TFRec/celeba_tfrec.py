from __future__ import print_function

import os
import os.path
import imageio
import matplotlib.image as mpimg
import tensorflow as tf



tf.compat.v1.flags.DEFINE_string("fn_root", "", "Name of root file path.")
tf.compat.v1.flags.DEFINE_string("partition_fn", "", "Partition file path.")
tf.compat.v1.flags.DEFINE_string("number", "", "Number of files.")


FLAGS = tf.compat.v1.flags.FLAGS


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
list_attr = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]


def main():
	"""Main converter function."""
	# Celeb A
	with open(FLAGS.partition_fn, "r") as infile:
		img_fn_list = infile.readlines()
		
	img_fn_list = [elem.strip().split() for elem in img_fn_list]
	
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
		img_path = os.path.join(fn_root, img_fn[0])
		img_shape = mpimg.imread(img_path).shape
		filename = os.path.basename(img_path)
		
		# Read image data in terms of bytes
		with tf.io.gfile.GFile(img_path, 'rb') as fid:
			image_data = fid.read()
			
		attr = img_fn[1:]
			
		example = tf.train.Example(features = tf.train.Features(feature = {
			'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
			'height': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
			'width': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
			'depth': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[2]])),
			'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
		}))
		
		
		
		writer.write(example.SerializeToString())
	writer.close()


if __name__ == "__main__":
	main()