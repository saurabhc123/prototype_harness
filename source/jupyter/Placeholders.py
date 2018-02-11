import tensorflow as tf


img_dim = 48
n_channels = 3
adience_img_feature_width = 64
img_feature_width = 64

n_inputs = 20  # word vector dimension
n_steps = 80  # number of words fed to each RNN. We will feed the profile and tweet together

text_feature_length = n_steps * n_inputs
profile_color_feature_length = 6
feature_width = img_feature_width + text_feature_length + profile_color_feature_length
other_features_width = img_feature_width  + profile_color_feature_length


adience_keep_prob = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

best_accuracy_so_far = 0.0


n_neurons = 300
learning_rate = 0.01

n_classes = 2
word_vec_dim = 300
hidden1_neurons = 150
hidden2_neurons = 50

sentence = tf.placeholder(tf.float32, shape=[None, word_vec_dim])
class_labels = tf.placeholder(tf.float32, shape=[None, n_classes])
class_centroids = tf.placeholder(tf.float32, shape=[n_classes,word_vec_dim])
