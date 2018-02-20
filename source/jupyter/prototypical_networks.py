import gensim
import numpy as np
import string
import tensorflow as tf
import tensorflow.contrib.slim as slim
import Placeholders
import datetime
import random as random

x = Placeholders.sentence
y_ = Placeholders.class_labels
c = Placeholders.class_centroids


def get_cluster_based_labels1(fc_layer_representation):
    indexed_centroids = tf.matmul(y_,c)
    print(indexed_centroids)
    x_distances = tf.exp(-(tf.square(fc_layer_representation - indexed_centroids)))
    sum_x_distances = tf.log(tf.reduce_sum(x_distances))
    logits = tf.divide(x_distances, sum_x_distances)
    return logits

def get_cluster_based_labels_revised(fc_layer_representation):
    #centroids_0 = tf.matmul(tf.ones(fc_layer_representation.get_shape()[0]),c[0])
    #centroids_1 = tf.matmul(tf.ones(fc_layer_representation.get_shape()[0]),c[1])
    centroids_0 = c[0]
    centroids_1 = c[1]
    centroid_0_differences  = tf.reduce_sum(tf.exp(-(tf.square(fc_layer_representation - centroids_0))),1) #[n x 1]
    centroid_1_differences  = tf.reduce_sum(tf.exp(-(tf.square(fc_layer_representation - centroids_1))),1) #[n x 1]
    centroid_0_sum = tf.reduce_sum(centroid_0_differences)
    centroid_1_sum = tf.reduce_sum(centroid_1_differences)
    centroid_0_differences = tf.divide(centroid_0_differences, centroid_0_sum)
    centroid_1_differences = tf.divide(centroid_1_differences, centroid_1_sum)
    #print(indexed_centroids)
    #x_distances = tf.exp(-(tf.square(fc_layer_representation - indexed_centroids)))
    #sum_x_distances = tf.log(tf.reduce_sum(x_distances))
    #logits = tf.divide(x_distances, sum_x_distances)
    #logits = tf.nn.softmax(-tf.stack([centroid_0_differences, centroid_1_differences],1))
    logits = tf.stack([centroid_0_differences, centroid_1_differences],1)
    logits_sum = tf.reduce_sum(logits,1)
    final_logits = logits/tf.reshape(logits_sum, (-1, 1))
    return final_logits

def get_cluster_based_labels(fc_layer_representation):
    #centroids_0 = tf.matmul(tf.ones(fc_layer_representation.get_shape()[0]),c[0])
    #centroids_1 = tf.matmul(tf.ones(fc_layer_representation.get_shape()[0]),c[1])
    centroids_0 = c[0]
    centroids_1 = c[1]
    centroid_0_differences  = tf.reduce_mean(tf.square(fc_layer_representation - centroids_0),1) #[n x 1]
    centroid_1_differences  = tf.reduce_mean(tf.square(fc_layer_representation - centroids_1),1) #[n x 1]
    #centroid_0_sum = tf.reduce_sum(centroid_0_differences)
    #centroid_1_sum = tf.reduce_sum(centroid_1_differences)
    #centroid_0_differences = tf.divide(centroid_0_differences, centroid_0_sum)
    #centroid_1_differences = tf.divide(centroid_1_differences, centroid_1_sum)
    #print(indexed_centroids)
    #x_distances = tf.exp(-(tf.square(fc_layer_representation - indexed_centroids)))
    #sum_x_distances = tf.log(tf.reduce_sum(x_distances))
    #logits = tf.divide(x_distances, sum_x_distances)
    logits = tf.nn.log_softmax(-tf.stack([centroid_0_differences, centroid_1_differences],1))
    return logits

def get_cluster_distance_loss(fc_layer_representation, centroid):
    x_distances = tf.square(fc_layer_representation - centroid)
    mean_x_distances = tf.reduce_mean(x_distances)
    return mean_x_distances

def one_hot(vec, vals=Placeholders.n_classes):
    n = vec.shape[1]
    out = np.zeros((n, vals))
    out[np.arange(n), vec] = 1
    return out

def get_random_examples(data,labels,number_of_examples,label=0):
    vector_encoded_labels = np.argmax(labels, axis=1)
    requested_labeled_examples = data[vector_encoded_labels==label,:]
    requested_labels = labels[vector_encoded_labels==label,:]
    total_examples = requested_labeled_examples.shape[0]
    random_index = random.randint(0,total_examples - number_of_examples - 1)
    result = requested_labeled_examples[random_index:random_index+number_of_examples,:]
    labels = requested_labels[random_index:random_index+number_of_examples,:]
    return result, labels

def get_centroid(support_examples,hidden2,sess):
    hidden_vectors = sess.run(hidden2, feed_dict={x: support_examples})
    centroid = np.mean(hidden_vectors,axis = 0)
    return centroid

def train(sess):
    he_init = slim.variance_scaling_initializer()
    hidden1 =tf.layers.dense(Placeholders.sentence, Placeholders.hidden1_neurons,activation=tf.nn.relu,kernel_initializer=he_init)
    hidden2 =tf.layers.dense(Placeholders.sentence, Placeholders.hidden2_neurons,activation=tf.nn.relu,kernel_initializer=he_init)
    ##Basic Neural Network
    #cluster_based_labels = hidden2
    #Cluster based objective
    cluster_based_labels = get_cluster_based_labels_revised(hidden2)
    logits = cluster_based_labels #tf.layers.dense(cluster_based_labels, Placeholders.n_classes)

    #cluster_based_labels = get_cluster_based_labels1(hidden2)
    #logits = tf.layers.dense(cluster_based_labels, Placeholders.n_classes)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels=Placeholders.class_labels)
    loss = tf.reduce_mean(cross_entropy)
    #loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(Placeholders.class_labels, logits)))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Placeholders.class_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    STEPS = 400000
    MINIBATCH_SIZE = 20

    print ("Starting at:" , datetime.datetime.now())
    sess.run(tf.global_variables_initializer())
    print ("Initialization done at:" , datetime.datetime.now())


    n_samples = 200
    n_test_samples = 100
    mu, sigma = 10, 10
    X = np.array([np.random.normal(mu, sigma, Placeholders.word_vec_dim) for i in range(n_samples)])
    Y = np.zeros(n_samples, dtype=np.int16)

    X_validation= np.array([np.random.normal(mu, sigma, Placeholders.word_vec_dim) for i in range(n_samples)])
    Y_validation = np.zeros(n_samples, dtype=np.int16)
    X_test = np.array([np.random.normal(mu, sigma, Placeholders.word_vec_dim) for i in range(n_test_samples)])
    Y_test = np.zeros(n_test_samples, dtype=np.int16)

    mu, sigma = 50, 10

    X = np.vstack((X, np.array([np.random.normal(mu, sigma, Placeholders.word_vec_dim) for i in range(n_samples)])))
    Y = np.vstack((Y,np.ones(n_samples,dtype=np.int16))).reshape(-1,n_samples *2 )

    X_test = np.vstack((X_test, np.array([np.random.normal(mu, sigma, Placeholders.word_vec_dim) for i in range(n_test_samples)])))
    Y_test = np.vstack((Y_test,np.ones(n_test_samples,dtype=np.int16))).reshape(-1,n_test_samples *2 )
    X_validation = np.vstack((X_validation, np.array([np.random.normal(mu, sigma, Placeholders.word_vec_dim) for i in range(n_samples)])))
    Y_validation = np.vstack((Y_validation,np.ones(n_samples,dtype=np.int16))).reshape(-1,n_samples *2 )
    #print(Y)
    Y = one_hot(Y)
    Y_test= one_hot(Y_test)
    Y_validation= one_hot(Y_validation)
    #print (X.shape,Y.shape)
    C = np.ones(Placeholders.hidden2_neurons)*10
    C = np.vstack((C,np.ones(Placeholders.hidden2_neurons)*50)).reshape(-1,Placeholders.hidden2_neurons)

    #sess.run(train_step, feed_dict={x: X, y_: Y, c:C})
    SUPPORT_SIZE = 40
    QUERY_SIZE = 40

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(STEPS):
        random_positive_examples, positive_labels = get_random_examples(X,Y,SUPPORT_SIZE + QUERY_SIZE,1)
        random_negative_examples, negative_labels = get_random_examples(X,Y,SUPPORT_SIZE + QUERY_SIZE,0)

        support_positive_examples = random_positive_examples[:SUPPORT_SIZE]
        support_negative_examples = random_negative_examples[:SUPPORT_SIZE]
        query_positive_examples = random_positive_examples[SUPPORT_SIZE:SUPPORT_SIZE + QUERY_SIZE].reshape(-1, Placeholders.word_vec_dim)
        query_negative_examples = random_negative_examples[SUPPORT_SIZE:SUPPORT_SIZE + QUERY_SIZE].reshape(-1, Placeholders.word_vec_dim)
        query_positive_labels = positive_labels[SUPPORT_SIZE:SUPPORT_SIZE + QUERY_SIZE].reshape(-1, Placeholders.n_classes)
        query_negative_labels = negative_labels[SUPPORT_SIZE:SUPPORT_SIZE + QUERY_SIZE].reshape(-1, Placeholders.n_classes)

        random_X = np.vstack((query_positive_examples,query_negative_examples))
        random_Y = np.vstack((query_positive_labels,query_negative_labels))

        c_zero = get_centroid(support_positive_examples,hidden2,sess)
        c_one = get_centroid(support_negative_examples,hidden2,sess)
        C = np.vstack((c_zero,c_one))

        sess.run(train_step, feed_dict={x: random_X, y_: random_Y, c:C})

        if(i%10000 == 0):
            #logits_ = np.array(sess.run(cluster_based_labels, feed_dict={x: X_validation, y_: Y_validation, c:C}))
            #print(logits_[:5,:])
            training_accuracy = sess.run(accuracy, feed_dict={x: random_X, y_: random_Y, c:C})*100
            training_loss = sess.run(loss, feed_dict={x: random_X, y_: random_Y, c:C})
            #print("Training Accuracy: {0:.2f}".format(training_accuracy), " Training Loss: {0:.2f}".format(training_loss))
            validation_accuracy = sess.run(accuracy, feed_dict={x: X_validation, y_: Y_validation, c:C})*100
            test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_: Y_test, c:C})*100
            print(" Training Loss: {0:.2f}".format(training_loss), "Training Accuracy: {0:.2f}".format(training_accuracy),"Validation Accuracy: {0:.2f}".format(validation_accuracy), " Test Accuracy: {0:.2f}".format(test_accuracy))


with tf.Session() as sess:
    train(sess)