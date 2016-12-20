import tensorflow as tf
import os
import cPickle
from training_fns import train
from build_classifier import build_cnn_classifier


def main():

    results_dir = './training_results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    params_cnn = {
        'kernel_size': [[5, 5], [5, 5]],
        'num_outputs': [6, 12],
        'activations': ['relu', 'relu'],
        'pool': [True, True],
        'fc_params': {'num_outputs': [10],
                      'activations': ['identity'],
                      'dropout': [True]}
    }
    params_train = {
        'miniBatchSize': 100,
        'epochs': 10,
        'learning_rate': 0.01,
        'dropout_keep_prob': 0.5,
        'monitor_frequency': 100,
        'momentum': 0.9,
        'grad_clip': 5
    }

    params = {
        'cnn': params_cnn,
        'train': params_train,
        'inpt_shape': {'x': [None, 28, 28, 1], 'y_': [None, 10]},
        'device': '/cpu:0',
        'results_dir': results_dir
    }

    model = build_cnn_classifier(params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    with open('../data/mnist.pkl', 'rb') as f:
        train_set, val_set, test_set = cPickle.load(f)

    train(train_set, val_set,
          test_set, params['train'],
          model, sess, results_dir)

if __name__ == '__main__':
    main()
