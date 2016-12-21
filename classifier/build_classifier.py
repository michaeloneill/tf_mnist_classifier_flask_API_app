import tensorflow as tf
import numpy as np
from layers import fully_connected_layer, conv_pool_layer

ACTIVATIONS = {
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'softmax': tf.nn.softmax,
    'identity': tf.identity
}


def build_cnn_classifier(params):
    
    with tf.name_scope('input'):
        
        x = tf.placeholder(tf.float32,
                           shape=params['inpt_shape']['x'],
                           name='x')
        tf.add_to_collection('x', x)

        y_ = tf.placeholder(tf.float32,
                            shape=params['inpt_shape']['y_'],
                            name='y_')
        tf.add_to_collection('y_', y_)

    with tf.name_scope('dropout_keep_prob'):
        dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')
        tf.add_to_collection('dropout_keep_prob', dropout_keep_prob)

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.float32,
                                     shape=(),
                                     name='is_training')
        tf.add_to_collection('is_training', is_training)

    with tf.variable_scope('cnn'):
        output = build_cnn_inference_graph(x, dropout_keep_prob,
                                           is_training, params['cnn'])
        tf.add_to_collection('output', output)
            
    with tf.name_scope('loss'):
        loss = get_loss(output, y_)
    tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    model = {
        'x': x,
        'y_': y_,
        'dropout_keep_prob': dropout_keep_prob,
        'is_training': is_training,
        'loss': loss,
        'train': build_train_graph(loss, params['train']),
        'logits': output,
        'accuracy': accuracy
    }
    
    return model


def build_cnn_inference_graph(x, dropout_keep_prob, is_training, params):

    nConvLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nConvLayers):

        kernel_size = params['kernel_size'][i]
        num_outputs = params['num_outputs'][i]
        layer_name = 'conv_layer'+str(i+1)
        act = ACTIVATIONS[params['activations'][i]]
        pool = params['pool'][i]
        
        if i == 0:
            inpt = x
        else:
            inpt = prev_layer

        layer = conv_pool_layer(
            inputs=inpt,
            num_outputs=num_outputs,
            kernel_size=kernel_size,
            stride=1,
            padding='SAME',
            layer_name=layer_name,
            is_training=is_training,
            activation_fn=act,
            pool=pool
        )
        prev_layer = layer

    # Flatten output of last conv_layer and pass to fc layers
    flattened_dim = np.prod(prev_layer.get_shape().as_list()[1:])
    flattened = tf.reshape(prev_layer, [-1, flattened_dim])

    output = build_mlp_inference_graph(flattened, dropout_keep_prob,
                                       is_training, params['fc_params'])
    return output


def build_mlp_inference_graph(x, dropout_keep_prob, is_training, params):
    
    nLayers = len(params['num_outputs'])
    prev_layer = None

    for i in range(nLayers):
            
        num_outputs = params['num_outputs'][i]
        layer_name = 'fc_layer'+str(i+1)
        act = ACTIVATIONS[params['activations'][i]]
        
        if i == 0:
            inpt = x
        else:
            inpt = prev_layer

        if params['dropout'][i]:
            with tf.name_scope('dropout'):
                inpt = tf.nn.dropout(inpt, dropout_keep_prob)

        layer = fully_connected_layer(
            inputs=inpt,
            num_outputs=num_outputs,
            layer_name=layer_name,
            is_training=is_training,
            activation_fn=act
        )
        prev_layer = layer

    return prev_layer


def build_train_graph(loss, params):

    with tf.name_scope('train'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), params['grad_clip'])

        optimizer = tf.train.MomentumOptimizer(
            params['learning_rate'], momentum=params['momentum'])
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        tf.add_to_collection('train_op', train_op)

        return train_op


def get_loss(logits, targets):
    return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, targets))


