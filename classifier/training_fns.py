import tensorflow as tf


def run_train(train_set, params, model, sess, mini_batch_index, merged):
    
    return sess.run(
        [model['train'], model['loss'], merged],
        feed_dict={
            model['x']:
            train_set[0][mini_batch_index*params['mini_batch_size']:
                         (mini_batch_index+1)*params['mini_batch_size']],
            model['y_']:
            train_set[1][mini_batch_index*params['mini_batch_size']:
                         (mini_batch_index+1)*params['mini_batch_size']],
            model['dropout_keep_prob']: params['dropout_keep_prob'],
            model['is_training']: 1.0
        }
    )


def run_val(val_set, params, model, sess, merged):

    return sess.run(
        [model['accuracy'], merged],
        feed_dict={
            model['x']: val_set[0],
            model['y_']: val_set[1],
            model['dropout_keep_prob']: 1.0,
            model['is_training']: 0.0
        }
    )


def run_test(test_set, params, model, sess):

    return sess.run(
        model['accuracy'],
        feed_dict={
            model['x']: test_set[0],
            model['y_']: test_set[1],
            model['dropout_keep_prob']: 1.0,
            model['is_training']: 0.0
        }
    )


def train(train_set, val_set, test_set, params, model, sess, results_dir):
    
    saver = tf.train.Saver(max_to_keep=1)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(results_dir+'logs/train', sess.graph)
    val_writer = tf.summary.FileWriter(results_dir+'logs/val', sess.graph)
    
    n_batch_train = train_set[0].shape[0]/params['mini_batch_size']
    best_accuracy_val = 0.0
    best_model_file = None
    best_iter = None
    
    print 'starting training...'
    for epoch in range(params['epochs']):
        for mini_batch_index in range(n_batch_train):
            iteration = epoch*n_batch_train + mini_batch_index

            _, loss_train, summary = run_train(train_set, params, model,
                                               sess, mini_batch_index, merged)
            train_writer.add_summary(summary, iteration)
            
            if (iteration+1) % params['monitor_frequency'] == 0:
                print 'training loss for minibatch {0}/{1} '\
                    'epoch {2} is: {3:.2f}'.format(
                        mini_batch_index+1, n_batch_train, epoch+1, loss_train)
                
                accuracy_val, summary = run_val(
                    val_set, params, model, sess, merged)
                val_writer.add_summary(summary, iteration)
                print 'val accuracy is: {:.2f}'.format(accuracy_val)

                if accuracy_val > best_accuracy_val:
                    best_accuracy_val = accuracy_val
                    best_iter = iteration
                    best_model_file = saver.save(
                        sess, results_dir+'best_model', global_step=best_iter)

    saver.restore(sess, best_model_file)  # sess modified
    accuracy_test = run_test(val_set, params, model, sess)
    print 'Training complete. Test set accuracy at '\
        'highest validation accuracy is: {:.2f}'.format(accuracy_test)
    


