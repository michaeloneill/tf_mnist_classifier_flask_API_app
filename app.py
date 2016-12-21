import os
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename

import tensorflow as tf
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT, 'server_uploads/')
ALLOWED_EXTENSIONS = set(['png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def classify(image_filename):

    img = Image.open(image_filename)
    img = np.array(img)

    if img.dtype != 'uint8' or img.shape != (28, 28):
        raise ValueError('image must 28x28 uint8')

    img = img.astype(np.float32)/255
    img = np.reshape(img, (1, 28, 28, 1))
    
    with tf.Session() as sess:

        model_path_prefix = os.path.join(
            ROOT, 'classifier/training_results/best_model-4699')
        saver = tf.train.import_meta_graph(model_path_prefix+'.meta')
        saver.restore(sess, model_path_prefix)

        x = tf.get_collection('x')[0]
        dropout_keep_prob = tf.get_collection('dropout_keep_prob')[0]
        is_training = tf.get_collection('is_training')[0]
        logits = tf.get_collection('output')[0]
        predictions = tf.squeeze(tf.argmax(logits, 1))

        preds = sess.run([predictions],
                         feed_dict={
                             x: img,
                             dropout_keep_prob: 1.0,
                             is_training: 0.0
                         }
        )
        return preds[0]


@app.route('/mnist/classify/', methods=['POST'])
def upload_file():

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # check if the post request has the file part
    if 'file' not in request.files:
        print('No file part')
        return abort(404)
    file = request.files['file']

    if file and allowed_file(file.filename):
        print('classifying {}'.format(file.filename))
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = classify(filepath)
        except Exception as e:
            print type(e)
            print e

        os.remove(filepath)  # free up space on server
        return jsonify(result=result)
    
    else:
        print('Invalid file')
        return abort(404)

