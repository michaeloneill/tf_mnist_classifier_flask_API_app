import os
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename

import tensorflow as tf
import numpy as np
from PIL import Image

UPLOAD_FOLDER = './server_uploads/'
ALLOWED_EXTENSIONS = set(['png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def classify(image_filename):

    img = Image.open(image_filename)
    img = np.array(img)

    assert img.dtype == 'uint8', \
        'image data must be uint8'
    assert img.shape == (28, 28), \
        'image shape must be 28x28'

    img = img.astype(np.float32)/255
    img = np.reshape(img, (1, 28, 28, 1))
    
    with tf.Session() as sess:
        
        saver = tf.train.import_meta_graph(
            './classifier/training_results/best_model-4699.meta')
        saver.restore(sess, './classifier/training_results/best_model-4699')

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
        print(file.filename)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = classify(filepath)

        if result:
            return jsonify(result=result)
        else:
            return abort(404)
