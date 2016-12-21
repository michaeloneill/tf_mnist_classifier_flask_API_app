# tf_mnist_classifier_flask_API_app


###  Requirement ###

- Python 2.7.11


### Setup ###

    $ pip install -r requirements.txt
    $ python load_mnist.py

The second command downloads and unpacks the mnist dataset (from  http://yann.lecun.com/exdb/mnist/) into a newly created directory data/

### Model training ###

Note that the classifier model (a CNN built in tensorflow and including dropout and batch-normalisation) HAS ALREADY BEEN TRAINED to >99% accuracy and saved into training_results/ as a set of files prefixed by 'best_model-4699'. These files allow for reloading and evaluating the model on new test images (see below). training_results/logs/ contains the training log files which can be viewed in tensorboard.

Alternatively, to train the model from scratch:

     $ cd classifier
     $ python train_classifier.py
     
This will overwrite the saved model and log files mentioned above.


### Running the app locally ###
    
    $ python
    $ from app import classify
    $ classify('/path/to/image.png')  # returns corresponding class label prediction

where '/path/to/image.png' is the filepath to any 28 x 28 pixel png image containing data of type uint8. For convenience, data/png_test_images/ contains 100 randomly chosen mnist digits from the test set that can be used for this task (this assumes Setup above has been completed).

### Running the app via the Flask API ###

    $ export FLASK_APP=app.py
    $ flask run
    $ curl -X POST -F file=@/path/to/image.png http://localhost:5000/mnist/classify/

This outputs a JSON containing the class label prediction, or a 404.





