"""
Flask Serving
This file is a sample flask app that can be used to test your model with an REST API.
This app does the following:
    - Look for a Image and then process it to be MNIST compliant
    - Returns the evaluation
Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: ckp
    - It is loaded from /model

POST req:
    parameter:
        - file, required, a handwritten digit in [0-9] range
        - ckp, optional, load a specific chekcpoint from /model

"""
import os
import torch
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from ConvNet import ConvNet

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

MODEL_PATH = '/input'
print('Loading model from path: %s' % MODEL_PATH)

EVAL_PATH = '/eval'
# Is there the EVAL_PATH?
try:
    os.makedirs(EVAL_PATH)
except OSError:
    pass

app = Flask('MNIST-Classifier')


# Return an Image
@app.route('/<path:path>', methods=['POST'])
def geneator_handler(path):
    """Upload an handwrittend digit image in range [0-9], then
    preprocess and classify"""
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    filename = secure_filename(file.filename)
    image_folder = os.path.join(EVAL_PATH, "images")
    # Create dir /eval/images
    try:
        os.makedirs(image_folder)
    except OSError:
        pass
    # Save Image to process
    input_filepath = os.path.join(image_folder, filename)
    file.save(input_filepath)
    # Get ckp
    checkpoint = request.form.get("ckp") or "mnist_convnet_model_epoch_10.pth" # FIX to

    # Preprocess, Build and Evaluate
    Model = ConvNet(ckp=checkpoint)
    Model.image_preprocessing()
    Model.build_model()
    pred = Model.classify()

    # Return classification and remove uploaded file
    # output = "Images: " + file.filename + ", Classified as: " + pred.data.max(1, keepdim=True)[1]
    # TODO: label = numpy.asscalar(output.data.max(1, keepdim=True)[1])
    output = "Images: {file}, Classified as {pred}".format(file=file.filename,
        pred=pred.data.max(1, keepdim=True)[1])
    os.remove(input_filepath)
    return output


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')
