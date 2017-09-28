"""
Flask Serving
This file is a sample flask app that can be used to test your model with an REST API.
This app does the following:
    - Look for a Zvector(n_samples is encoded in this file) parameter
    - Returns the output file generated at /output
Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: ckp
    - It is loaded from /model

GET req:
    paramrter:
        - ckp, optional, load a specific chekcpoint from /model
    no parameter:
        - generate 1 image from random noise

POST req:
    parameter:
        - file, required, a serialized Zvector file(the number of images to return is encoded in this vector)
        - ckp, optional, load a specific chekcpoint from /model

"""
import os
import torch
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from dcgan import DCGAN

ALLOWED_EXTENSIONS = set(['jpg, png'])

MODEL_PATH = '/input'
print('Loading model from path: %s' % MODEL_PATH)

app = Flask('MNIST-Classifier')

#  2 possible parameters - checkpoint, zinput(file.cpth)
# Return an Image
@app.route('/<path:path>', methods=['POST'])
def geneator_handler(path):
    zvector = None
    batchSize = 1
    # Upload a serialized Zvector
    if request.method == 'POST':
        # DO things
        # check if the post request has the file part
        if 'file' not in request.files:
            return BadRequest("File not present in request")
        file = request.files['file']
        if file.filename == '':
            return BadRequest("File name is not present in request")
        if not allowed_file(file.filename):
            return BadRequest("Invalid file type")
        filename = secure_filename(file.filename)
        input_filepath = os.path.join('/output', filename)
        file.save(input_filepath)
        # Load a Z vector and Retrieve the N of samples to generate
        zvector = torch.load(input_filepath)
        batchSize = zvector.size()[0]

    checkpoint = request.form.get("ckp") or "netG_epoch_99.pth"
    # Check for cuda availability
    if torch.cuda.is_available():
        # GPU and cuda
        Generator = DCGAN(netG=os.path.join(MODEL_PATH, checkpoint), zvector=zvector, batchSize=batchSize, ngpu=1, cuda=True)
    else:
        # CPU
        Generator = DCGAN(netG=os.path.join(MODEL_PATH, checkpoint), zvector=zvector, batchSize=batchSize, ngpu=0)
    Generator.build_model()
    Generator.generate()
    return send_file(OUTPUT_PATH, mimetype='image/png')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')
