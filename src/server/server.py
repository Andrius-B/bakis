from flask import Flask
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from src.config import Config
from src.datasets.diskds.memory_file_storage import MemoryFileDiskStorage
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.datasets.dataset_provider import DatasetProvider
from src.datasets.diskds.ceclustering_model_loader import CEClusteringModelLoader
from torch.utils.data import DataLoader
import io
import torch
import torchaudio
import logging

UPLOAD_FOLDER = './data/server_downloads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac'}

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

searchify_config = Config()
searchify_config.run_device = torch.device("cpu")

spectrogram_t = torchaudio.transforms.Spectrogram(
    n_fft=2048, win_length=2048, hop_length=1024, power=None
).to(searchify_config.run_device) # generates a complex spectrogram
time_stretch_t = torchaudio.transforms.TimeStretch(hop_length=1024, n_freq=1025).to(searchify_config.run_device)
low_pass_t = torchaudio.functional.lowpass_biquad
high_pass_t = torchaudio.functional.highpass_biquad
mel_t = torchaudio.transforms.MelScale(n_mels=64, sample_rate=44100).to(searchify_config.run_device)
norm_t = torchaudio.transforms.ComplexNorm(power=2).to(searchify_config.run_device)
ampToDb_t = torchaudio.transforms.AmplitudeToDB().to(searchify_config.run_device)

run_params = RunParameters("disk-ds(/home/andrius/git/bakis/data/spotifyTop10000)")
run_params.apply_overrides(
    {
            R.DATASET_NAME: 'disk-ds(/home/andrius/git/bakis/data/spotifyTop10000)',
            # R.DATASET_NAME: 'disk-ds(/home/andrius/git/searchify/resampled_music)',
            R.DISKDS_NUM_FILES: '5000',
        }
)
ce_clustering_loader = CEClusteringModelLoader()
net_save_path = "temp.pth"
cec_save_path = "temp.csv"
if(os.path.isfile(net_save_path)):
    log.info(f"Using saved model from: {net_save_path}")
    model = torch.load(net_save_path)
# this loader generation consumes quite a bit of memory because it regenerates
# all the idx's for the train set, maybe this would make sense to make DiskDsProvider
# stateful and able to list all files (by calling a static method on disk_storage..)
train_l, train_bs, valid_l, valid_bs = DatasetProvider().get_datasets(run_params)
# here the train dataset has the file list we are interested in..
file_list = train_l.dataset.get_file_list()
ceclustring = model.classification[-1]
ceclustring = ce_clustering_loader.load(ceclustring, cec_save_path, file_list)
model.classification[-1] = ceclustring
log.info("Loading complete, server ready")

def allowed_file(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension[1:] in ALLOWED_EXTENSIONS


@app.route('/searchify', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            final_file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            log.info(f"File received: {file}")
            memory_file = io.BytesIO()
            _, file_extension = os.path.splitext(file.filename)
            writer = io.BufferedWriter(memory_file)
            file.save(writer)
            writer.flush()
            bytes_written = writer.tell()
            memory_file.seek(0)
            reader = io.BufferedReader(memory_file)
            log.info(f"Bytes transfered to memory file: {bytes_written}")
            data = reader.read(10)
            log.info(f"Test read of first 10 bytes: {data}")
            memory_file.seek(0)
            file.save(memory_file)
            with MemoryFileDiskStorage(memory_file, format=file_extension[1:], features=["data"]) as memory_storage:
                # do the processing...
                loader = DataLoader(memory_storage, shuffle=False, batch_size=256, num_workers=0)
                for item_data in loader:
                    log.info(f"Batch from loader: {item_data}")
                # output = model(samples)
                # log.info(f"Model output: {output}")

            return f"File uploaded!"
        else:
            return f"File format not allowed: {os.path.splitext(file.filename)}"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''