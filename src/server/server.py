from flask import Flask
from flask_cors import CORS
import os
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from src.config import Config
from src.datasets.diskds.memory_file_storage import MemoryFileDiskStorage
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.datasets.dataset_provider import DatasetProvider
from src.datasets.diskds.ceclustering_model_loader import CEClusteringModelLoader
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
from src.models.working_model_loader import *
from torch.utils.data import DataLoader
from src.server.dto import *
import json
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
CORS(app)

searchify_config = Config()
searchify_config.run_device = torch.device("cpu")

spectrogram_generator = SpectrogramGenerator(searchify_config)

run_params = RunParameters("disk-ds(/media/andrius/FastBoi/bakis_data/top10000_meta_22k)")
run_params.apply_overrides(
    {
            R.CLUSTERING_MODEL: 'mass',
            R.DISKDS_NUM_FILES: '5000',
            R.DISKDS_WINDOW_LENGTH: str((2**17)),
            R.DISKDS_WINDOW_HOP_TRAIN: str((2**14)),
        }
)
model, file_list = load_working_model(run_params, "zoo/5000massv1", True)
model_type = run_params.getd(R.CLUSTERING_MODEL, "cec")
if model_type == "cec":
    cluster_positions = model.classification[-1].centroids.clone().detach().to("cpu")
    cluster_sizes = model.classification[-1].cluster_sizes.clone().detach().to("cpu")
    cec_max_dist = model.classification[-1].max_dist
elif model_type == "mass":
    cluster_positions = model.classification[-1].centroids.clone().detach().to("cpu")
    cluster_mass = model.classification[-1].cluster_mass.clone().detach().to("cpu")
    g_constant = model.classification[-1].g_constant.clone().detach().to("cpu")
    mass_max_dist = model.classification[-1].max_dist
model.save_distance_output = True
model.train(False)
model.to(searchify_config.run_device)
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
        query_topn = 5
        log.info(f"Received request with the following params: {request.args}")
        if "topn" in request.args:
            query_topn = min(15, int(request.args["topn"]))
            query_topn = max(1, query_topn)
            log.info(f"Converted `{request.args['topn']}` to {query_topn}")
        query_fully_connected_graph = False
        if "full_graph" in request.args:
            if request.args["full_graph"] == "true":
                query_fully_connected_graph = True
        query_graph_connectivity = 3
        if "graph_connectivity" in request.args:
            query_graph_connectivity = min(10, int(request.args["graph_connectivity"]))
            query_graph_connectivity = max(1, query_graph_connectivity)
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
            log.info(f"Bytes transfered to memory file: {bytes_written}")
            memory_file.seek(0)
            file.save(memory_file)
            topk_classes_to_evaluate = torch.Tensor()
            sample_output_centroid_positions = torch.Tensor()
            total_samples = 0
            topk = query_topn
            with MemoryFileDiskStorage(memory_file, format=file_extension[1:], run_params=run_params, features=["data"]) as memory_storage:
                loader = DataLoader(memory_storage, shuffle=False, batch_size=5, num_workers=0)
                top1_answers = torch.tensor([])
                for item_data in loader:
                    with torch.no_grad():
                        samples = item_data["samples"]
                        samples = samples.to(searchify_config.run_device)
                        spectrogram = spectrogram_generator.generate_spectrogram(
                            samples, narrow_to=128,
                            timestretch=False, random_highpass=False,
                            random_bandcut=False, normalize_mag=True)
                        outputs = model(spectrogram)
                        top1_answers = torch.cat([top1_answers, torch.argmax(outputs, dim=-1).view(-1)])
                        # log.info(f"Outputs: {torch.argmax(outputs, dim=-1)}")
                        _, topk_indices = outputs.topk(topk)
                        topk_indices = topk_indices.unique()
                        topk_classes_to_evaluate = torch.cat([topk_classes_to_evaluate, topk_indices]).unique()

                        batch_sample_distances = model.distance_output.to("cpu")
                        batch_sample_distances = torch.sigmoid(batch_sample_distances)
                        sample_output_centroid_positions = torch.cat([sample_output_centroid_positions, batch_sample_distances])
                model_top1_idx = int(torch.mode(top1_answers, 0).values)
                log.info(f"Top1 model answer mode: {torch.mode(top1_answers, 0)}")
                log.info(f"Top1 model answer: {os.path.basename(file_list[model_top1_idx])}")
                log.info(f"Topk class indexes to evaluate in search: {topk_classes_to_evaluate.shape}")
                log.info(f"Output centroid positions shape: {sample_output_centroid_positions.shape}")
                log.info(f"")
                sample_distances_to_clusters = {}
                if model_type == "mass":
                    average_cluster_mass = 0
                    for track_idx in topk_classes_to_evaluate.numpy():
                        average_cluster_mass += float(cluster_mass[int(track_idx)].item())
                    average_cluster_mass = average_cluster_mass / len(topk_classes_to_evaluate)
                for track_idx in topk_classes_to_evaluate.numpy():
                    track_idx = int(track_idx)
                    track_norm_space_position = cluster_positions[track_idx]

                    track_cluster_position_repeated = track_norm_space_position.repeat(sample_output_centroid_positions.shape[-2], 1)
                    # log.info(f"Repeated track cluster position: {track_cluster_position_repeated.shape} should be: {sample_output_centroid_positions.shape}")
                    # log.info(f"Repeated track cluster position: \n{track_cluster_position_repeated}")
                    # log.info(f"Sample output position: \n{sample_output_centroid_positions}")
                    abs_output_distances = track_cluster_position_repeated - sample_output_centroid_positions
                    # log.info(f"Absolute output distances: {abs_output_distances}")
                    output_distances = torch.norm(abs_output_distances, p=2, dim=-1)
                    # log.info(f"Normed output distances: {output_distances}")
                    if model_type == "cec":
                        track_cluster_size = float(cluster_sizes[track_idx])
                        output_distances = torch.div(output_distances, cec_max_dist)
                        output_distances = torch.mul(output_distances, torch.div(1, track_cluster_size))
                        output_distances = torch.pow(output_distances, 2)
                        mean_output_distance_for_track = torch.mean(output_distances)
                        sample_distances_to_clusters[track_idx] = mean_output_distance_for_track.item()
                    elif model_type == "mass":
                        track_cluster_size = float(cluster_mass[track_idx])
                        output_distances = torch.div(output_distances, mass_max_dist)
                        cluster_forces = (g_constant * track_cluster_size  * average_cluster_mass)/ torch.square(output_distances)
                        mean_force_for_track = torch.mean(cluster_forces)
                        # log.info(f"Mean stats for samples to track {track_idx}:\tr={torch.mean(output_distances)} f={mean_force_for_track.item()} r_min={torch.min(output_distances).item()} f_min={torch.min(cluster_forces).item()} r_max={torch.max(output_distances).item()} f_max={torch.max(cluster_forces).item()}")
                        sample_distances_to_clusters[track_idx] = -mean_force_for_track.item()
                    # output_distances = torch.mul(output_distances, torch.div(1, track_cluster_size))
                    # log.info(f"Output distances: {output_distances.shape} -- \n {output_distances}")
                sorted_sample_avg_distances_to_clusters = dict(sorted(sample_distances_to_clusters.items(), key=lambda item: item[1], reverse=False))
                graph_nodes = [GraphNode(-1, "Provided sample audio", float(0))]
                graph_links = []
                log.info("Found the following recommendataions:")
                for track_idx in sorted_sample_avg_distances_to_clusters:
                    track_name = os.path.basename(file_list[track_idx])
                    track_name, _ = os.path.splitext(track_name)
                    if model_type == "cec":
                        graph_nodes.append(GraphNode(track_idx, track_name, float(cluster_sizes[track_idx].item())))
                        log.info(f"\t{track_name}: {sorted_sample_avg_distances_to_clusters[track_idx]} cluster_size={cluster_sizes[track_idx].item()}")
                    elif model_type == "mass":
                        graph_nodes.append(GraphNode(track_idx, track_name, float(cluster_mass[track_idx].item())))
                        log.info(f"\t{track_name}: {sorted_sample_avg_distances_to_clusters[track_idx]} cluster_mass={cluster_mass[track_idx].item()}")
                    graph_links.append(GraphLink(-1, track_idx, float(sorted_sample_avg_distances_to_clusters[track_idx])))
                
                topn_links = []
                for source_idx in sorted_sample_avg_distances_to_clusters:
                    for target_idx in sorted_sample_avg_distances_to_clusters:
                        if source_idx == target_idx:
                            continue
                        indicies = [source_idx, target_idx]
                        def f(l: GraphLink):
                            return l.source_track_index in indicies and l.target_track_index in indicies
                        if len(list(filter(f, graph_links))) == 0: # if there is no link yet between the two nodes
                            if model_type == "cec":
                                source_pos = cluster_positions[source_idx]
                                target_pos = cluster_positions[target_idx]
                                source_size = cluster_sizes[source_idx]
                                target_size = cluster_sizes[target_idx]
                                abs_output_distance = source_pos - target_pos
                                output_distances = torch.norm(abs_output_distance, p=2, dim=-1)
                                link_distance = torch.div(output_distances, cec_max_dist)
                                link_distance = torch.pow(link_distance, 2)
                                # log.info(f"Link distance between {source_idx} and {target_idx}: {link_distance}")
                                topn_links.append(GraphLink(int(source_idx), int(target_idx), float(link_distance.item())))
                            elif model_type == "mass":
                                source_pos = cluster_positions[source_idx]
                                target_pos = cluster_positions[target_idx]
                                source_mass = cluster_mass[source_idx]
                                target_mass = cluster_mass[target_idx]
                                abs_output_distance = source_pos - target_pos
                                output_distances = torch.norm(abs_output_distance, p=2, dim=-1)
                                link_distance = torch.div(output_distances, mass_max_dist)
                                cluster_force = (g_constant * source_mass * target_mass) / torch.square(link_distance)
                                # log.info(f"Mean stats for {source_idx} -> {target_idx}:\tr={link_distance.item()} f={cluster_force.item()}")
                                # logging.info(f"Cluster force: {cluster_force}")
                                topn_links.append(GraphLink(int(source_idx), int(target_idx), float(-cluster_force.item())))
                # add links between other top-n tracks
                if(query_fully_connected_graph):
                    graph_links.extend(topn_links)
                else:
                    # fully connected graphs are not easy to visualize, so here we make it so
                    # that a node has at max N connections:
                    N = query_graph_connectivity
                    def f(l: GraphLink):
                        return l.distance
                    topn_links = sorted(topn_links, key=f)
                    for topn_link in topn_links:
                        link_indicies = [topn_link.source_track_index, topn_link.target_track_index]
                        def f(l: GraphLink):
                            return l.source_track_index in link_indicies or l.target_track_index in link_indicies
                        current_node_links = list(filter(f, graph_links))
                        if len(current_node_links) < N*2: # twice, because we are counting links for both target and source
                            graph_links.append(topn_link)
                if model_type == "mass":
                    link_distances = []
                    for link in graph_links:
                        link_distances.append(link.distance)
                    np_link_distances = np.array(link_distances)
                    min_distance = np.min(np_link_distances)
                    for link in graph_links:
                        link.distance = link.distance - min_distance
                final_graph = GraphData(graph_nodes, graph_links, total_samples)
                return final_graph.json()

            return f"File uploaded!"
        else:
            return f"File format not allowed: {os.path.splitext(file.filename)}"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type="hidden" id="renderHtml" name="renderHtml" value="True">
      <input type=submit value=Upload>
    </form>
    '''