from src.runners.abstract_runner import AbstractRunner
from src.experiments.base_experiment import BaseExperiment 
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.models.ceclustering import CEClustering
from src.models.working_model_loader import load_working_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from tqdm import tqdm
import os
from src.util.mutagen_utils import read_mp3_metadata
import math
import torch
import matplotlib.cm
import pandas as pd
import plotly.graph_objects as go
import logging

cmap = matplotlib.cm.get_cmap('plasma')
log = logging.getLogger(__name__)

class AudioAnalyzeExperiment(BaseExperiment):

    def show_distance_from_zero_hist(self, clustering_module: CEClustering):
        clustering_module = clustering_module.cpu()
        centroids = clustering_module.centroids.detach().numpy()
        centroid_sizes = clustering_module.cluster_sizes.detach().numpy()
        magnitudes = []
        for i, centroid in enumerate(centroids):
            cluster_size = centroid_sizes[i]
            centroid_len = np.linalg.norm(centroid) / math.sqrt(len(centroid))
            magnitudes.append(centroid_len)
        max_len = max(magnitudes)
        max_item = magnitudes.index(max_len)
        print(f"Mean of centroids: {np.mean(centroids, axis=0)}")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(magnitudes, 20)
        ax1.set_title('Centroid magnitudes')
        ax2.hist(centroid_sizes, 50)
        ax2.set_title('Centroid sizes')
        print(f"Max centroid distance from (0): {max_len}")
        plt.show()
        # print(f"Centroid s: {cluster_size} of {filepath} at: {centroid}")
    
    def run_centroids_tsne(self, tsne_data, text_resolver = None, color_resolver = None, playlist_resolver = None):
        tsne_fn = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=10000, learning_rate=200)
        tsne_results = tsne_fn.fit_transform(tsne_data)
        centroids = tsne_results
        print(f"Transformed centroids: {centroids.shape}")
        centroids_list = []
        for centroid_idx, centroid_pos in tqdm(enumerate(centroids), total=len(centroids)):

            if color_resolver is None:
                color = cmap(centroid_idx/(len(centroids)))
            else:
                color = color_resolver(centroid_idx)

            if text_resolver is not None:
                classText = text_resolver(centroid_idx)
            else:
                classText = str(centroid_idx)
            playlist = "None"
            if playlist_resolver is not None:
                playlist =  playlist_resolver(centroid_idx)
            centroids_list.append({
                'xpos': centroid_pos[0],
                'ypos': centroid_pos[1],
                'color': color,
                'text': classText,
                'centroid_idx': centroid_idx,
                'playlist': playlist,
            })
        centroids_by_playlist = {}
        for c in centroids_list:
            p = c['playlist']
            if p not in centroids_by_playlist:
                centroids_by_playlist[p] = [c]
            else:
                centroids_by_playlist[p].append(c)
        log.info(f"Formed centroid lists: {len(centroids_by_playlist)}")
        for p_name in centroids_by_playlist:
            log.info(f"From: {p_name} - {len(centroids_by_playlist[p_name])} centroids")
        fig = go.Figure()
        for playlist_name in centroids_by_playlist:
            playlist_centroids = centroids_by_playlist[playlist_name]
            centroids_tsne = {k: [dic[k] for dic in playlist_centroids] for k in playlist_centroids[0]}
            log.info(f"Adding centroids for trace {playlist_name} ({len(centroids_tsne['xpos'])} centroids)")
            fig.add_trace(
                go.Scattergl(
                    x=centroids_tsne['xpos'],
                    y=centroids_tsne['ypos'],
                    # mode='markers+text',
                    mode='markers',
                    marker=dict(
                        color=centroids_tsne['color'],
                        line_width=0.5,
                        symbol="circle",
                        opacity=0.7,
                        size=10,
                    ),
                    text=centroids_tsne['text'],
                    name=playlist_name
                ),
            )
        fig.update_layout(legend_title_text='Spotify playlist')
        # fig.show()
        fig.write_html('t-sne.html', auto_open=True)

    def load_spotify_playlists(self):
        f_names = {
            "Dance Party":"test/playlists/danceparty.csv",
            "Rap Caviar": "test/playlists/rapcaviar.csv",
            "Rock Classics": "test/playlists/rockclassics.csv",
            "Shoegaze": "test/playlists/shoegaze.csv",
            "Soft Hits": "test/playlists/softhits.csv",
        }
        playlists = {}
        for name in f_names:
            playlists[name] = pd.read_csv(f_names[name])["spotify_uri"].tolist()
        return playlists

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/final22k/train)',
            R.CLUSTERING_MODEL: 'mass',
            R.DISKDS_NUM_FILES: '9500'
        }

    def run(self):
        run_params = super().get_run_params()
        method = 'T-SNE'
        playlists = self.load_spotify_playlists()
        net, files = load_working_model(run_params, 'zoo/9500massv2', reload_classes_from_dataset=False)
        clustering_module = net.classification[-1].cpu()
        def playlist_resolver(idx):
            filename = files[idx]
            try:
                metadata = read_mp3_metadata(filename)
                spotify_uri = metadata["spotify_uri"]
                for i, playlist_name in enumerate(playlists):
                    if spotify_uri in playlists[playlist_name]:
                        return playlist_name
            except Exception as e:
                log.error(f"Failed loading metadata for song: {filename}")
                log.exception(e)
                return "None"
            return "None"

        def color_resolver(idx):
            filename = files[idx]
            try:
                metadata = read_mp3_metadata(filename)
                spotify_uri = metadata["spotify_uri"]
                for i, playlist_name in enumerate(playlists):
                    if spotify_uri in playlists[playlist_name]:
                        # log.info(f"Song {metadata['artist']}-{metadata['title']} ({metadata['spotify_uri']}) belongs to {playlist_name}")
                        color = cmap((i+1)/len(playlists))
                        return f"rgb({color[0]},{color[1]},{color[2]})"
                # log.info(f"Song {metadata['artist']}-{metadata['title']} ({metadata['spotify_uri']}) does not belong to any playlist")
            except Exception as e:
                log.error(f"Failed loading metadata for song: {filename}")
                log.exception(e)
                color = cmap(0)
                return f"rgb({color[0]},{color[1]},{color[2]})"
            color = cmap(0)
            return f"rgb({color[0]},{color[1]},{color[2]})"

        def text_resolver(idx):
            filename = os.path.basename(files[idx])
            name, _ = os.path.splitext(filename)
            return name
        num_classes = int(run_params.get(R.DISKDS_NUM_FILES))
        tsne_data = clustering_module.centroids.detach().numpy()[:num_classes, :]
        self.run_centroids_tsne(tsne_data, text_resolver, color_resolver, playlist_resolver)
        

    def help_str(self):
        return """This experiment is not designed for sustained use -- it's used for one off analysis, where it has some utility
         functions that help out in analyzing how an audio net is working or why it's behaving the way it is."""