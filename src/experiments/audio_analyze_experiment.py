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
import math
import torch
import matplotlib.cm
import pandas as pd
import plotly.graph_objects as go

cmap = matplotlib.cm.get_cmap('plasma')

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
    
    def run_centroids_tsne(self, tsne_data, classes = None):
        tsne_fn = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=10000)
        tsne_results = tsne_fn.fit_transform(tsne_data)
        centroids = tsne_results
        print(f"Transformed centroids: {centroids.shape}")
        centroids_list = []
        for centroid_idx, centroid_pos in enumerate(centroids):
            color = cmap(centroid_idx/(len(centroids)))
            classText = str(centroid_idx)
            if classes is not None:
                classText = classes[centroid_idx]
            centroids_list.append({
                'xpos': centroid_pos[0],
                'ypos': centroid_pos[1],
                'color': color,
                'text': classText
            })
        centroids_tsne = {k: [dic[k] for dic in centroids_list] for k in centroids_list[0]}
        # plt.scatter(centroid_pos[0], centroid_pos[1], color=color, marker="x", s=15)
        # if(centroid_idx == max_item):
        #     plt.text(centroid_pos[0], centroid_pos[1], files[centroid_idx])
        # else:
        #     plt.text(centroid_pos[0], centroid_pos[1], files[centroid_idx], alpha=0.3)
        data = [
                go.Scattergl(
                    x=centroids_tsne['xpos'],
                    y=centroids_tsne['ypos'],
                    # mode='markers+text',
                    mode='markers',
                    marker=dict(
                        color=centroids_tsne['color'],
                        line_width=1,
                        symbol="x",
                        size=10,
                    ),
                    text=centroids_tsne['text']
                ),
            ]
        fig = go.Figure(
            data=data
        )
        fig.write_html('t-sne.html', auto_open=True)


    def load_cec_from_csv(self, csv_path) -> (CEClustering, List[str]):
        pass

    def get_experiment_default_parameters(self):
        return {
            R.CLUSTERING_MODEL: 'mass',
            R.DISKDS_NUM_FILES: '5000'
        }

    def run(self):
        run_params = super().get_run_params()
        method = 'T-SNE'
        net, files = load_working_model(run_params, 'zoo/5000massv1', reload_classes_from_dataset=False)
        clustering_module = net.classification[-1].cpu()
        # self.show_distance_from_zero_hist(clustering_module)
        tsne_data = clustering_module.centroids.detach().numpy()
        self.run_centroids_tsne(tsne_data, files)
        

    def help_str(self):
        return """This experiment is not designed for sustained use -- it's used for one off analysis, where it has some utility
         functions that help out in analyzing how an audio net is working or why it's behaving the way it is."""