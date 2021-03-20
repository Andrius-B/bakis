import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class CEClustering(nn.Module):
    def __init__(
            self,
            n_dim,
            n_clusters,
            init_radius = 0.4,
            init_pos_offset = 0.2,
            init_pos_var = 0.4
        ):
        super().__init__()
        self.out_features = n_clusters
        self.in_features = n_dim
        c = torch.mul(torch.rand((n_clusters, n_dim)), init_pos_var)
        c = torch.add(c, torch.full((n_clusters, n_dim), init_pos_offset))
        self.centroids = nn.Parameter(c)
        # this shall be the radius of each cluster
        s = torch.mul(torch.ones((n_clusters,)), init_radius)
        self.cluster_sizes = nn.Parameter(s)
        self.max_dist = math.sqrt(self.in_features)

    def forward(self, input):
        # centroids out of the output latent space don't make sense - clamp them in
        self.centroids.data = torch.clamp(self.centroids, 0, 1)
        self.cluster_sizes.data = torch.clamp(self.cluster_sizes, 1e-3, 3)
        # we need to convert the input point so that we can
        # perform other operations as matrix ops.
        # this makes input into (n, 2) where n is the number of centroids
        # print(f"Centroids: {self.centroids.shape} -- \n{self.centroids}")
        # print(f"Radii: \n{self.cluster_sizes.shape} --\n{self.cluster_sizes}")
        # print(f"Input:\n{input}")
        num_centroids = self.centroids.shape[0]
        input_repeated = input.repeat((1, num_centroids)).view(-1, num_centroids, self.in_features)
        # print(f"Input repeated:\n{input_repeated.shape}")
        # input_distances contains:
        # [[dx1, dy1]
        #  [dx2, dy2]] - the distances of input to the centroid
        # print(f"input_rep: {input_repeated.shape} centroids: {self.centroids.shape}")
        input_distances = self.centroids - input_repeated
        # print(f"Distances:\n{input_distances.shape}")
        # by taking the second norm in dim 1 we get:
        # [
        #   dx1*dx1 + dy1*dy1,
        #   dx2*dx2 + dy2*dy2
        # ]
        # which is the Euclidean distance of input to each centroid
        input_distances = torch.norm(input_distances, p=2, dim=-1)
        input_distances = torch.div(input_distances, self.max_dist)
        # print(f"Reduced distances:\n{input_distances.shape}")
        # inverse distance - the closer the better.
        # note to self: I think inverse distance would reinfoce overfitting,
        # so i'm just using max distance in the latent space ~ sqrt(2)

        transformed_input_distances = torch.mul(input_distances, torch.div(1, self.cluster_sizes))
        # print(f"Transformed distances:\n{transformed_input_distances.shape}")
        distance_probabilities = torch.add(-torch.pow(transformed_input_distances, 2), 1)
        # print(f"Inverse input distances:\n{distance_probabilities.shape}")
        # normalized = 4*nn.functional.sigmoid(distance_probabilities) - 1
        return distance_probabilities