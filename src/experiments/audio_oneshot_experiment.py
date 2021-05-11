import logging
import os
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from src.experiments.base_experiment import BaseExperiment
from src.runners.run_parameters import RunParameters
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
from src.datasets.dataset_provider import DatasetProvider
from src.models.res_net_akamaster_audio import *
from src.models.working_model_loader import *
from src.runners.run_parameter_keys import R
from src.config import Config
from src.datasets.diskds.memory_file_storage import MemoryFileDiskStorage
from torch.utils.data import DataLoader


class AudioOneShotExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/final22k/train)',
            R.DISKDS_NUM_FILES: '9500',
            R.BATCH_SIZE_TRAIN: '75',
            R.CLUSTERING_MODEL: 'mass',
            R.MODEL_SAVE_PATH: 'zoo/9500massv2',
            R.EPOCHS: '40',
            R.BATCH_SIZE_VALIDATION: '150',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: str(1e-3),
            R.DISKDS_WINDOW_HOP_TRAIN: str((2**14)),
            R.DISKDS_WINDOW_HOP_VALIDATION: str((2**16)),
            R.MEASUREMENTS: 'loss,accuracy',
            R.DISKDS_WINDOW_LENGTH: str((2**17)),
            R.DISKDS_TRAIN_FEATURES: 'data,onehot',
            R.DISKDS_VALID_FEATURES: 'data,onehot',
            R.DISKDS_USE_SOX_RANDOM_PRE_SAMPLING_TRAIN: 'True',
        }

    def run(self):
        log = logging.getLogger(__name__)
        self.run_params = super().get_run_params()
        self.config = Config()
        model_save_path = self.run_params.get(R.MODEL_SAVE_PATH)
        self.model, self.file_list = load_working_model(self.run_params, model_save_path)

        self.model = self.model.to(self.config.run_device)
        self.model = self.model.train(False)
        self.spectrogram_generator = SpectrogramGenerator(self.config)
        self.dataset_provider = DatasetProvider()
        self.train_l, self.train_bs, self.valid_l, self.valid_bs = self.dataset_provider.get_datasets(self.run_params)

        sample_output_centroid_positions = torch.Tensor()
        with MemoryFileDiskStorage("/media/andrius/FastBoi/test_samples/05 - Someday.mp3", format="mp3", run_params=self.run_params, features=["data"]) as memory_storage:
        # with MemoryFileDiskStorage("/media/andrius/FastBoi/bakis_data/final22k/train/New Order - Blue Monday.mp3", format="mp3", run_params=self.run_params, features=["data"]) as memory_storage:
            loader = DataLoader(memory_storage, shuffle=False, batch_size=80, num_workers=0)
            pbar = tqdm(loader, total=len(loader), leave=True)
            for item_data in pbar:
                with torch.no_grad():
                    samples = item_data["samples"]
                    samples = samples.to(self.config.run_device)
                    spectrogram = self.spectrogram_generator.generate_spectrogram(
                        samples, narrow_to=128,
                        timestretch=False, random_highpass=False,
                        random_bandcut=False, normalize_mag=True)
                    outputs = self.model(spectrogram)
                    batch_sample_distances = self.model.distance_output.detach().to("cpu")
                    batch_sample_distances = torch.sigmoid(batch_sample_distances)
                    sample_output_centroid_positions = torch.cat([sample_output_centroid_positions, batch_sample_distances])
        log.info(f"Sample output centroid positions {sample_output_centroid_positions.shape} -- \n {sample_output_centroid_positions}")
        log.info(f"Position deviation: {torch.std(sample_output_centroid_positions, dim=0)}")
        log.info(f"Position max: {torch.max(sample_output_centroid_positions, dim=0)}")
        log.info(f"Position min: {torch.min(sample_output_centroid_positions, dim=0)}")
        log.info(f"Position mean: {torch.mean(sample_output_centroid_positions, dim=0)}")
        log.info(f"Position median: {torch.median(sample_output_centroid_positions, dim=0)}")
        mean_positions = torch.mean(sample_output_centroid_positions, dim=0)
        median_positions = torch.median(sample_output_centroid_positions, dim=0).values
        self.model = self.model.to("cpu")
        centroids = self.model.classification[-1].centroids.data
        new_centroids = torch.cat((centroids.to("cpu"), mean_positions.view(1, -1)))
        self.model.classification[-1].centroids = torch.nn.Parameter(new_centroids)

        masses = self.model.classification[-1].cluster_mass.data
        masses = torch.cat((masses, torch.tensor(0.39650386571884155).view(-1)))
        self.model.classification[-1].cluster_mass = torch.nn.Parameter(masses)
        self.model = self.model.to("cuda")
        
        torch.save(self.model.state_dict(), "temp_oneshot.pth")


        self.model.classification[-1].centroids = torch.nn.Parameter(torch.zeros(9501,256))
        self.model.classification[-1].cluster_mass = torch.nn.Parameter(torch.zeros((9501,)))
        self.model.load_state_dict(torch.load("temp_oneshot.pth"))
        self.model = self.model.to("cuda")

        with MemoryFileDiskStorage("/media/andrius/FastBoi/test_samples/05 - Someday.mp3", format="mp3", run_params=self.run_params, features=["data"]) as memory_storage:
        # with MemoryFileDiskStorage("/media/andrius/FastBoi/bakis_data/final22k/train/New Order - Blue Monday.mp3", format="mp3", run_params=self.run_params, features=["data"]) as memory_storage:
            loader = DataLoader(memory_storage, shuffle=False, batch_size=80, num_workers=0)
            pbar = tqdm(enumerate(loader), total=len(loader), leave=True)
            running_loss = 0.0
            predicted_correctly = 0
            predicted_total = 0
            topn = 5
            criterion = torch.nn.CrossEntropyLoss()
            for i, item_data in pbar:
                with torch.no_grad():
                    samples = item_data["samples"]
                    samples = samples.to(self.config.run_device)
                    spectrogram = self.spectrogram_generator.generate_spectrogram(
                        samples, narrow_to=128,
                        timestretch=False, random_highpass=False,
                        random_bandcut=False, normalize_mag=True)
                    outputs = self.model(spectrogram)
                    labels = torch.full((outputs.shape[-2],), 9500).to("cuda")
                    log.info(f"Output vec: {outputs.shape} -- \n {outputs}")
                    log.info(f"Label vec: {labels.shape} -- \n {labels}")
                    loss = criterion(outputs, labels.view(labels.shape[0],))
                    output_cat = torch.argmax(outputs.detach(), dim=1)
                    log.info(f"Output cat: {output_cat.shape} -- \n {output_cat}")
                    labels = labels.view(labels.shape[0],)
                    correct = labels.eq(output_cat).detach()
                    predicted_correctly += correct.sum().item()
                    predicted_total += correct.shape[0]
                    running_loss += loss.item()
                    top_cats = outputs.clone().topk(topn)[1]
                    # log.info(f"Top cats: {top_cats.shape} -- \n {top_cats}")
                    # log.info(f"Labels: {labels.shape} --\n {labels}")
                    target_expanded = labels.detach().view((-1, 1)).expand_as(top_cats).detach()
                    # log.info(f"Targets labels expanded: {target_expanded.shape} --\n {target_expanded}")
                    topk_correct = target_expanded.eq(top_cats)
                    training_summary = f'[{i + 1:03d}](oneshot) loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}'
                    pbar.set_description(training_summary)
            log.info("===========================")
            log.info(f"Of {predicted_total} samples from the one-shot learned song, {predicted_correctly} were predicted correctly - {predicted_correctly/predicted_total}")
            log.info("===========================")
        with torch.no_grad():
            output_data = self.calculate_total_accuracy()
            df = pd.DataFrame(data=output_data)
            df.set_index('idx')
            df.to_csv("oneshot_audio_accuracy.csv")

    def calculate_total_accuracy(self):
        pbar = tqdm(enumerate(self.valid_l, 0), total=len(self.valid_l), leave=True)
        running_loss = 0.0
        predicted_correctly = 0
        predicted_total = 0
        criterion = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax()
        topn = 5
        correct_guesses_by_class_idx = {}
        correct_guesses_top5_by_class_idx = {}
        for i, data in pbar:
            samples = data["samples"]
            samples = samples.to(self.config.run_device)
            spectrogram = self.spectrogram_generator.generate_spectrogram(
                    samples, narrow_to=128,
                    timestretch=False, random_highpass=False,
                    random_bandcut=False, normalize_mag=True)
            labels = data["onehot"].to(self.config.run_device)
            outputs = self.model(spectrogram)


            loss = criterion(outputs, labels.view(labels.shape[0],))
            output_cat = torch.argmax(outputs.detach(), dim=1)
            labels = labels.view(labels.shape[0],)
            correct = labels.eq(output_cat).detach()
            predicted_correctly += correct.sum().item()
            predicted_total += correct.shape[0]
            running_loss += loss.item()
            top_cats = outputs.clone().topk(topn)[1]
            # log.info(f"Top cats: {top_cats.shape} -- \n {top_cats}")
            # log.info(f"Labels: {labels.shape} --\n {labels}")
            target_expanded = labels.detach().view((-1, 1)).expand_as(top_cats).detach()
            # log.info(f"Targets labels expanded: {target_expanded.shape} --\n {target_expanded}")
            topk_correct = target_expanded.eq(top_cats)
            # log.info(f"Correct: {topk_correct.shape} --\n {topk_correct}")
            labels = labels.detach().to("cpu").numpy()
            outputs = softmax(outputs).detach().to("cpu").numpy()
            correct = correct.detach().to("cpu").numpy()
            topk_correct = topk_correct.detach().to("cpu").numpy()
            
            # log.info(f"Outputs: {outputs}")
            # log.info(f"Labels: {labels}")
            # log.info(f"correct: {correct}")
            for b_i, idx in enumerate(labels):
                if idx not in correct_guesses_by_class_idx:
                    correct_guesses_by_class_idx[idx] = [correct[b_i]]
                else:
                    correct_guesses_by_class_idx[idx].append(correct[b_i])
            for b_i, idx in enumerate(labels):
                if idx not in correct_guesses_top5_by_class_idx:
                    correct_guesses_top5_by_class_idx[idx] = [topk_correct[b_i]]
                else:
                    correct_guesses_top5_by_class_idx[idx].append(topk_correct[b_i])
            # log.info(predictions_by_label)
            training_summary = f'[{i + 1:03d}] loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}'
            pbar.set_description(training_summary)
            # break
        output_data = {
            'idx': [],
            'accuracy': [],
            'accuracy_top5': [],
            'num_samples': [],
            'file_name': [],
        }
        for idx in correct_guesses_by_class_idx:
            correctness = np.array(correct_guesses_by_class_idx[idx])
            correctness_top5 = np.array(correct_guesses_top5_by_class_idx[idx])
            num_samples = correctness.size
            accuracy = (correctness*1).sum()/num_samples
            accuracy_top5 = (correctness_top5*1).sum()/num_samples
            output_data['idx'].append(idx)
            output_data['accuracy'].append(accuracy)
            output_data['accuracy_top5'].append(accuracy_top5)
            output_data['num_samples'].append(num_samples)
            if idx in self.file_list:
                output_data['file_name'].append(self.file_list[idx])
            else:
                output_data['file_name'].append(-1)
            
        return output_data

    @staticmethod
    def help_str():
        return """Experiment that tries to learn a song using one shot learning abilities of the clustering based classification models"""