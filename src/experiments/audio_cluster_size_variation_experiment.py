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


class AudioClusterSizeVariationExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/final22k/train)',
            R.DISKDS_NUM_FILES: '9500',
            R.BATCH_SIZE_TRAIN: '75',
            R.CLUSTERING_MODEL: 'mass',
            R.MODEL_SAVE_PATH: 'zoo/9500massv2',
            # these are the recommendations provided by the GUI with the original training file and TOP-5:
            R.DISKDS_FILE_SUBSET: '5478,4862,5900,4534,3567,5314,2643,2785,6136,7654,8892,5749,279,6181,3593,5899,3387,4856,53,7026,1867,7257,3364,5127,907,2241,1766,5653,6611,4827,4054,477,2912,1441,6496,3080,546,9376,189,1312,1744,7448,3421,1,5356,9122,5118,5558,3379,8573,2007,5651,1251,8894,362,1767,5248,9390,6672,1604,1285,5897,7413,8547,2172,8023,9461,2676,5564,5562,5824,4582,4857,3222,5976,7323,111,6614,5751,3280,8260,4823,5638,3465,2152,4963,2823,2891,3759,2670,6745,7292,9374,2106,4334,9080,5124,7074,5261,3783,4319,548,7947,3064,5012,4089,6754,4325,613,2207,8942,1850,9450,2682,2974,5108,2975,6438,7272,5769,7999,592',
            R.EPOCHS: '40',
            R.BATCH_SIZE_VALIDATION: '150',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: str(1e-3),
            R.DISKDS_WINDOW_HOP_TRAIN: str((2**30)),
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

        with torch.no_grad():
            accuracy_data = self.calculate_accuracy_varying_cluster_size()
            df = pd.DataFrame(data=accuracy_data)
            df.to_csv("cluster_variation_experiment.csv")
            
    def calculate_accuracy_varying_cluster_size(self):
        # fetch "New Order blue monday" - song under test
        song_under_test_idx = 5478
        accuracy_data = {
            'song_under_test_total_samples': [],
            'song_under_test_correct_samples': [],
            'song_under_test_top5_correct_samples': [],
            'other_total_samples': [],
            'other_correct_samples': [],
            'other_top5_correct_samples': [],
            'cluster_reduction': [],
        }
        original_target_cluster_mass = self.model.classification[-1].cluster_mass[song_under_test_idx].item()
        for size_reduction in np.linspace(0, 0.3, 30):
            self.model.classification[-1].cluster_mass[song_under_test_idx] = original_target_cluster_mass - size_reduction * original_target_cluster_mass
            output_data = self.calculate_total_accuracy(self.model)
            df = pd.DataFrame(data=output_data)
            df.set_index('idx')
            # import code
            # code.interact(local=locals())
            song_under_test = df.loc[df['idx'] == song_under_test_idx]
            song_under_test_total_samples = song_under_test["num_samples"]
            song_under_test_correct_samples = song_under_test["num_samples"] * song_under_test["accuracy"]
            song_under_test_top5_correct_samples = song_under_test["num_samples"] * song_under_test["accuracy_top5"]
            other_total_samples = 0
            other_correct_samples = 0
            other_top5_correct_samples = 0
            for index, row in df.loc[df['idx'] != song_under_test_idx].iterrows():
                other_total_samples += row["num_samples"]
                other_correct_samples += row["num_samples"] * row["accuracy"]
                other_top5_correct_samples += row["num_samples"] * row["accuracy_top5"]
            accuracy_data['song_under_test_total_samples'].append(int(round(song_under_test_total_samples)))
            accuracy_data['song_under_test_correct_samples'].append(int(round(song_under_test_correct_samples)))
            accuracy_data['song_under_test_top5_correct_samples'].append(int(round(song_under_test_top5_correct_samples)))
            accuracy_data['other_total_samples'].append(round(other_total_samples))
            accuracy_data['other_correct_samples'].append(round(other_correct_samples))
            accuracy_data['other_top5_correct_samples'].append(round(other_top5_correct_samples))
            accuracy_data['cluster_reduction'].append(size_reduction)
        log.info(f"Accuract data: {accuracy_data}")
        return accuracy_data
            


    def calculate_total_accuracy(self, model):
        pbar = tqdm(enumerate(self.valid_l, 0), total=len(self.valid_l), leave=True)
        running_loss = 0.0
        predicted_correctly = 0
        predicted_total = 0
        criterion = torch.nn.CrossEntropyLoss()
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
            outputs = model(spectrogram)


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
            outputs = outputs.detach().to("cpu").numpy()
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
            output_data['file_name'].append(self.file_list[idx])
            
        return output_data

    @staticmethod
    def help_str():
        return """Validation experiment for the audio models, that runs a bigger accuracy test across the validation set"""