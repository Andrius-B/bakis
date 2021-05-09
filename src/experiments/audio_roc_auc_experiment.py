import logging
import os
import torch
from tqdm import tqdm
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from torchsummary import summary
from src.experiments.base_experiment import BaseExperiment
from src.runners.run_parameters import RunParameters
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
from src.datasets.dataset_provider import DatasetProvider
from src.models.res_net_akamaster_audio import *
from src.models.working_model_loader import *
from src.runners.run_parameter_keys import R
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from src.config import Config

@dataclass
class RocAucResult:
    class_idx: int
    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    thresholds: np.ndarray
    roc_auc_score: float

class AudioRocAucExperiment(BaseExperiment):

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/final22k/train)',
            R.DISKDS_NUM_FILES: '9500',
            R.DISKDS_FILE_SUBSET: '1,2,3,4,5,6',
            R.BATCH_SIZE_TRAIN: '75',
            R.CLUSTERING_MODEL: 'mass',
            R.MODEL_SAVE_PATH: 'zoo/9500massv2',
            R.EPOCHS: '40',
            R.BATCH_SIZE_VALIDATION: '150',
            R.TRAINING_VALIDATION_MODE: 'epoch',
            R.LR: str(1e-3),
            R.DISKDS_WINDOW_HOP_TRAIN: str((2**20)),
            R.DISKDS_WINDOW_HOP_VALIDATION: str((2**12)),
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

        # roc and auc visualization
        self.show_roc_auc_curves()

    def show_roc_auc_curves(self):
        with torch.no_grad():
            roc_auc_results = self.calculate_roc_auc_simple([1, 2, 3, 4, 5, 6])
            for r in roc_auc_results:
                name = os.path.basename(self.file_list[r.class_idx])
                plt.plot(r.false_positive_rate, r.true_positive_rate, marker='.', label=f'class {r.class_idx} AUC={r.roc_auc_score}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.show()

    def calculate_roc_auc_simple(self, idxs_to_analyze) -> List[RocAucResult]:
        pbar = tqdm(enumerate(self.valid_l, 0), total=len(self.valid_l), leave=True)
        running_loss = 0.0
        predicted_correctly = 0
        predicted_total = 0
        criterion = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax()
        predictions_by_label = {}
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
            labels = labels.detach().to("cpu").numpy()
            outputs = softmax(outputs).detach().to("cpu").numpy()
            # log.info(f"Outputs: {outputs}")
            # log.info(f"Labels: {labels}")
            # log.info(f"correct: {correct}")
            for i, label in enumerate(labels):
                if label not in predictions_by_label:
                    predictions_by_label[label] = [outputs[i]]
                else:
                    predictions_by_label[label].append(outputs[i])
            # log.info(predictions_by_label)
            training_summary = f'[{i + 1:03d}] loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}'
            pbar.set_description(training_summary)
            # break
        results = []
        collected_predictions = []
        for idx in idxs_to_analyze:
            collected_predictions.extend(predictions_by_label[idx])
        collected_predictions = np.array(collected_predictions)
        log.info(f"Collected predictions: {collected_predictions.shape} -- \n {collected_predictions.shape}")
        for idx in idxs_to_analyze:
            true_classes = []
            for idx_tmp in idxs_to_analyze:
                true_classes.extend([int(idx_tmp == idx)]*len(predictions_by_label[idx_tmp]))
            true_classes = np.array(true_classes)
            class_preds = collected_predictions[:, idx]
            log.info(f"roc_auc data:")
            log.info(f"Classes: {true_classes.shape} -- \n {true_classes.shape}")
            log.info(f"preds: {class_preds.shape} -- \n {class_preds}")
            fpr, tpr, thresholds = roc_curve(true_classes, class_preds)
            score = roc_auc_score(true_classes, class_preds)
            results.append(RocAucResult(idx, fpr, tpr, thresholds, score))
        return results

    @staticmethod
    def help_str():
        return """Validation experiment for the audio models, that displays ROC curves"""