import logging
import os
from src.experiments.base_experiment import BaseExperiment
from src.runners.run_parameters import RunParameters
from src.runners.run_parameter_keys import R
from src.models.res_net_akamaster_audio import *
from src.models.working_model_loader import *
from src.runners.spectrogram.spectrogram_generator import SpectrogramGenerator
from src.datasets.diskds.sox_transforms import FileLoadingSoxEffects
from src.config import Config
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import cv2

log = logging.getLogger(__name__)


class GradCAMExperiment(BaseExperiment):
    # references:
    # https://arxiv.org/pdf/1610.02391v1.pdf
    # https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

    def get_experiment_default_parameters(self):
        return {
            R.DATASET_NAME: 'disk-ds(/media/andrius/FastBoi/bakis_data/final22k/train)',
            R.DISKDS_NUM_FILES: '9500',
            R.BATCH_SIZE_TRAIN: '75',
            R.CLUSTERING_MODEL: 'mass',
            R.MODEL_SAVE_PATH: 'zoo/9500massv2',
        }

    def load_spectrogram(self, filepath, offset_frames, length_frames):
        if not os.path.exists(filepath):
            log.error(f"Requested test file not found at: {filepath}")
            raise RuntimeError(f"File not found: {filepath}")

        samples, sample_rate = torchaudio.backend.sox_backend.load(
            filepath,
            offset=offset_frames,
            num_frames=length_frames,
            normalization=False,
        )
        # sample_path = os.path.join(os.path.dirname(filepath), "samples", f"sample_{os.path.basename(filepath)}")
        # torchaudio.backend.sox_backend.save(sample_path, samples, 41000)
        samples = samples[0].view((1, -1))  # only take one channel.
        log.info(f"loaded samples shape: {samples.shape}")
        samples, sample_rate = FileLoadingSoxEffects(sample_rate, Config().sample_rate, False).forward(samples)
        samples = samples.view(1, 1, -1)
        log.info(f"Loaded samples reshaped to: {samples.shape}")
        raw_samples = samples[0][0].cpu().numpy()
        config = Config()
        config.run_device = torch.device("cpu")
        spectrogram_generator = SpectrogramGenerator(config)
        return spectrogram_generator.generate_spectrogram(samples, normalize_mag=True, random_poly_cut=False, inverse_poly_cut=False, narrow_to=128).cpu()

    def run(self):
        log = logging.getLogger(__name__)
        run_params = super().get_run_params()
        model_save_path = run_params.get(R.MODEL_SAVE_PATH)
        model, filelist = load_working_model(run_params, model_path=model_save_path, reload_classes_from_dataset=False)
        model.save_gradient = True
        model.cpu()
        # test_filepath = "/home/andrius/git/bakis/data/test_data/blue_monday_desktop_mic_speaker_youtube_resampled.mp3"
        test_filepath = "/media/andrius/FastBoi/bakis_data/final22k/train/Adele - Hello.mp3"
        spectrogram_raw = self.load_spectrogram(test_filepath, 40*22050, 2**17)
        spectrogram = spectrogram_raw.clone()
        print(spectrogram.shape)

        predictions = model(spectrogram)
        # print(f"Predictions: {predictions.shape} -- \n {predictions}")
        print(f"{predictions.argmax(dim=-1)} => {predictions.max()}")
        print(filelist[int(predictions.argmax(dim=-1))])
        target_class = predictions.argmax(dim=-1)
        loss = torch.nn.CrossEntropyLoss()
        print(f"Prediction vector: {predictions}")
        print(f"Prediction vector stats: min={predictions.min()}, max={predictions.max()}")
        target = torch.zeros((1))
        target[0] = 1
        loss_value = loss(predictions, target.long())
        print(f"Loss with a o-h-e vector: {loss_value}")
        # target_class = 36
        # return
        # here we do the cheeky trick from the paper -
        # create a onehot encoded vector with one in the target position
        # reference: https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py
        one_hot = np.zeros((1, predictions.size()[-1]), dtype=np.float32)
        one_hot = one_hot * -10
        one_hot[0][target_class] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        one_hot = torch.sum(one_hot * predictions)

        # propagate the gradients backward via auto differentiation of pytorch
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        resnet_gradients = model.get_resnet_gradient()
        mean_gradients_per_pane = torch.mean(resnet_gradients, dim=(0, 2, 3))
        resnet_activation = model.get_resnet_activations(spectrogram).detach()

        # print(f"mean_gradients: {mean_gradients_per_pane.shape} -- \n {mean_gradients_per_pane}")
        # print(f"resnet_activations: {resnet_activation.shape} -- \n {resnet_activation}")

        for i, mean_gradient in enumerate(mean_gradients_per_pane):
            resnet_activation[:, i, :, :] *= mean_gradient

        heatmap = torch.mean(resnet_activation, dim=1)[0]
        # print(f"generated heatmap: {heatmap.shape} -- \n{heatmap}")

        heatmap = np.maximum(heatmap, 0)
        if(torch.max(heatmap) > 0):
            heatmap /= torch.max(heatmap)
        heatmap = heatmap.numpy()
        # plt.matshow(heatmap)
        # plt.show()
        # print(f"Heatmap before resize: {heatmap}")
        spectrogram = spectrogram.view((spectrogram.shape[-2], spectrogram.shape[-1])).numpy()
        # print(f"Spectrogram before resize: {spectrogram.shape} -- \n{spectrogram}")
        heatmap = cv2.resize(heatmap, (spectrogram.shape[1], spectrogram.shape[0]))
        # print(f"Heatmap after resize: {heatmap.shape} -- \n {heatmap}")
        # print(f"Spectrogram before show: {spectrogram.shape} -- \n {spectrogram}")

        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title(os.path.basename(test_filepath) + (" (Clean)"))
        ax.imshow(np.flip(spectrogram, axis=-2), cmap='plasma')

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title(os.path.basename(test_filepath) + " (Grad-CAM)")
        ax.imshow(np.flip(spectrogram, axis=-2), cmap='gray')
        ax.imshow(np.flip(heatmap, axis=-2), cmap='plasma', alpha=0.4)

        ax = fig.add_subplot(2, 2, 3)
        ax.set_title(os.path.basename(test_filepath) + (" (Masked)"))
        spectrogram = spectrogram_raw.clone()
        mask_value = (spectrogram.max() - spectrogram.min()) / 2
        mask_size = (8, 8)
        masked_spectrogram = spectrogram.clone()
        masked_spectrogram[:, :, 10:(10+mask_size[0]), 20:(20+mask_size[1])] = mask_value
        masked_spectrogram = masked_spectrogram.numpy()
        ax.imshow(np.flip(masked_spectrogram[0][0], axis=-2), cmap='plasma')

        ax = fig.add_subplot(2, 2, 4)
        ax.set_title(os.path.basename(test_filepath) + " (Occluded class probabilities)")
        class_heatmap = self.generate_occlusion_class_probability_heatmap(model, spectrogram, target_class)
        # print(f"Generated class heatmap: {class_heatmap.shape} -- \n {class_heatmap}")
        spectrogram = spectrogram
        class_heatmap = cv2.resize(class_heatmap.detach().numpy(), (spectrogram.shape[-1], spectrogram.shape[-2]))
        # print(f"spectrogram shape before showing: {spectrogram.shape}")
        ax.imshow(np.flip(spectrogram_raw[0][0].numpy(), axis=-2), cmap='gray')
        ax.imshow(np.flip(class_heatmap, axis=-2), cmap='plasma', alpha=0.4)

        plt.tight_layout()
        plt.show()

    def generate_occlusion_class_probability_heatmap(self, model: ResNet, spectrogram: torch.Tensor, target_class: int, mask_size=(16, 16), stride=4) -> torch.Tensor:
        """
        Implementation of class-probability heatmap based on https://arxiv.org/pdf/1311.2901.pdf
        """
        mask_value = (spectrogram.max() - spectrogram.min()) / 2
        _, _, spec_height, spec_width = spectrogram.shape
        output_heatmap = torch.zeros(int((spec_height - mask_size[0])/stride), int((spec_width - mask_size[1])/stride))
        print(f"Selected mask value: {mask_value}")
        print(f"Expected output size: {output_heatmap.shape}")
        conv_x, conv_y = 0, 0
        model.save_gradient = False
        with torch.no_grad():
            while conv_y < (spec_height - mask_size[0])/stride:
                while conv_x < (spec_width - mask_size[1])/stride:
                    masked_spectrogram = spectrogram.clone()
                    # print(f"Applying mask at: ({conv_y}:{conv_x})")
                    mask_pos_x = conv_x * stride
                    mask_pos_y = conv_y * stride
                    masked_spectrogram[:, :, mask_pos_y:(mask_pos_y+mask_size[0]), mask_pos_x:(mask_pos_x+mask_size[1])] = mask_value
                    predictions = model(masked_spectrogram)
                    output_heatmap[conv_y][conv_x] = predictions[:, target_class].mean()
                    conv_x += 1
                conv_x = 0
                conv_y += 1
        # output_heatmap += torch.min(output_heatmap)
        # output_heatmap /= torch.max(output_heatmap)
        # output_heatmap = output_heatmap
        return output_heatmap

    @staticmethod
    def help_str():
        return """Tries to visualize how the network chose a specific class using gradcam """
