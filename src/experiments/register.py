from typing import Dict
import logging
from .cec_vs_linear_experiment import CECvsLinearExperiment
from .hyperparameter_searcher_experiment import HyperParameterSearcherExperiment
from .oneshot_learning_experiment import CecOneshotTrainingExperiment
from .tester import Tester
from .disk_ds_learner import DiskDsLearner
from .window_to_spectro import WindowToSpectro
from .sqlite_ds_learner import SQLiteDsLearner
from .audio_analyze_experiment import AudioAnalyzeExperiment
from .preprocessing_test_experiment import PreprocessingTestExperiment
from .spectrogram_analysis_experiment import SpectrogramAnalysisExperiment
from .mean_std_calculator import MeanStdCalculator
from .show_spectrograms_experiment import ShowSpectrogramsExperiment
from .gradcam_experiment import GradCAMExperiment
from .restore_temp_save_to_zoo import RestoreTempSaveToZooExperiment
from .search_experiment import SearchExperiment
from .monte_single_file_tester import MonteSingleFileTester
from .audio_roc_auc_experiment import AudioRocAucExperiment
from .total_audio_accuracy_experiment import TotalAudioAccuracyExperiment
from .audio_cluster_size_variation_experiment import AudioClusterSizeVariationExperiment

logger = logging.getLogger(__name__)


class ExperimentRegistry:
    def __init__(self):
        # Add experiments here so that the runner can pick them up:
        self.experiments = {
            'cec-lin-cifar': CECvsLinearExperiment,
            'hyper-searcher': HyperParameterSearcherExperiment,
            'oneshot': CecOneshotTrainingExperiment,
            'tester': Tester,
            'disk-ds-learner': DiskDsLearner,
            'audio-analyze': AudioAnalyzeExperiment,
            'spectro-precompute': WindowToSpectro,
            'sqlite-ds-learner': SQLiteDsLearner,
            'preprocessing': PreprocessingTestExperiment,
            'spectrogram-analysis': SpectrogramAnalysisExperiment,
            'mean-std-calculator': MeanStdCalculator,
            'show-spectrograms': ShowSpectrogramsExperiment,
            'grad-cam': GradCAMExperiment,
            'restore-temp-save-to-zoo': RestoreTempSaveToZooExperiment,
            'search': SearchExperiment,
            'monte-file-tester': MonteSingleFileTester,
            'audio-roc-auc': AudioRocAucExperiment,
            'total-audio-accuracy': TotalAudioAccuracyExperiment,
            'audio-cluster-size-variation': AudioClusterSizeVariationExperiment,
        }

    def run_experiment(self, experiment_name: str, parameter_overrides: Dict[str, str]):
        if experiment_name not in self.experiments:
            logger.error(f"Experiment `{experiment_name}` not found in the registry,\n"
                         f"Did you mean one of: {', '.join(self.get_experiment_names())}""")
            return
        experiment = self.experiments[experiment_name](parameter_overrides)
        experiment.run()

    def get_experiment_names(self):
        return [x for x in self.experiments]

    def get_experiment_help(self, experiment_name):
        return self.experiments[experiment_name].help_str()
