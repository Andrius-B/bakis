import argparse
from src.experiments.register import ExperimentRegistry

experiment_registry = ExperimentRegistry()

def generate_help_message():
    experiments = ', '.join(experiment_registry.get_experiment_names())
    return f"Runnable experiments: {experiments}\n"

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(generate_help_message())
    parser.add_argument("experiment", type=str,
                        help="Which experiment to run")
    args = parser.parse_args()
    experiment_registry.run_experiment(args.experiment, None)


