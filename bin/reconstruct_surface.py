import argparse
import dill
from ngt.tasks import make_environment, run_task



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Optimizes an SDF over a point cloud in order to reconstruct its original surface.'
    )
    parser.add_argument('config', type=str, help='Path to yaml configuration file')
    parser.add_argument('out_file', type=str, help='Path to output file to save optimized function')
    parser.parse_args()

    data, task, trainer = make_environment(parser.config)
    F = run_task(data, task, trainer)

    with open(parser.out_file, 'wb') as f:
        dill.dump(F, f)