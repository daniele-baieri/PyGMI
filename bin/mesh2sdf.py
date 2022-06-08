import argparse
from ngt.tasks import make_environment, run_task



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Optimizes SDFs over a set of meshes in order to learn their SDF representation.'
    )
    parser.add_argument('config', type=str, help='Path to yaml configuration file')
    parser.parse_args()

    data, task, trainer = make_environment(parser.config)
    F = run_task(data, task, trainer)

