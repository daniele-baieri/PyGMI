import argparse
import dill
from ngt.tasks import make_environment, run_task



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Optimizes SDFs over a set of meshes in order to learn their SDF representation.'
    )
    parser.add_argument('config', type=str, help='Path to yaml configuration file')
    parser.add_argument('out_file', type=str, help='Path to output file to save optimized function')
    parser.parse_args()

    data, task, trainer = make_environment(parser.config)
    F = run_task(data, task, trainer)

    autodec = getattr(task, 'autodecoder', None)
    with open(parser.out_file, 'wb') as f:
        dill.dump(F, f)
        if autodec is not None:
            dill.dump(autodec, f)
