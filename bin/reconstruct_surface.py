import argparse
import yaml



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='\
            Optimizes SDFs over a set of point clouds in order to reconstruct their original surface.\
            Can also be used to simply encode a set of meshes/point clouds into signed distance functions.'
    )
    parser.add_argument('config', type=str, help='Path to yaml configuration file')
    parser.parse_args()
    conf = yaml.safe_load(open(parser.config))

