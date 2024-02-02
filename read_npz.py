import argparse

import matplotlib.pyplot as plt
import numpy as np

from utils import load_data, particles2pose


def get_args():
    parser = argparse.ArgumentParser()

    # results
    parser.add_argument('--loc_results', type=str,
                        default='results/simulation/SE1/T1/loc_results_T1.npz',
                        help='the file path of localization results')

    # map
    parser.add_argument('--occ_map', type=str, default='data/simulation/occmap_T1.npy',
                        help='the file path of the occupancy grid map for visualization.')
    parser.add_argument('--map_size', nargs='+', type=float, default=[-4.5, 4, -7.5, 7.1],
                        help='the size of the map.')

    # output GIF
    parser.add_argument('--output_gif', type=str,
                        default=None,
                        help='the GIF path for saving the localization process.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    loc_results = np.load(args.loc_results)

    timestamps = loc_results['timestamps']
    odoms = loc_results['odoms']
    poses_gt = loc_results['poses_gt']
    particles = loc_results['particles']
    start_idx = loc_results['start_idx']
    numParticles = loc_results['numParticles']

    if args.occ_map:
        occ_map = np.load(args.occ_map)
    else:
        occ_map = None
    
    localized = False

    for i in range(start_idx, len(timestamps)):
        estimated_pose = particles2pose(particles[i])
        print("Timestamp: ", timestamps[i])
        print("Estimated Pose: ", estimated_pose)
        print("Ground Truth Pose: ", poses_gt[i])
        print("Odometry: ", odoms[i])
        loc_error = np.linalg.norm(estimated_pose - poses_gt[i])
        print("Localization Error: ", loc_error)
        if(loc_error < 0.1 and not localized):
            start_idx = i
            localized = True


    print("Start Index: ", start_idx)
    print("Num Frames ", len(timestamps))