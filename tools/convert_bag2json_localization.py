import argparse

from libs.rosbag_lib import SyncROSBag
from libs.format_convert_lib import DataToJsonConverter


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--bag_file', type=str,
                        default='~/ir-mcl/data/intel/intel.bag',
                        help='the file path of rosbag')
    parser.add_argument('--output_json', type=str,
                        default='~/ir-mcl/data/intel/loc_test/seq.json',
                        help='the directory for saving the output Json file')
    parser.add_argument('--save_pc_and_tf', action="store_true",
                        help='whether to save point cloud and tf data to the output json file. Default is False.')
    parser.add_argument('--inference_bag', action="store_true",
                        help='names of topics changes according to inference bag nomenclature. Default is False.')
    parser.add_argument('--no_gt', action="store_true",
                        help='if ground truth poses are not available, use odometry as gt. Default is False.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.save_pc_and_tf:    
        ############## Step 1: Load bag file ##############
        print("-> Loading bag file... saving point cloud and tf data to the output json file.")
        # Create an instance of SyncROSBag
        sync_bag = SyncROSBag(args.bag_file, save_pc_and_tf=True, inference_bag=args.inference_bag, no_gt=args.no_gt)
        timestamps, scan_data, odom_data, T_b2l, lidar_info, gt_pose_data, pc_data, tf_data = sync_bag.get_data()

        print("-> Done!")

        ############## Step 2: Save to JSON ##############
        # We assume the ground truth is the same as the odometry in the example!
        # In practice, you may want to use real ground truth data.
        # ToDO: loading ground truth data from a file
        # ToDO: synchronization ros records with ground truth data

        json_converter = DataToJsonConverter(
            timestamps=timestamps,
            odom_data=odom_data,
            gt_pose_data=gt_pose_data,
            scan_data=scan_data,
            T_b2l=T_b2l,
            lidar_info=lidar_info,
            pc_data=pc_data,
            tf_data=tf_data
        )

        # convert to json for localization
        json_converter.convert_to_json_localization(args.output_json)

    else:
                    
        ############## Step 1: Load bag file ##############
        print("-> Loading bag file... not saving point cloud and tf data to the output json file.")

        # Create an instance of SyncROSBag
        sync_bag = SyncROSBag(args.bag_file, save_pc_and_tf=False, inference_bag=args.inference_bag, no_gt=args.no_gt)
        timestamps, scan_data, odom_data, T_b2l, lidar_info, gt_pose_data = sync_bag.get_data()

        print("-> Done!")

        ############## Step 2: Save to JSON ##############
        # We assume the ground truth is the same as the odometry in the example!
        # In practice, you may want to use real ground truth data.
        # ToDO: loading ground truth data from a file
        # ToDO: synchronization ros records with ground truth data

        json_converter = DataToJsonConverter(
            timestamps=timestamps,
            odom_data=odom_data,
            gt_pose_data=gt_pose_data,
            scan_data=scan_data,
            T_b2l=T_b2l,
            lidar_info=lidar_info,
 
        )

        # convert to json for localization
        json_converter.convert_to_json_localization(args.output_json)