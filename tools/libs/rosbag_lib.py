from tqdm import tqdm

import numpy as np
from scipy.spatial import cKDTree

import rosbag

from .tools_utils import tf_to_matrix, odometry_to_matrix, extract_laser_config, point_cloud2_to_dict, tf_to_dict


class SyncROSBag:
    """
    Synchronize data from a ROS bag file.

    Arguments:
        bag_file (str):
            Path to the ROS bag file.
        pose_from (str, optional):
            Pose source for synchronization. Defaults to 'odom'.
        time_threshold (float, optional):
            Time threshold for data synchronization in seconds. Defaults to 0.1.

    Attributes:
        bag_file (str): Path to the ROS bag file.
        pose_from (str): Pose source for synchronization.
        time_threshold (float): Time threshold for data synchronization in seconds.
        odom_data (dict): Dictionary to store raw odometry data from the ROS bag.
        scan_data (dict): Dictionary to store raw scan data from the ROS bag.
        sync_timestamps (list): List of synchronized timestamps.
        sync_odom_data (list): List of synchronized odometry data.
        sync_scan_data (list): List of synchronized scan data.
        T_base2laser (numpy.ndarray): Static transformation matrix from lidar to base_link.
        lidar_info (dict): 2D LiDAR configuration information.

    Methods:
        get_data(): Get synchronized data, static transformation matrix and 2D LiDAR configuration,.
        _load_data(): Load raw data from the ROS bag.
        _synchronize_data(): Synchronize the loaded data based on timestamps and pose source.
    """
    def __init__(self, bag_file, save_pc_and_tf, inference_bag=False, no_gt=False, pose_from='odom', time_threshold=0.1):
        self.bag_file = bag_file
        self.pose_from = pose_from
        self.time_threshold = time_threshold
        self.save_pc_and_tf = save_pc_and_tf
        self.inference_bag = inference_bag
        self.no_gt = no_gt
        print(self.no_gt)

        # raw data from ROS bag
        self.odom_data = {}
        self.scan_data = {}
        self.gt_pose_data = {}
        if save_pc_and_tf:
            self.pc_data = {}
            self.tf_data = {}

        # aligned data from ROS bag
        self.sync_timestamps = []
        self.sync_odom_data = []
        self.sync_scan_data = []
        self.sync_gt_data = []
        if save_pc_and_tf:
            self.sync_pc_data = []
            self.sync_tf_data = []

        # static transformation from lidar to base_link
        self.T_base2laser = None

        # 2D LiDAR configuration
        self.lidar_info = None

        self._load_data()
        self._synchronize_data()

    def get_data(self):
        """
        Get synchronized data, static transformation matrix, and 2D LiDAR configuration.

        Returns:
            timestamps (numpy.ndarray, (N,)):
                Array of synchronized timestamps.
            scan_data (numpy.ndarray, (N, num_beams)):
                Array of synchronized scan data.
            odom_data (numpy.ndarray, (N, 4, 4)):
                Array of synchronized odometry data,
                transformation matrix from base_link to world frame, T_w2b.
            T_base2laser (numpy.ndarray, (N, 4, 4)):
                Static transformation matrix from lidar to base_link.
            lidar_info (dict):
                2D LiDAR configuration information.
        """
        timestamps = np.array(self.sync_timestamps)
        odom_data = np.array(self.sync_odom_data)
        scan_data = np.array(self.sync_scan_data)
        gt_pose_data = np.array(self.sync_gt_data)
        if self.save_pc_and_tf:
            pc_data = np.array(self.sync_pc_data)
            tf_data = np.array(self.sync_tf_data)
            return timestamps, scan_data, odom_data, self.T_base2laser, self.lidar_info, gt_pose_data, pc_data, tf_data
        return timestamps, scan_data, odom_data, self.T_base2laser, self.lidar_info, gt_pose_data

    def _load_data(self):
        # Open the ROS bag file
        bag = rosbag.Bag(self.bag_file)

        if self.inference_bag == True:
            print("Inference bag detected. Loading additional topics: scan, /platform/odometry, tf, /platform/velodyne_points")
            # Iterate over the messages in the bag
            for topic, msg, t in bag.read_messages(topics=['scan', '/platform/odometry', 'tf', '/platform/velodyne_points']):
                if topic == 'scan':
                    self._process_scan_message(msg)
                elif topic == '/platform/odometry':
                    self._process_odom_message(msg)
                elif topic == 'tf':
                    self._process_tf_message(msg)
                elif topic == '/platform/velodyne_points' and self.save_pc_and_tf:
                    self._process_pointcloud_message(msg)
            
            # If ground truth data is not available (real experiments), use odometry data as ground truth
            if self.no_gt:
                print("Ground truth data not detected. Using odometry data as ground truth.")
                self.gt_pose_data = self.odom_data
        else:
            print("Inference bag not detected. Loading additional topics: /odom, /scan, /tf")
            # Iterate over the messages in the bag
            for topic, msg, t in bag.read_messages(topics=['scan', 'odom', 'tf']):
                if topic == 'scan':
                    self._process_scan_message(msg)
                elif topic == 'odom':
                    self._process_odom_message(msg)
                elif topic == 'tf':
                    self._process_tf_message(msg)

        # Close the ROS bag file
        bag.close()

    def _synchronize_data(self):
        # Convert the timestamps to numpy arrays
        scan_timestamps = np.array(list(self.scan_data.keys()))
        odom_timestamps = np.array(list(self.odom_data.keys()))
        gt_timestamps = np.array(list(self.gt_pose_data.keys()))
        if self.save_pc_and_tf:
            pc_timestamps = np.array(list(self.pc_data.keys()))
            tf_timestamps = np.array(list(self.tf_data.keys()))

        # Build KDTree for faster nearest neighbor search
        odom_kdtree = cKDTree(odom_timestamps[:, None])
  
        # Get the total number of timestamps
        total_timestamps = len(self.scan_data)

        # Iterate over the timestamps in scan_data with a progress bar
        for timestamp in tqdm(scan_timestamps, total=total_timestamps, desc="Synchronizing data"):
            # Find the nearest timestamp in odom_data using KDTree
            _, odom_idx = odom_kdtree.query(timestamp)
            closest_odom_timestamp = odom_timestamps[odom_idx]

            # Check if the time difference is within the threshold
            if abs(timestamp - closest_odom_timestamp) > self.time_threshold:
                continue
            self.sync_odom_data.append(self.odom_data[closest_odom_timestamp])
            # Store the laser scan data with its timestamp
            self.sync_scan_data.append(self.scan_data[timestamp])
            self.sync_timestamps.append(timestamp)
        
        # Build KDTree for ground truth data 
        gt_kdtree = cKDTree(gt_timestamps[:, None])
        
        for timestamp in self.sync_timestamps:
            # Find the nearest timestamp in gt_data using KDTree
            _, gt_idx = gt_kdtree.query(timestamp)
            closest_gt_timestamp = gt_timestamps[gt_idx]

            # Check if the time difference is within the threshold
            if abs(timestamp - closest_gt_timestamp) > self.time_threshold:
                self.sync_gt_data.append(np.zeros((4, 4)))
                continue
            self.sync_gt_data.append(self.gt_pose_data[closest_gt_timestamp])
        if self.save_pc_and_tf:
            # Build KDTree for pc data 
            pc_kdtree = cKDTree(pc_timestamps[:, None])
            
            for timestamp in self.sync_timestamps:
                # Find the nearest timestamp in pc_data using KDTree
                _, pc_idx = pc_kdtree.query(timestamp)
                closest_pc_timestamp = pc_timestamps[pc_idx]

                # Check if the time difference is within the threshold
                if abs(timestamp - closest_pc_timestamp) > self.time_threshold:
                    continue
                self.sync_pc_data.append(self.pc_data[closest_pc_timestamp])

            # Build KDTree for tf data
            tf_kdtree = cKDTree(tf_timestamps[:, None])

            for timestamp in self.sync_timestamps:
                # Find the nearest timestamp in tf_data using KDTree
                _, tf_idx = tf_kdtree.query(timestamp)
                closest_tf_timestamp = tf_timestamps[tf_idx]

                # Check if the time difference is within the threshold
                if abs(timestamp - closest_tf_timestamp) > self.time_threshold:
                    continue
                self.sync_tf_data.append(self.tf_data[closest_tf_timestamp])    
    
    def _process_scan_message(self, msg):
        # Store the LaserScan message with its timestamp (in seconds)
        self.scan_data[msg.header.stamp.to_sec()] = msg.ranges
        # Extract LiDAR configuration
        if self.lidar_info is None:
            self.lidar_info = extract_laser_config(msg)

    def _process_odom_message(self, msg):
        if self.pose_from == 'odom':
            # Store the Odometry message with its timestamp (in seconds)
            self.odom_data[msg.header.stamp.to_sec()] = odometry_to_matrix(msg)

    def _process_tf_message(self, msg):
        # Check for desired transforms
        for transform in msg.transforms:
            if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_link':
                # Store the odom->base_link (gt poses) transform with its timestamp (in seconds)
                self.gt_pose_data[transform.header.stamp.to_sec()] = tf_to_matrix(transform)
            elif self.save_pc_and_tf:
                # Store all the other transforms with their timestamps (in seconds)
                tf_dict = tf_to_dict(transform)
                self.tf_data[transform.header.stamp.to_sec()] = tf_dict

            if transform.header.frame_id == 'base_link' and \
                    transform.child_frame_id == 'laser_link':
                # Store the base_link->laser_link transform (static)
                if self.T_base2laser is None:
                    self.T_base2laser = tf_to_matrix(transform)


    def _process_pointcloud_message(self, msg):
        pc_dict = point_cloud2_to_dict(msg)
        self.pc_data[msg.header.stamp.to_sec()] = pc_dict
