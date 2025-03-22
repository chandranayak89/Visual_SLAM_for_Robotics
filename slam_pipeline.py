#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import os
from cv_bridge import CvBridge
import open3d as o3d
from scipy.spatial.transform import Rotation

class VisualSLAM:
    def __init__(self):
        # Initialize ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Camera intrinsic parameters (adjust these for your specific camera)
        self.K = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1]
        ])
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
        # Initialize trajectory and point cloud
        self.trajectory = []
        self.point_cloud = []
        
        # Set initial camera pose
        self.curr_R = np.eye(3)
        self.curr_t = np.zeros((3, 1))
        
        # Previous frame data
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        
    def process_frame(self, frame):
        """Process a single frame for SLAM"""
        if frame is None:
            return None
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect keypoints
        kp, des = self.orb.detectAndCompute(gray, None)
        
        # If this is the first frame, just store the keypoints
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            self.trajectory.append((0, 0, 0))  # Initial position
            return frame
        
        # Match features
        if des is not None and self.prev_des is not None and len(des) > 0 and len(self.prev_des) > 0:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use only the best matches
            good_matches = matches[:50]
            
            # Estimate motion if we have enough good matches
            if len(good_matches) >= 8:
                # Get matched point coordinates
                prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])
                
                # Calculate essential matrix
                E, mask = cv2.findEssentialMat(curr_pts, prev_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    # Recover pose from essential matrix
                    _, R, t, mask = cv2.recoverPose(E, curr_pts, prev_pts, self.K)
                    
                    # Update current camera pose
                    self.curr_t = self.curr_t + self.curr_R.dot(t)
                    self.curr_R = R.dot(self.curr_R)
                    
                    # Triangulate points to get 3D coordinates
                    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = np.hstack((R, t))
                    P1 = self.K.dot(P1)
                    P2 = self.K.dot(P2)
                    
                    # Reshape points for triangulation
                    prev_pts_normalized = cv2.undistortPoints(prev_pts.reshape(-1, 1, 2), self.K, None)
                    curr_pts_normalized = cv2.undistortPoints(curr_pts.reshape(-1, 1, 2), self.K, None)
                    
                    # Triangulate points
                    points_4d = cv2.triangulatePoints(P1, P2, 
                                                    prev_pts_normalized.reshape(-1, 2).T,
                                                    curr_pts_normalized.reshape(-1, 2).T)
                    
                    # Convert to 3D points
                    points_3d = points_4d[:3] / points_4d[3]
                    
                    # Add points to point cloud
                    for i in range(points_3d.shape[1]):
                        point = points_3d[:, i]
                        if abs(point[2]) < 50:  # Filter out points that are too far away
                            self.point_cloud.append(point)
            
            # Draw matches for visualization
            vis_frame = cv2.drawMatches(self.prev_frame, self.prev_kp, gray, kp, good_matches, None)
            
            # Store current position for trajectory
            self.trajectory.append((self.curr_t[0, 0], self.curr_t[1, 0], self.curr_t[2, 0]))
            
            # Update previous frame data
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            
            return vis_frame
        
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        return frame
        
    def visualize_results(self):
        """Visualize trajectory and point cloud"""
        # Plot trajectory
        trajectory = np.array(self.trajectory)
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(trajectory[:, 0], trajectory[:, 2])
        plt.title('Camera Trajectory (Top View)')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.title('Camera Trajectory (Side View)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("trajectory.png")
        plt.close()
        
        # Visualize point cloud with Open3D
        if len(self.point_cloud) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(self.point_cloud))
            o3d.io.write_point_cloud("point_cloud.ply", pcd)
            print("Point cloud saved to 'point_cloud.ply'")
    
    def process_bag_file(self, bag_path, image_topic="/camera/image_raw", max_frames=None):
        """Process a ROS bag file"""
        bridge = CvBridge()
        
        if not os.path.exists(bag_path):
            print(f"Error: Bag file {bag_path} does not exist!")
            return
            
        print(f"Processing bag file: {bag_path}")
        bag = rosbag.Bag(bag_path)
        frame_count = 0
        
        # Create output directory for frames
        output_dir = "output_frames"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # Get the total number of messages (for progress reporting)
            total_msgs = bag.get_message_count(topic_filters=[image_topic])
            print(f"Total frames in bag: {total_msgs}")
            
            # Process each frame
            for topic, msg, t in bag.read_messages(topics=[image_topic]):
                try:
                    # Convert ROS image to OpenCV image
                    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    
                    # Process frame for SLAM
                    vis_frame = self.process_frame(frame)
                    
                    # Save frame with features
                    if vis_frame is not None:
                        cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", vis_frame)
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"Processed {frame_count} frames")
                    
                    if max_frames is not None and frame_count >= max_frames:
                        print(f"Reached maximum number of frames ({max_frames})")
                        break
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error reading bag file: {e}")
        finally:
            bag.close()
            
        print(f"Processed {frame_count} frames")
        # Visualize results
        self.visualize_results()
        print("SLAM processing complete!")

def main():
    # Create a list of available bag files
    available_bags = [
        file for file in os.listdir(".")
        if file.endswith(".bag") and os.path.isfile(file)
    ]
    
    if not available_bags:
        print("No bag files found in the current directory!")
        return
    
    print("\nAvailable bag files:")
    for i, bag in enumerate(available_bags):
        print(f"{i+1}. {bag}")
    
    try:
        bag_idx = int(input("\nSelect a bag file (enter number): ")) - 1
        if bag_idx < 0 or bag_idx >= len(available_bags):
            print("Invalid selection!")
            return
        
        selected_bag = available_bags[bag_idx]
        max_frames = input("Enter maximum number of frames to process (or press Enter for all): ")
        max_frames = int(max_frames) if max_frames.strip() else None
        
        # Initialize SLAM
        slam = VisualSLAM()
        
        # Process the selected bag file
        slam.process_bag_file(selected_bag, max_frames=max_frames)
        
    except ValueError:
        print("Please enter a valid number!")

if __name__ == "__main__":
    main() 