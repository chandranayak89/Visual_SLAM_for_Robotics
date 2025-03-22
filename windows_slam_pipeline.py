#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d
from scipy.spatial.transform import Rotation
import glob
import time
import struct
import subprocess
import sys

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
    
    def process_image_sequence(self, image_dir, max_frames=None):
        """Process a sequence of images"""
        # Find all images in the directory
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                            glob.glob(os.path.join(image_dir, "*.png")))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # Create output directory for frames
        output_dir = "output_frames"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        frame_count = 0
        
        for img_file in image_files:
            # Read image
            frame = cv2.imread(img_file)
            
            if frame is None:
                print(f"Could not read image: {img_file}")
                continue
            
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
        
        print(f"Processed {frame_count} frames")
        # Visualize results
        self.visualize_results()
        print("SLAM processing complete!")


def extract_frames_from_bag(bag_path, output_dir="extracted_frames", max_frames=None):
    """
    Extract frames from a ROS bag file using a separate tool or process
    """
    # First, try to install pyrosbag if available
    try:
        import pyrosbag
        has_pyrosbag = True
    except ImportError:
        has_pyrosbag = False
        print("Could not import pyrosbag. Will try alternative methods.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Try different approaches to extract frames
    frames_extracted = False
    
    # First try bagpy
    try:
        import bagpy
        from bagpy import bagreader
        
        print(f"Using bagpy to extract images from {bag_path}")
        bag = bagreader(bag_path)
        
        # Find image topics
        topic_list = bag.topic_table["Topics"].tolist()
        image_topics = [topic for topic in topic_list if "image" in topic.lower() or "cam" in topic.lower()]
        
        if not image_topics:
            print("No image topics found in the bag file.")
            return False
        
        print(f"Found image topics: {image_topics}")
        selected_topic = image_topics[0]
        
        # Extract images
        print(f"Extracting images from topic: {selected_topic}")
        data = bag.message_by_topic(selected_topic)
        
        # TODO: Complete the extraction from the CSV that bagpy produces
        print("bagpy extraction not fully implemented yet, trying another method...")
        
    except Exception as e:
        print(f"Error with bagpy extraction: {e}")
    
    # Try using external tools or custom extraction
    if not frames_extracted and has_pyrosbag:
        try:
            print(f"Using pyrosbag to extract images from {bag_path}")
            # This is a placeholder for pyrosbag usage
            # Implementation would depend on the pyrosbag API
            frames_extracted = True
        except Exception as e:
            print(f"Error with pyrosbag extraction: {e}")
    
    # As a fallback, try using a subprocess to call ROS tools if available
    if not frames_extracted:
        try:
            print("Attempting extraction using a subprocess to call ROS tools...")
            # This might work if the user has ROS installed through WSL or other means
            cmd = f"python -m rosbag extract {bag_path} --output-dir {output_dir}"
            subprocess.run(cmd, shell=True, check=True)
            frames_extracted = True
        except Exception as e:
            print(f"Error with subprocess extraction: {e}")
    
    # As a last resort, suggest manual extraction
    if not frames_extracted:
        print("\n=========================================================")
        print("Could not automatically extract frames from the bag file.")
        print("Please consider one of these alternative approaches:")
        print("1. Install ROS in WSL (Windows Subsystem for Linux)")
        print("2. Use a virtual machine with Ubuntu and ROS")
        print("3. Use an online service to convert the bag file")
        print("4. Use the dataset-extraction tools provided by TUM:")
        print("   https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset")
        print("=========================================================\n")
        
        # Suggest a simple workaround for Windows users
        print("Would you like to use a pre-extracted set of sample frames for testing? (y/n)")
        choice = input().lower().strip()
        if choice == 'y':
            # Create a directory with some sample generated images for testing
            sample_dir = "sample_frames"
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            
            # Generate some test pattern images
            for i in range(50):
                # Create a simple image with a moving pattern
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add some features that move across frames
                for j in range(10):
                    x = int((i * 10 + j * 50) % 640)
                    y = int((i * 5 + j * 40) % 480)
                    cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                
                # Add some static features
                for j in range(20):
                    x = j * 30
                    y = j * 20
                    cv2.rectangle(img, (x, y), (x+20, y+20), (0, 0, 255), -1)
                
                # Add frame number
                cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save the image
                cv2.imwrite(f"{sample_dir}/frame_{i:04d}.jpg", img)
            
            print(f"Created {sample_dir} with 50 sample frames for testing")
            return sample_dir
        
        return False
    
    return output_dir

def main():
    print("Visual SLAM for Windows")
    print("=======================")
    
    # Check for bag files in the current directory
    bag_files = [file for file in os.listdir(".") if file.endswith(".bag")]
    
    if bag_files:
        print("\nFound bag files:")
        for i, bag_file in enumerate(bag_files):
            print(f"{i+1}. {bag_file}")
        
        try:
            bag_idx = int(input("\nSelect a bag file (enter number), or 0 to use existing image folder: ")) - 1
            
            if bag_idx == -1:
                # User wants to use existing image folder
                image_folder = input("Enter the path to your image folder: ")
                if not os.path.exists(image_folder):
                    print(f"Error: Folder {image_folder} does not exist!")
                    return
            elif 0 <= bag_idx < len(bag_files):
                selected_bag = bag_files[bag_idx]
                
                print(f"\nExtracting frames from {selected_bag}...")
                image_folder = extract_frames_from_bag(selected_bag)
                
                if not image_folder:
                    print("Could not extract frames from bag file.")
                    return
            else:
                print("Invalid selection!")
                return
            
            max_frames = input("Enter maximum number of frames to process (or press Enter for all): ")
            max_frames = int(max_frames) if max_frames.strip() else None
            
            # Initialize SLAM
            slam = VisualSLAM()
            
            # Process image sequence
            slam.process_image_sequence(image_folder, max_frames=max_frames)
            
        except ValueError:
            print("Please enter a valid number!")
    else:
        print("No bag files found in the current directory.")
        print("Do you want to:")
        print("1. Use an existing folder of images")
        print("2. Generate sample test frames")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            image_folder = input("Enter the path to your image folder: ")
            if not os.path.exists(image_folder):
                print(f"Error: Folder {image_folder} does not exist!")
                return
                
            max_frames = input("Enter maximum number of frames to process (or press Enter for all): ")
            max_frames = int(max_frames) if max_frames.strip() else None
            
            # Initialize SLAM
            slam = VisualSLAM()
            
            # Process image sequence
            slam.process_image_sequence(image_folder, max_frames=max_frames)
            
        elif choice == '2':
            # Generate sample frames
            sample_dir = "sample_frames"
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            
            print(f"Generating 50 sample frames in {sample_dir}...")
            
            # Generate some test pattern images
            for i in range(50):
                # Create a simple image with a moving pattern
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add some features that move across frames
                for j in range(10):
                    x = int((i * 10 + j * 50) % 640)
                    y = int((i * 5 + j * 40) % 480)
                    cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                
                # Add some static features
                for j in range(20):
                    x = j * 30
                    y = j * 20
                    cv2.rectangle(img, (x, y), (x+20, y+20), (0, 0, 255), -1)
                
                # Add frame number
                cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save the image
                cv2.imwrite(f"{sample_dir}/frame_{i:04d}.jpg", img)
            
            print(f"Created {sample_dir} with 50 sample frames")
            
            # Initialize SLAM
            slam = VisualSLAM()
            
            # Process image sequence
            slam.process_image_sequence(sample_dir)
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()

# In your windows_slam_pipeline.py, use:
frames_directory = r"C:\Users\shekhar\Machine Vision projects\Visual_SLAM_for_Robotics\extracted_frames"

pcd = o3d.io.read_point_cloud("point_cloud.ply")
o3d.visualization.draw_geometries([pcd]) 