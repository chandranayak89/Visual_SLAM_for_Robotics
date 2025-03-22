#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import subprocess
import glob
import time

def generate_synthetic_dataset(output_dir="synthetic_dataset", num_frames=100):
    """
    Generate a synthetic dataset for testing SLAM when bag files cannot be processed
    
    Args:
        output_dir (str): Directory to save the synthetic frames
        num_frames (int): Number of frames to generate
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generating {num_frames} synthetic frames in {output_dir}...")
    
    # Parameters for a simulated camera trajectory
    width, height = 640, 480
    center_x, center_y = width // 2, height // 2
    radius = 200
    
    # Create some random 3D points that will be visible in the frames
    num_points = 50
    points_3d = np.random.uniform(-5, 5, (num_points, 3))
    points_3d[:, 2] += 10  # Move points in front of the camera
    
    # Define camera intrinsics
    fx, fy = 525.0, 525.0
    cx, cy = 320.0, 240.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    for i in range(num_frames):
        # Create a blank image
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simulate camera motion in a circle
        angle = 2 * np.pi * i / num_frames
        # Position of camera
        x = radius * np.cos(angle)
        y = 0
        z = radius * np.sin(angle)
        
        # Rotation matrix - camera looking at origin
        look_at = np.array([0, 0, 0]) - np.array([x, y, z])
        look_at = look_at / np.linalg.norm(look_at)
        
        # Up vector
        up = np.array([0, 1, 0])
        
        # Right vector
        right = np.cross(look_at, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up to ensure orthogonality
        up = np.cross(right, look_at)
        
        # Rotation matrix (world to camera)
        R = np.vstack([right, up, -look_at])
        
        # Translation vector
        t = -R @ np.array([x, y, z])
        
        # Project 3D points to 2D
        for point in points_3d:
            # Transform point to camera coordinates
            p_camera = R @ point + t
            
            # Only draw points in front of the camera
            if p_camera[2] > 0:
                # Project to image plane
                p_image = K @ p_camera
                p_image = p_image / p_image[2]
                
                # Check if point is within image bounds
                if 0 <= p_image[0] < width and 0 <= p_image[1] < height:
                    # Draw the point
                    cv2.circle(frame, (int(p_image[0]), int(p_image[1])), 5, (0, 255, 0), -1)
        
        # Add frame number and position info
        cv2.putText(frame, f"Frame {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Pos: ({x:.1f}, {y:.1f}, {z:.1f})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add grid lines for visual reference
        for j in range(0, width, 50):
            cv2.line(frame, (j, 0), (j, height), (50, 50, 50), 1)
        for j in range(0, height, 50):
            cv2.line(frame, (0, j), (width, j), (50, 50, 50), 1)
        
        # Save frame
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), frame)
        
        if i % 10 == 0:
            print(f"Generated {i} frames...")
    
    print(f"Generated {num_frames} synthetic frames in {output_dir}")
    return output_dir

def try_extract_ros_bag(bag_path, output_dir):
    """
    Try different methods to extract images from a ROS bag file on Windows
    """
    print(f"Attempting to extract frames from {bag_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Try different approaches
    
    # 1. Try using bagpy if available
    try:
        print("Trying extraction with bagpy...")
        import bagpy
        from bagpy import bagreader
        
        bag = bagreader(bag_path)
        
        # Get the list of topics
        print("Available topics:")
        for idx, topic in enumerate(bag.topic_table["Topics"].tolist()):
            print(f"{idx+1}: {topic}")
        
        # Ask user which topic contains images
        topic_idx = int(input("Enter number of the topic with images: ")) - 1
        selected_topic = bag.topic_table["Topics"].tolist()[topic_idx]
        
        print(f"Extracting from topic: {selected_topic}")
        # Note: This will extract the data to a CSV file, 
        # but we would need additional steps to convert that to images
        print("bagpy extraction started (this might take a while)...")
        
        # Unfortunately, bagpy doesn't have a direct way to extract images
        print("bagpy doesn't directly support image extraction on Windows.")
        print("Trying alternative methods...")
        return False
        
    except ImportError:
        print("bagpy not available.")
    except Exception as e:
        print(f"Error with bagpy extraction: {e}")
    
    # 2. Try using pyrosbag if available
    try:
        print("Trying extraction with pyrosbag...")
        import pyrosbag
        
        # This would be the implementation if pyrosbag is available
        # Currently pyrosbag doesn't have good Windows support
        print("pyrosbag doesn't fully support Windows yet.")
        
    except ImportError:
        print("pyrosbag not available.")
    except Exception as e:
        print(f"Error with pyrosbag extraction: {e}")
    
    # 3. As a last resort, check if the user has WSL with ROS
    try:
        print("Checking if WSL with ROS is available...")
        result = subprocess.run(["wsl", "rosversion", "-d"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        if result.returncode == 0:
            print("ROS detected in WSL!")
            print("You can extract the bag file using WSL with the command:")
            wsl_output_dir = output_dir.replace('\\', '/')
            print(f"wsl rostopic echo -b {bag_path} -p /camera/image_raw > {wsl_output_dir}/images.csv")
            
            choice = input("Would you like to attempt extraction via WSL? (y/n): ")
            if choice.lower() == 'y':
                # This is a placeholder for WSL bag extraction
                # Implementation would depend on the WSL setup
                print("WSL extraction not implemented in this script.")
        else:
            print("ROS not detected in WSL or WSL not available.")
        
    except Exception as e:
        print(f"Error checking WSL: {e}")
    
    # If all extraction methods failed
    print("\nCould not extract images from the bag file using available tools.")
    print("Would you like to generate a synthetic dataset instead? (y/n)")
    
    choice = input().lower()
    if choice == 'y':
        num_frames = int(input("How many frames would you like to generate? (default: 100): ") or "100")
        return generate_synthetic_dataset(output_dir, num_frames)
    else:
        return False

def main():
    print("ROS Bag Image Extractor for Windows")
    print("===================================")
    
    # Check for bag files in the current directory
    bag_files = [file for file in os.listdir(".") if file.endswith(".bag")]
    
    if bag_files:
        print("\nFound bag files:")
        for i, bag_file in enumerate(bag_files):
            print(f"{i+1}. {bag_file}")
        
        try:
            bag_idx = int(input("\nSelect a bag file (enter number): ")) - 1
            
            if 0 <= bag_idx < len(bag_files):
                selected_bag = bag_files[bag_idx]
                output_dir = input("Enter output directory (default: 'extracted_frames'): ") or "extracted_frames"
                
                print(f"Will extract images from {selected_bag} to {output_dir}")
                
                result = try_extract_ros_bag(selected_bag, output_dir)
                
                if result:
                    print(f"Successfully extracted or generated frames in {result}")
                    print("You can now use these frames with the windows_slam_pipeline.py script")
                else:
                    print("Extraction failed.")
                    print("Consider generating synthetic data or manually extracting frames")
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")
    else:
        print("No bag files found in the current directory.")
        print("Would you like to generate a synthetic dataset for testing? (y/n)")
        
        choice = input().lower()
        if choice == 'y':
            output_dir = input("Enter output directory (default: 'synthetic_dataset'): ") or "synthetic_dataset"
            num_frames = int(input("How many frames would you like to generate? (default: 100): ") or "100")
            
            result = generate_synthetic_dataset(output_dir, num_frames)
            
            if result:
                print(f"Successfully generated synthetic frames in {result}")
                print("You can now use these frames with the windows_slam_pipeline.py script")

if __name__ == "__main__":
    main() 