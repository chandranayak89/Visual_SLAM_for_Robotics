#!/usr/bin/env python3
import rosbag
from cv_bridge import CvBridge
import cv2
import os

# Path to your bag file - update this to match your setup
bag_path = '/mnt/hgfs/Visual_SLAM_for_Robotics/dataset-corridor4_512_16.bag'  

# Image topic found in your bag file
image_topic = '/cam0/image_raw'

# Output directory - using your shared folder
output_dir = '/mnt/hgfs/Visual_SLAM_for_Robotics/extracted_frames'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bag = rosbag.Bag(bag_path)
bridge = CvBridge()
count = 0

for topic, msg, t in bag.read_messages(topics=[image_topic]):
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    # Save timestamp information (optional but useful for SLAM)
    timestamp = t.to_sec()
    
    # Save image with timestamp in filename
    cv2.imwrite(f'{output_dir}/frame_{count:06d}_{timestamp:.6f}.jpg', cv_img)
    count += 1
    if count % 100 == 0:
        print(f'Extracted {count} frames')

bag.close()
print(f'Extracted {count} frames total')

# Force write buffers to be flushed to disk
os.system('sync')
print("Files synchronized to disk")  #!/usr/bin/env python3

import os
import rosbag
from cv_bridge import CvBridge
import cv2
import argparse

def extract_frames(bag_path, output_dir="extracted_frames", image_topic="/camera/image_raw", max_frames=None):
    """
    Extract frames from a ROS bag file and save them as images
    
    Args:
        bag_path (str): Path to the bag file
        output_dir (str): Directory to save extracted frames
        image_topic (str): ROS topic containing image messages
        max_frames (int, optional): Maximum number of frames to extract. None for all frames.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Extracting frames from {bag_path}")
    print(f"Saving to {output_dir}")
    
    # Create bridge for converting ROS image messages to OpenCV images
    bridge = CvBridge()
    
    # Open the bag file
    bag = rosbag.Bag(bag_path)
    count = 0
    
    # Get total number of messages
    total_msgs = bag.get_message_count(topic_filters=[image_topic])
    print(f"Total messages on topic {image_topic}: {total_msgs}")
    
    # Extract frames
    try:
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            try:
                # Convert ROS image to OpenCV image
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                
                # Save the image
                filename = os.path.join(output_dir, f"frame_{count:06d}.jpg")
                cv2.imwrite(filename, cv_img)
                
                count += 1
                if count % 50 == 0:
                    print(f"Extracted {count} frames")
                
                if max_frames is not None and count >= max_frames:
                    print(f"Reached maximum number of frames ({max_frames})")
                    break
                    
            except Exception as e:
                print(f"Error extracting frame {count}: {e}")
                
    except Exception as e:
        print(f"Error reading bag file: {e}")
    finally:
        bag.close()
        
    print(f"Extracted {count} frames from {bag_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from ROS bag files")
    parser.add_argument("bag_file", help="Path to the ROS bag file")
    parser.add_argument("--output", "-o", default="extracted_frames", 
                        help="Output directory for frames (default: extracted_frames)")
    parser.add_argument("--topic", "-t", default="/camera/image_raw", 
                        help="ROS topic containing image messages (default: /camera/image_raw)")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="Maximum number of frames to extract (default: all)")
    
    args = parser.parse_args()
    
    # Check if the bag file exists
    if not os.path.exists(args.bag_file):
        print(f"Error: Bag file {args.bag_file} not found")
        return
    
    extract_frames(args.bag_file, args.output, args.topic, args.max_frames)

if __name__ == "__main__":
    main() 
