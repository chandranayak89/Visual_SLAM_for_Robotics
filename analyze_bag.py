#!/usr/bin/env python3

import rosbag
import sys
import os
import argparse
from collections import defaultdict

def analyze_bag(bag_path):
    """
    Analyze a ROS bag file and display information about its contents
    
    Args:
        bag_path (str): Path to the bag file
    """
    if not os.path.exists(bag_path):
        print(f"Error: Bag file {bag_path} does not exist!")
        return
    
    print(f"\nAnalyzing bag file: {bag_path}")
    print("-" * 50)
    
    try:
        bag = rosbag.Bag(bag_path)
        
        # Get basic info
        info = bag.get_type_and_topic_info()
        topics = info[1]
        
        # Duration
        start_time = None
        end_time = None
        
        for _, msg, t in bag.read_messages():
            if start_time is None or t.to_sec() < start_time:
                start_time = t.to_sec()
            if end_time is None or t.to_sec() > end_time:
                end_time = t.to_sec()
        
        # Print general info
        print(f"Duration: {end_time - start_time:.2f} seconds")
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")
        print(f"Size: {os.path.getsize(bag_path) / (1024 * 1024):.2f} MB")
        print(f"Topics: {len(topics)}")
        
        # Print topic info
        print("\nTopic Information:")
        print("-" * 50)
        print(f"{'Topic':<40} {'Type':<30} {'Messages':<10} {'Frequency':<10}")
        print("-" * 50)
        
        for topic_name, topic_info in topics.items():
            msg_count = topic_info.message_count
            if msg_count > 0 and end_time > start_time:
                frequency = msg_count / (end_time - start_time)
            else:
                frequency = 0
            
            print(f"{topic_name:<40} {topic_info.msg_type:<30} {msg_count:<10} {frequency:.2f} Hz")
        
        # Sample messages for each topic
        print("\nSample Messages:")
        print("-" * 50)
        
        topic_messages = defaultdict(list)
        for topic, msg, _ in bag.read_messages():
            if len(topic_messages[topic]) < 1:  # Only store one sample message per topic
                topic_messages[topic].append(msg)
        
        for topic, messages in topic_messages.items():
            if len(messages) > 0:
                print(f"\nTopic: {topic}")
                try:
                    # For image messages, print dimensions
                    if topics[topic].msg_type == 'sensor_msgs/Image':
                        print(f"  Image dimensions: {messages[0].width} x {messages[0].height}")
                        print(f"  Encoding: {messages[0].encoding}")
                        print(f"  Is bigendian: {messages[0].is_bigendian}")
                        print(f"  Step: {messages[0].step}")
                    else:
                        # For other messages, just print the first few fields
                        print("  Sample attributes:")
                        msg_str = str(messages[0])
                        msg_lines = msg_str.split('\n')
                        for i, line in enumerate(msg_lines[:5]):  # Print first 5 lines
                            print(f"    {line}")
                        if len(msg_lines) > 5:
                            print("    ...")
                except Exception as e:
                    print(f"  Error analyzing message: {e}")
                
        bag.close()
        
    except Exception as e:
        print(f"Error analyzing bag file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze ROS bag files")
    parser.add_argument("bag_file", help="Path to the ROS bag file to analyze")
    
    if len(sys.argv) == 1:
        # No arguments, scan for bag files in current directory
        bag_files = [f for f in os.listdir('.') if f.endswith('.bag')]
        
        if not bag_files:
            print("No bag files found in current directory!")
            parser.print_help()
            return
        
        print("Available bag files:")
        for i, bag in enumerate(bag_files):
            print(f"{i+1}. {bag}")
        
        try:
            selected = int(input("\nSelect a bag file to analyze (enter number): ")) - 1
            if 0 <= selected < len(bag_files):
                analyze_bag(bag_files[selected])
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")
    else:
        args = parser.parse_args()
        analyze_bag(args.bag_file)

if __name__ == "__main__":
    main() 