# Visual SLAM for Robotics

## Overview
This project implements a Visual SLAM (Simultaneous Localization and Mapping) system that enables robots or autonomous vehicles to map their surroundings while determining their location in real-time using camera inputs.

## Features
- Feature extraction and matching using ORB algorithm
- Camera pose estimation using epipolar geometry
- 3D point cloud generation via triangulation
- Trajectory visualization
- Support for ROS bag files and video files

## Requirements
- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- Open3D (for 3D visualization)
- ROS (optional, for bag file processing)

## Usage
1. Place your bag files or video files in the project directory
2. Run the main script:
```
python windows_slam_pipeline.py
```
3. Follow the on-screen prompts to select input files and parameters

## Sample Datasets
The system has been tested with:
- corridor4 dataset
- magistrale5 dataset
- outdoors4 dataset
- room4 dataset

## Output
- Trajectory visualization (trajectory.png)
- 3D point cloud (point_cloud.ply)
- Feature matching frames (output_frames directory)

## Advanced Usage

You can modify the `VisualSLAM` class in `slam_pipeline.py` to:
- Adjust feature detection parameters
- Change camera intrinsic parameters
- Implement loop closure detection
- Add additional visualization options

## Troubleshooting

- **Error importing rosbag**: Make sure your Python environment has access to ROS packages
- **Slow processing**: Reduce the number of frames or adjust feature detection parameters
- **Inaccurate tracking**: Fine-tune the camera intrinsic parameters for your specific dataset

## References

- ORB feature detector: [OpenCV docs](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html)
- Visual SLAM tutorial: [OpenCV SLAM tutorial](https://docs.opencv.org/master/d9/db7/tutorial_py_table_of_contents_calib3d.html)
- TUM RGB-D datasets: [TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) 