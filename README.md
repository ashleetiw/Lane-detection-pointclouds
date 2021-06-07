# UNSUPERVISED LANE DETECTION 
This example shows how to detect lanes in lidar point cloud.Lidar lane detection enables you to build complex workflows like lane keep assist, lane departure warning, and adaptive cruise control for autonomous driving. 

**DATASET VISUALIZATION** 
Used The KITTI Vision road dataset for lane detection 

``2D visualization``
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/2d_visualization.png)

``3D visualization``
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/3d_visualization.png)


**ALGORITHM** 
Lane detection in lidar involves detection of lanes with respect to the lidar sensor. It involves the following steps:

## Preprocessing
`Class Lidar` contains methods to preprocess the lidar pointclouds 
### Region of interest extraction
-`remove_noise` function uses DBSCAN clustering to remove noise from pointclouds data
-` render_lidar_on_image` in class `Image` retains only pointlcouds which overlays only within the range (0,image_width) and (*0,image_height)

### Ground plane segmentation
Requirement `pip install python-pcl`
simple plane segmentation of a set of points, that is to find all the points within a point cloud that support a plane model. 
I am using the ransac algorithm to segment the ground plane lidar rings using python pcl library .
* Note: In order to retain a dense pointclouds retaning pointclouds in range (z-epsilon,z+epsilon) where epsilon is a tunable parameter *

## Lane Point Detection
`Class Lane`  in detect_lanes.py contains methods to detect lane points from the pointcloud data 

### Peak intensity detection using histogram
Lane points in the lidar point cloud have a distinct distribution of intensities. Usually, these intensities occupy the upper region in the histogram distribution and appear as high peaks. Computing a histogram of intensities from the detected ground plane will give all the lanes points 
`peak_intensity_ratio` :Creates a histogram of intensity points. Control the number of bins for the histogram by specifying the bin resolution.
`find_peaks` :Obtains peaks in the histogram  

### lane detection
`DetectLanes`:detects lane datapoints where the initial estimates for the sliding windows are made using an intensity-based histogram.

## Polynomial fitting 
**Parabolic Polynomial Fitting**
The polynomial is fitted on X-Y points using a 2-degree polynomial represented as `ax2+bx+c`, where a, b, and c are polynomial parameters
To perform curve fitting, use the `fit_polynomial` function

## Image and lidar data visualization
`Class Image` performs tranformation from 3d pointlouds to 2d pixel points to project lidardata on top of rgb image 

**DEMONSTRATION** 

