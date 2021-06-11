#  Lane Detection on Small 3D Point Cloud Data
The goal of the project is to detect the lanes for a small LIDAR point clouds.
The approach used was detecting lanes using windows sliding search from  a multi-aspect airborne laser scanning point clouds which were recorded in a forward looking view.Since the resolution of the point cloud is low,deep learing approach or ML-unsueprvised learning will not work .Although Clustering has been used but for filtering out noise.

![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/demo.gif)

## Data visualization 
``2D visualization``
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/2d_original_img.png)

``3D visualization``
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/original_pcl.png)
Used The KITTI Vision road dataset to perform testing for lane detection 


## Algorithm
Lane detection in lidar involves detection of the immediate left and right lanes, also known as ego vehicle lanes, with respect to the lidar sensor. The flowchart gives an overview of the workflow 
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/flowdiagram.png)

## Preprocessing
`Class Lidar` contains methods to preprocess the lidar pointclouds 
### Region of interest extraction
-`remove_noise` function uses DBSCAN clustering to remove noise from pointclouds data
-` render_lidar_on_image` in class `Image` retains only pointlcouds which overlays only within the range (0,image_width) and (0,image_height)

![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/lidarpoints.png)

### Ground plane segmentation
Requirement `pip install python-pcl`
Simple plane segmentation of a set of points, that is to find all the points within a point cloud that support a plane model. I used the ransac algorithm to segment the ground plane lidar rings using python pcl library .
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/groundplane.png)

## Lane Point Detection
`Class Lane`  in detect_lanes.py contains methods to detect lane points from the pointcloud data 

### Peak intensity detection using histogram
Lane points in the lidar point cloud have a distinct distribution of intensities. Usually, these intensities occupy the upper region in the histogram distribution and appear as high peaks. Computing a histogram of intensities from the detected ground plane will give all the lanes points 
`peak_intensity_ratio` :Creates a histogram of intensity points. Control the number of bins for the histogram by specifying the bin resolution.
`find_peaks` :Obtains peaks in the histogram  

![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/intensity_histogram.png)

### lane detection
`DetectLanes`: peaks in the density of the  point clouds are used to detect the windows.Sliding Window approach is used to detect lanes from each window 
<!-- ![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/window_searchresult.png) -->

## Polynomial fitting 
### Parabolic Polynomial Fitting
The polynomial is fitted on X-Y points using a 2-degree polynomial represented as `ax2+bx+c`, where a, b, and c are polynomial parameters
To perform curve fitting, use the `fit_polynomial` function
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/scatter2dplot_fittedlanes.png)

## Image and lidar data visualization
`Class Image` performs tranformation from 3d pointlouds to 2d pixel points to project lidardata on top of rgb image 
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/img/lanelines.png)


## Demonstration
Lane detection is performed on data which was collected in realtime by Velodyne sensor mounted on top of a vehicle and its correspoding 2D plot shows the variation in density of point clouds with lanes detected 
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/output.gif)
![Demonstration](https://github.com/ashleetiw/Lane-detection-pointclouds/blob/main/output2d.gif)

