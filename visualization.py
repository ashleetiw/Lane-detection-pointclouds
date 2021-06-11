#!/usr/bin/env python3
'''
Visualization code for pointcloud data and image data  
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

dataset_gray0 = imageio.imread("data_road_gray/training/image_0/um_000069.png")
dataset_gray1 =imageio.imread("data_road_gray/training/image_1/um_000069.png")
dataset_rgb = imageio.imread("data_road/training/image_2/um_000069.png")
dataset_velo = np.fromfile(str("data_road_velodyne/training/velodyne/um_000069.bin"), dtype=np.float32, count=-1).reshape([-1,4])

x = dataset_velo[:, 0]  # x position of point
y = dataset_velo[:, 1]  # y position of point
z = dataset_velo[:, 2]  # z position of point
r = dataset_velo[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor


# Draw camera data
f, ax = plt.subplots(2,2, figsize=(15, 5))
ax[0,0].imshow(dataset_gray0, cmap='gray')
ax[0,0].set_title('Left Gray Image (cam0)')
ax[0,1].imshow(dataset_gray1, cmap='gray')
ax[0,1].set_title('Right Gray Image (cam1)')
ax[1,0].imshow(dataset_rgb)
ax[1,0].set_title('RGB Image (cam2)')
ax[1, 1].scatter(x,y,c=r)
ax[1, 1].set_title('2d projection of data points')
plt.show()

# points_step = int(1. / points)
# point_size = 0.01 * (1. / points)
# velo_range = range(0, dataset_velo.shape[0], points_step)
# velo_frame = dataset_velo[velo_range, :]     
 
def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    ax.scatter(*np.transpose(dataset_velo [:, axes]), s=0.1, c=dataset_velo [:, 3])
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)
        
    # for t_rects, t_type in zip([frame], tracklet_types[frame]):
    #     draw_box(ax, t_rects, axes=axes, colotracklet_rectsr=colors[t_type])
        

# # Draw point cloud data as 3D plot
f2 = plt.figure(figsize=(15, 8))
ax2 = f2.add_subplot(111, projection='3d')                    
draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))
plt.show()

# Draw point cloud data as plane projections
f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
draw_point_cloud(
    ax3[0], 
    'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
    axes=[0, 2] # X and Z axes
)
draw_point_cloud(
    ax3[1], 
    'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
    axes=[0, 1] # X and Y axes
)
draw_point_cloud(
    ax3[2], 
    'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
    axes=[1, 2] # Y and Z axes
)
plt.show()







