#!/usr/bin/env python3
import imageio
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import norm
from numpy.polynomial import polynomial as P
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from matplotlib.patches import Circle
from sklearn import svm
import pyransac3d as pyrsc
from mpl_toolkits.mplot3d import Axes3D
import pcl
import struct
from open3d import *
from sklearn import linear_model


def read_data(data_path):
        pointcloud = np.fromfile(str(data_path), dtype=np.float32, count=-1).reshape([-1,4])
        x = pointcloud[:, 0]  # x position of point
        y = pointcloud[:, 1]  # y position of point
        z = pointcloud[:, 2]  # z position of point
        I = pointcloud[:, 3]  # reflectance value of point
        d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
        
        data=pd.DataFrame()
        data['x']=x
        data['y']=y
        data['z']=z
        data['Intensity']=I
        return data,pointcloud

def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def mean_filter(pc):

        mean = pc["Intensity"].mean()
        std = pc["Intensity"].std()

        meanz = pc["z"].min()
        stdz = pc["z"].std()

        filtered_lanes = pc[pc["Intensity"] > mean - 1 * std]
        #  remove z as well
        filtered_lanes = filtered_lanes[filtered_lanes["z"] > meanz + stdz ]
        
        return filtered_lanes

def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    # points = np.vstack((poin)))
    # print(points.shape[0], points.shape[1])
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]

def render_lidar_on_image(pts_velo, img, calib, img_width, img_height,label):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    # imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    # Create a figure. Equal aspect so circles look circular
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(img)
    
    ax.scatter(imgfov_pc_pixel[0], imgfov_pc_pixel[1],s=3,c=label[inds])
    # print(len(label[inds]))

    plt.yticks([])
    plt.xticks([])


    # plt.show()
    # print(label)
    return imgfov_pc_pixel[0], imgfov_pc_pixel[1],pts_velo[inds,2],pts_velo[inds,3]


def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = pcl.PointCloud()
    pcd = open3d.utility.Vector3dVector(np_pcd)
    return pcd


# running RANSAC Algo
def find_road_plane(points):

        
        # XY = points[:, :2]
        # Z =  points[:, 2]
        # ransac = linear_model.RANSACRegressor(residual_threshold=0.01)
        # ransac.fit(XY, Z)
        # inlier_mask = ransac.inlier_mask_
        # outlier_mask = np.logical_not(inlier_mask)
        # inliers = np.zeros(shape=(len(inlier_mask), 4))
        # outliers = np.zeros(shape=(len(outlier_mask), 4))
        # a, b = ransac.estimator_.coef_
        # d = ransac.estimator_.intercept_
        # for i in range(len(inlier_mask)):
        #     if not outlier_mask[i]:
        #         inliers[i] = points[i]
        #     else:
        #         outliers[i] = points[i]

        # return a,b,d,inliers,outliers


    cloud = pcl.PointCloud_PointXYZI()
    cloud.from_array(points.astype('float32'))
    
     #  create a pcl object 
    seg = cloud.make_segmenter()

    seg.set_optimize_coefficients(True)

    seg.set_model_type(pcl.SACMODEL_PLANE)

    seg.set_method_type(pcl.SAC_RANSAC)

    seg.set_distance_threshold(0.5)

    indices, model = seg.segment()

    cloud_plane = cloud.extract(indices, negative=False)
    return cloud_plane.to_array(), np.array(indices)


# def peak_intensity_ratio(data,bin):
    
#     max_dist=int(data['r'].max())
#     counts, bins = np.histogram(data['Intensity'])
#     # plot histogram centered on values 0..255
#     plt.bar(bins[:-1] - 0.2, counts, width=1, edgecolor='red')
#     # plt.xlim([-0.5, max_dist])
#     plt.show()
#     # plt.style.use('seaborn-white')


rgb = imageio.imread("data_road/training/image_2/um_000051.png")
data,lidar=read_data("data_road_velodyne/training/velodyne/um_000051.bin")
calib = read_calib_file('data_road/training/calib/um_000051.txt')

h, w, c = rgb.shape
print('before road plane',len(lidar))
cloud,ind=find_road_plane(lidar[:,0:4])

print(np.unique(cloud[:,2]))

p=pd.DataFrame()
p['x']=cloud[:,0]
p['y']=cloud[:,1]
p['z']=cloud[:,2]
p['Intensity']=cloud[:,3]
p['r'] = np.sqrt(p['x'] ** 2 + p['y'] ** 2)
histBinResolution=0.2
# peak_intensity_ratio(p,histBinResolution)

#  adjusting z 
p['z']=p['z']-p['z'].min()
# # p = p[p["z"]< p['z'].mean()] 
# print(p['z'].max())
# print(p['z'].min())
# print(p['z'].mean())


# p=mean_filter(p)
# print(p['Intensity'].max())
# print(p['Intensity'].min())
# # print(p['Intensity'].mean())

p=p.to_numpy()

"""
    Fit the Data 
"""
# X = [i for i in zip(['x'],p['y'])]
X = StandardScaler().fit_transform(p[:,0:2])

db = DBSCAN(eps=0.5, min_samples=100).fit(X)
# print(type(db))
db_labels = db.labels_

data=pd.DataFrame()
data['x']=p[:,0]
data['y']=p[:,1]
data['z']=p[:,2]
data['Intensity']=p[:,3]
data['labels']=db_labels


#   remove noisy point clouds data

labels, cluster_size = np.unique(data['labels'], return_counts=True)
data = data[data['labels']>=0] 

# retain the largest cluster
max_label=labels[np.argmax(cluster_size)]
data = data[data['labels']==max_label] 
print(max_label,max(cluster_size))

data=data.to_numpy()
x,y,z,intensity=render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,4])
plt.show()



# p['Intensity']=lidar[:,3]
# # #################################################    modelling plane  ################################
# fig = plt.figure("Pointcloud")
# ax = Axes3D(fig)
# ax.grid = True

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # road_plane, road_plane_idx = find_road_plane(lidar)
# min_x = np.amin(inliers[:, 0])
# max_x = np.amax(inliers[:, 0])
# min_y = np.amin(inliers[:, 1])
# max_y = np.amax(inliers[:, 1])

# x = np.linspace(min_x, max_x)
# y = np.linspace(min_y, max_y)

# X, Y = np.meshgrid(x, y)
# Z = a * X + b * Y + d
# AA = ax.plot_surface(X, Y, Z, cmap='binary', rstride=1, cstride=1, 
# alpha=1.0)
# # BB = ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],c='red', s 
# # =1)
# CC = ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c='green', 
# s=1)


# road_plane_flatten = road_plane[:,0:2]
# db = DBSCAN(eps=0.1, min_samples=100).fit_predict(road_plane_flatten)


# p=pd.DataFrame()
# p['x']=lidar[:,0]
# p['y']=lidar[:,1]
# p['z']=lidar[:,2]
# p['Intensity']=lidar[:,3]
# p['label']=db

# # p['x']=road_plane[:,0]
# # p['y']=road_plane[:,1]
# # p['z']=road_plane[:,2]
# # p['Intensity']=road_plane[:,3]
# # p['label']=db

# p=p.to_numpy()
# x1,x2,z,y=render_lidar_on_image(p[:,0:4], rgb, calib, w, h,p[:,4])
# plt.show()
