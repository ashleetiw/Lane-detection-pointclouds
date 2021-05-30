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
import math 

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
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0)&
                    (pts_velo[:, 0] > 0)
                     )[0]

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
    # ax.label()
    # print(len(label[inds]))
    plt.yticks([])
    plt.xticks([])

    # plt.show()
    # X,Y,Z,INTENSITY

    # return imgfov_pc_pixel[0], imgfov_pc_pixel[1],pts_velo[inds,2],pts_velo[inds,3],pts_velo[inds,4]
    return imgfov_pc_pixel[0], imgfov_pc_pixel[1],pts_velo[inds,2],inds


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


    fil = cloud.make_passthrough_filter()
    fil.set_filter_field_name("z")
    # print(points[:,2].min(),points[:,2].mean(),points[:,2].max())
    fil.set_filter_limits(points[:,2].min(),0)
    cloud_filtered = fil.filter()

    
     #  create a pcl object 
    seg =  cloud_filtered.make_segmenter()

    seg.set_optimize_coefficients(True)

    seg.set_model_type(pcl.SACMODEL_PLANE)

    seg.set_method_type(pcl.SAC_RANSAC)

    # seg.set_max_iterations(100)

    seg.set_distance_threshold(0.8)

    indices, model = seg.segment()

    cloud_plane = cloud.extract(indices, negative=False)
    return cloud_plane.to_array(), np.array(indices)


def get_index_inrange(arr,start,end):
    ind=[i for i in range(len(arr)) if arr[i]>=start and arr[i]<end]
    return ind

def peak_intensity_ratio(ptCloud,bin_size):  
    
    y=ptCloud[:,1]
    min_y=math.ceil(y.min())
    max_y=math.ceil(y.max())

    y_val=np.linspace(min_y,max_y,bin_size)
    print(y_val)

    avg_intensity=[]
    ymean=[]
    for i in range(len(y_val)-1):

        index=get_index_inrange(y,y_val[i],y_val[i+1])
        # print(len(index))

        # summing up the intesity 
        intensity_sum=0
        for j in index:
            intensity_sum+=data[j,3]

        avg_intensity.append(intensity_sum)
        ymean.append((y_val[i]+y_val[i+1])/2)
    

    plt.plot(ymean,avg_intensity,'--k')
    return ymean,avg_intensity
    
    # plt.show()


def findpeaks(hist):


    hist.append(100000000)
    #  pad the array with nan
    hist=[100000000]+histVal

    # get difference between two adjacent hist values with sign 
    xdiff = np.diff(hist)

    # Take the sign of the first sample derivative
    s=np.sign(xdiff)

    # print(s)
    # print(xdiff)
    # print(hist)


    # Find local maxima
    targetarr=np.diff(s)
    peaks=[i for i in range(len(targetarr)) if targetarr[i]<0]

    p1=peaks[0]
    p2=peaks[1]

    # print(yval[p1],yval[p2])

    plt.plot((yval[p1],yval[p2]),(histVal[p1],histVal[p2]),'*r')
    
    return peaks


def window_initiaize(y,hist,peaks):
    left_lane_index=[ i for i in range(len(y)) if y[i]<0]
    right_lane_index=[ i for i in range(len(y)) if y[i]>=0]
    

    y_leftLane=[]
    y_rightLane=[]
    for ind  in left_lane_index:
         y_leftLane.append(y[ind])
    for ind  in right_lane_index:
         y_rightLane.append(y[ind])
    
    laneWidth=8
    

    # print('original yvals',y)
    # print(left_lane_index)     
    # print(y_rightLane,y_leftLane)

    diff = np.zeros([len(left_lane_index),len(right_lane_index)])
    for i in range(len(left_lane_index)):
        for j in range(len(right_lane_index)):
            diff[i][j] = abs(laneWidth - (y_leftLane[i] - y_rightLane[j]))
    

    arr=np.argwhere(diff == np.min(diff))
    # print(diff)
    # print(arr)
    row=arr[0][0]
    col=arr[0][1]

    yval = [y_leftLane[row], y_leftLane[col]]
    estimatedLaneWidth =y_leftLane[row]- y_leftLane[col]
    print(estimatedLaneWidth)

    return yval


def DisplayBins(x_val,y,color):
    y_val=[y]*len(x_val)

    # print(len(x_val))
    # print(len(y_val))

    plt.plot(x_val,y_val,c=color)



rgb = imageio.imread("data_road/training/image_2/uu_000069.png")
data,lidar=read_data("data_road_velodyne/training/velodyne/uu_000069.bin")
calib = read_calib_file('data_road/training/calib/uu_000069.txt')

h, w, c = rgb.shape
print('before road plane',len(lidar))

p=pd.DataFrame()
p['x']=lidar[:,0]
p['y']=lidar[:,1]
p['z']=lidar[:,2]
p['Intensity']=lidar[:,3]



p=p.to_numpy()
cloud,ind=find_road_plane(p)

print('after road plane',len(cloud))
# # print(np.unique(cloud[:,2]))

#  make ground plane as 0 z 
# cloud[:,2]=cloud[:,2]-cloud[:,2].max()

# print(cloud[:,2].min(),cloud[:,2].max())

# x,y,z,intensity,labels=render_lidar_on_image(cloud[:,0:4],rgb, calib, w,h,cloud[:,2])

# plt.savefig('test4.png')
# plt.show()



"""
    Fit the Data 
"""
# X = [i for i in zip(['x'],p['y'])]
X = StandardScaler().fit_transform(cloud[:,0:3])

# c=np.array(p[:,0:3])
# print(c.shape[0],c.shape[1])

db = DBSCAN(eps=0.05, min_samples=10).fit(X)
# # print(type(db))
db_labels = db.labels_

data=pd.DataFrame()
data['x']=cloud[:,0]
data['y']=cloud[:,1]
data['z']=cloud[:,2]
data['Intensity']=cloud[:,3]
data['labels']=db_labels
data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2)
# print(data.shape[0],data.shape[1])

#   remove noisy point clouds data
labels, cluster_size = np.unique(data['labels'], return_counts=True)
data = data[data['labels']>=0] 



# plt.figure()
# plt.scatter(data['x'],data['y'],c=data['r'])


# # retain the largest cluster
# max_label=labels[np.argmax(cluster_size)]
# data = data[data['labels']==max_label] 
# print(max_label,max(cluster_size))


    # break
#     # if data[i,3]>=data[:,3].mean():
#     #     c.append('red')
#     # else:
#         c.append('green')


# c=np.asarray(c)

data=data.to_numpy()

# x,y,z,index=render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,5])

# # histBinResolution=3
plt.figure()
yval,histVal=peak_intensity_ratio(data,20)
peaks= findpeaks(histVal) 
p1=peaks[0]
p2=peaks[1]
yval_hist=[yval[p1],yval[p2]]

yval_estimated=window_initiaize(yval,histVal,peaks)

print(yval_hist)
print(yval_estimated)

x,y,z,index=render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,3])

# # # print(yval)

plt.figure()
plt.scatter(data[index,0],data[index,1],c=data[index,3])


x=data[:,0]
min_x=math.ceil(x.min())
max_x=math.ceil(x.max())

nbin=max_x-min_x

# print(nbin)
x_val=np.linspace(min_x,max_x,nbin)

DisplayBins(x_val,yval[p1],'red')
DisplayBins(x_val,yval[p2],'green')
# plt.plot() 

# # # # # x,y,z,index=render_lidar_on_image(data[peak_indexes,0:4],rgb, calib, w,h,data[peak_indexes,4])
plt.show()





