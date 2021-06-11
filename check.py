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


    return imgfov_pc_pixel[0], imgfov_pc_pixel[1],pts_velo[inds,2],inds

def render_lanes_on_image(data,img, calib, img_width, img_height):
    proj_velo2cam2 = project_velo_to_cam2(calib)
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')


    for d in data:
        pts_2d = project_to_image(d.transpose(), proj_velo2cam2)
        inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) & (d[:, 0] > 0)
                     )[0]

        # Filter out pixels points
        imgfov_pc_pixel = pts_2d[:, inds]

        # Retrieve depth from lidar
        imgfov_pc_velo = d[inds, :]
        # imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
        imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
        # Create a figure. Equal aspect so circles look circular  
        # Show the image
        ax.imshow(img)
        # if len(imgfov_pc_pixel[0])>0:
        #     print(imgfov_pc_pixel[0].min(),imgfov_pc_pixel[0].max())
        #     x_ext = np.linspace(imgfov_pc_pixel[0].min(), imgfov_pc_pixel[0].max(), 100)
        #     p = np.polyfit(imgfov_pc_pixel[0],imgfov_pc_pixel[1] , deg=2)
        #     y_ext = np.poly1d(p)(x_ext)
        ax.plot(imgfov_pc_pixel[0],imgfov_pc_pixel[1],color='coral', linewidth=3)
        plt.savefig('result/um_000000.png')
       
    return imgfov_pc_pixel[0], imgfov_pc_pixel[1]


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
    avg_intensity=[]
    ymean=[]
    for i in range(len(y_val)-1):
        index=get_index_inrange(y,y_val[i],y_val[i+1])
        intensity_sum=0
        for j in index:
            intensity_sum+=ptCloud[j,3]

        avg_intensity.append(intensity_sum)
        ymean.append((y_val[i]+y_val[i+1])/2)
    
    plt.plot(ymean,avg_intensity,'--k')
    return ymean,avg_intensity

# def findpeaks(hist):


#     hist.append(100000000)
#     #  pad the array with nan
#     hist=[100000000]+histVal
#     # get difference between two adjacent hist values with sign 
#     xdiff = np.diff(hist)
#     # Take the sign of the first sample derivative
#     s=np.sign(xdiff)
#     # Find local maxima
#     targetarr=np.diff(s)
#     peaks=[i for i in range(len(targetarr)) if targetarr[i]<0]
#     p1=peaks[0]
#     p2=peaks[1]
#     # print(yval[p1],yval[p2])

#     plt.plot((yval[p1],yval[p2]),(histVal[p1],histVal[p2]),'*r')
#     return peaks


def find_peaks(a):
    x = np.array(a)
    max = np.max(x)
    lenght = len(a)
    ret = []
    for i in range(lenght):
        ispeak = True
        if i-1 > 0:
            ispeak &= (x[i] > 1.8 * x[i-1])
        if i+1 < lenght:
            ispeak &= (x[i] > 1.8 * x[i+1])

        ispeak &= (x[i] > 0.05 * max)
        if ispeak:
            ret.append(i)
    return ret


def DisplayBins(x_val,y,color):
    y_val=[y]*len(x_val)
    plt.plot(x_val,y_val,c=color)


def DetectLanes(data,hbin,vbin, start,min_x,max_x,num_lanes):
    verticalBins = np.zeros((vbin, 4, num_lanes))
    lanes = np.zeros((vbin, 4, num_lanes))
    # verticalBins=[]
    # lanes=[]
    laneStartX = np.linspace(min_x,max_x, vbin)
    
    startLanePoints=start

    for i in range(vbin-1):
        # print('for index i',i)
        # print('after each updation',startLanePoints)
        for j in range(num_lanes):
            laneStartY = startLanePoints[j]
            # print('starting x',laneStartX [i],laneStartX[i+1])

            # roi=[laneStartX[i], laneStartX[i+1], laneStartY - hbin/2, laneStartY + hbin/2, -math.inf, math.inf]
    

            lowerbound=math.ceil(laneStartY - hbin)
            upperbound=math.ceil(laneStartY + hbin)
            # print('range y', lowerbound,upperbound)

            # print('before ',len(data))


            inds = np.where((data[:,0] < laneStartX[i+1] )& (data[:,0] >= laneStartX[i]) &
                    (data[:,1] < upperbound) & (data[:,1] >= lowerbound))[0]
            
            # print(len(inds))
            plt.scatter(data[inds,0],data[inds,1],c='yellow')

            if len(inds)!=0:
                # plt.vlines(laneStartX[i],-15,15)
                roi_data=data[inds,:]
                max_intensity=np.argmax(roi_data[:,3].max())
                
                val=roi_data[max_intensity,:]
        
                verticalBins[i,:,j]=val
                lanes[i,:,j]=val
                startLanePoints[j]=roi_data[max_intensity,1]

                plt.scatter(roi_data[max_intensity,0],roi_data[max_intensity,1],s=50,c='hotpink')
            
            # else:
            #     print('no points',laneStartX[i])
            #     value =lanes[:, 0:2, j]
            
                
                
            #     print(' inside the else loop ')
            #     print(lanes[1:,0:2,j])
                # print(value)

    return lanes
        

def ransac_polyfit(x, y, order=2, n=20, k=100, t=0.1, d=100, f=0.8):
    # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus
    
    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required
  
    besterr = np.inf
    bestfit = None
    for kk in range(len(x)):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers])-y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit




    ###############################################################



rgb = imageio.imread("data_road/training/image_2/um_000000.png")
data,lidar=read_data("data_road_velodyne/training/velodyne/um_000000.bin")
calib = read_calib_file('data_road/training/calib/um_000000.txt')

h, w, c = rgb.shape
print('before road plane',len(lidar))

p=pd.DataFrame()
p['x']=lidar[:,0]
p['y']=lidar[:,1]
p['z']=lidar[:,2]
p['Intensity']=lidar[:,3]

p=p.to_numpy()
cloud,ind=find_road_plane(p)


"""
    Fit the Data 
"""
# X = [i for i in zip(['x'],p['y'])]
X = StandardScaler().fit_transform(cloud[:,0:3])

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

#   remove noisy point clouds data
labels, cluster_size = np.unique(data['labels'], return_counts=True)
# data = data[data['labels']>=0] 

data=data.to_numpy()

plt.figure()
yval,histVal=peak_intensity_ratio(data,30)

from scipy.signal import find_peaks
peaks= find_peaks(histVal)[0]


for p in peaks:
    plt.plot(yval[p],histVal[p],'*r')

################# my peak function ##################
# p1=np.argmax(histVal)
# temp=histVal[:p1]+histVal[p1+1:]
# max_val=max(temp)
# p2=temp.index(max_val)
# print( ' max y value',yval[p1])
# print( ' second y value',yval[p2])

# peaks=[p1,p2]
# yval_hist=[yval[p1],yval[p2]]
# plt.plot((yval[p1],yval[p2]),(histVal[p1],histVal[p2]),'*r')
# print( ' the peaks indexes are ',peaks)

# ################# my peak function ##################
print('original datashape',data[:,0:4].shape)
x,y,z,index=render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,3])

fig,ax = plt.subplots(1)
plt.scatter(data[index,0],data[index,1])

plt.ylim(-20,25)
plt.xlim(0,50)

x=data[index,0]
min_x=math.ceil(x.min())
max_x=math.ceil(x.max())

nbin=max_x-min_x
x_val=np.linspace(min_x,max_x,nbin)


# vbin=math.ceil( data[index,:].max()-data[index,:].min() )
arr=[]
for p in peaks:
    arr.append(yval[p])

# print(min_x,max_x,nbin)
lanes =DetectLanes(data[index,0:4],1,50,arr,min_x,max_x,len(peaks))
print('lane shape',lanes.shape)

from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score

polynomial_features = PolynomialFeatures(degree = 2)

l=[]
for i in range(len(peaks)):
    # print(lanes[:,:,i].shape)
    lane=lanes[:,:,i]
    # lane=[lane[i] for i in range(len(lane)) if lane[i,0]!=0 and lane[i,1]!=0]
    lane=np.array(lane)
    print('each lane shape',lane.shape)
    X_TRANSF=np.reshape(lane[:,0],(len(lane),1))
    
    y=np.reshape(lane[:,1],(len(lane),1))
    model = LinearRegression()
    model.fit(X_TRANSF,y)

    # Step 4: calculate bias and variance

    Y_NEW = model.predict(X_TRANSF)

    # rmse = np.sqrt(mean_squared_error(y,Y_NEW))
    # r2 = r2_score(y,Y_NEW)

    # print('RMSE: ', rmse)
    # print('R2: ', r2)

    Y_NEW = model.predict(X_TRANSF)

    # plt.plot(X_TRANSF, Y_NEW, color='coral', linewidth=3)

    # print(len(y),len(X_TRANSF),len(Y_NEW))

    # point1=np.hstack((lane[:,0],lane[:,1]))
    # point2=np.hstack((lane[:,2],lane[:,3]))

    print('for each lane min depth',lane[:,2].min())
    z=np.ones((len(lane[:,2])))*lane[:,2].min()
    point1=np.concatenate((X_TRANSF.reshape(-1,1),Y_NEW.reshape(-1,1)),axis=1)
    point2=np.concatenate((z.reshape(-1,1),lane[:,3].reshape(-1,1)),axis=1)

   
    newpoints=np.concatenate((point1,point2),axis=1)
    l.append(newpoints)
    # print('line shape',point1.shape)
    # print('line shape',point2.shape)
    # print('line shape',newpoints.shape)
    # break


render_lanes_on_image(l,rgb, calib, w,h)

plt.show() 




