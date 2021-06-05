#!/usr/bin/env python3

import imageio
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from lidar_lane_detection import Lidar,Image
from detect_lanes import Lane
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks


def remove_noise(data):
    """
    Fit the Data 
    """
    # X = [i for i in zip(['x'],p['y'])]
    X = StandardScaler().fit_transform(data[:,0:3])
    db = DBSCAN(eps=0.05, min_samples=10).fit(X)
    # # print(type(db))
    db_labels = db.labels_
    pc=pd.DataFrame()
    pc['x']=data[:,0]
    pc['y']=data[:,1]
    pc['z']=data[:,2]
    pc['Intensity']=data[:,3]
    pc['labels']=db_labels
    pc['r'] = np.sqrt(pc['x'] ** 2 + pc['y'] ** 2)

    #   remove noisy point clouds data
    labels, cluster_size = np.unique(pc['labels'], return_counts=True)
    # pc = pc[pc['labels']>=0] 

    return pc

def fit_polynomial(lanes,peaks):
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

        Y_NEW = model.predict(X_TRANSF)
        
        # plt.plot(X_TRANSF, Y_NEW, color='coral', linewidth=3)

        # print('for each lane min depth',lane[:,2].min())
        # z=np.ones((len(lane[:,2])))*lane[:,2].min()
        point1=np.concatenate((X_TRANSF.reshape(-1,1),Y_NEW.reshape(-1,1)),axis=1)
        point2=np.concatenate((lane[:,2].reshape(-1,1),lane[:,3].reshape(-1,1)),axis=1)

    
        newpoints=np.concatenate((point1,point2),axis=1)
        l.append(newpoints)
    
    return l



##################################################
import os


# filenames = os.listdir('data_road/training/image_2/')
# filename='data_road/training/image_2/umm_000011.png'

# for filename in filenames:
# filename=filename.replace('.png',"")

filename='umm_000011'

rgb = imageio.imread('data_road/training/image_2/'+ filename+'.png')


l=Lidar()
im=Image()
data,lidar=l.read_data("data_road_velodyne/training/velodyne/"+filename+".bin")
calib = im.read_calib_file('data_road/training/calib/'+filename+'.txt')

h, w, c = rgb.shape
print('before road plane',len(data))


data=data.to_numpy()
cloud,ind=l.find_road_plane(data)

data=remove_noise(cloud)

data=data.to_numpy()
print('after road plane and noise removal',len(data))


###############  fidinding the lanes ############

plt.figure()
lane=Lane()
yval,histVal=lane.peak_intensity_ratio(data,30)


peaks= find_peaks(histVal)[0]

for p in peaks:
    plt.plot(yval[p],histVal[p],'*r')


##################
x,y,z,index=im.render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,3])
fig,ax = plt.subplots(1)
plt.scatter(data[index,0],data[index,1])
plt.ylim(-20,25)
plt.xlim(0,50)


x=data[index,0]
min_x=math.ceil(x.min())
max_x=math.ceil(x.max())

nbin=max_x-min_x
x_val=np.linspace(min_x,max_x,nbin)

arr=[]
for p in peaks:
    arr.append(yval[p])

lanes =lane.DetectLanes(data[index,0:4],1,50,arr,min_x,max_x,len(peaks))

fitted_lane=fit_polynomial(lanes,peaks)


im.render_lanes_on_image(fitted_lane,rgb, calib, w,h,filename)

plt.show()








