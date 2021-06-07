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
from convert_lidar_to_paranomicview import overlay
import numpy.polynomial.polynomial as poly



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
        # lane=[lane[i,:] for i in range(len(lane)) if len(lane[i])>0]
        
        # X_TRANSF=np.reshape(lane[:,0],(-1,1))
        # y=np.reshape(lane[:,1],(-1,1))
        lane=[lane[i,:] for i in range(len(lane)) if lane[i,0]!=0 and lane[i,1]!=0]
        lane=np.array(lane).reshape(-1,4)
        x=lane[:,0]
        y=lane[:,1]
       
        if len(x)>0 and len(y)>0:
            coefs =  poly.polyfit(x,y,2)
            X_NEW=np.linspace(x.min()+5,x.max(),70)
            # X_NEW=x
            ffit = poly.Polynomial(coefs)
            Y_NEW=ffit(X_NEW)
            # print('for each lane min depth',lane[:,2].min())
            z=np.ones(len(X_NEW))*(lane[:,2].mean())
            # print('lane[:,2].min()',lane[:,2].min())
            # for i in range(len(X_NEW)):
            #     new=(lane[:,2].min())/(len(X_NEW)-i+1)
            #     z[i]=new 

            # # print('mean',lane[:,2].mean())
            intensity=np.ones(len(X_NEW))*lane[:,3].mean()
   
            point1=np.concatenate((X_NEW.reshape(-1,1),Y_NEW.reshape(-1,1)),axis=1)
            point2=np.concatenate((z.reshape(-1,1),intensity.reshape(-1,1)),axis=1)

            newpoints=np.concatenate((point1,point2),axis=1)
            l.append(newpoints)

            # print(abs(Y_NEW-y))
        
        # break
    
    return l
    # return np.array(l).reshape(-1,4,len(peaks))



##################################################
import os


filenames = os.listdir('umm')
# # # filename='data_road/training/image_2/umm_000011.png'

for filename in filenames:
    filename=filename.replace('.png',"")

# filename='um_000066'

    rgb = imageio.imread('data_road/training/image_2/'+ filename+'.png')


    l=Lidar()
    im=Image()
    data,lidar=l.read_data("data_road_velodyne/training/velodyne/"+filename+".bin")
    calib = im.read_calib_file('data_road/training/calib/'+filename+'.txt')

    h, w, c = rgb.shape

    # im.render_lidar_on_image(lidar,rgb, calib,w,h)

    data=data.to_numpy()
    cloud,ind=l.find_road_plane(data)

    data=remove_noise(cloud)

    data=data.to_numpy()
    print('after road plane and noise removal',len(data))


    # ###############  fidinding the lanes ############

    # plt.figure()
    lane=Lane()
    yval,histVal=lane.peak_intensity_ratio(data,50)

    peaks= find_peaks(histVal)[0]
    # peaks=lane.find_peaks(histVal)

    for p in peaks:
        plt.plot(yval[p],histVal[p],'*r')


    # ##################
    x,y,z,index=im.render_lidar_on_image(data[:,0:4],rgb, calib, w,h,data[:,3])


    fig,ax = plt.subplots(1)
    plt.scatter(data[index,0],data[index,1],color='yellow')
    # plt.ylim(-20,)
    plt.xlim(0,50)

    x=data[:,0]
    min_x=math.ceil(x.min())
    max_x=math.ceil(x.max())

    nbin=max_x-min_x
    x_val=np.linspace(min_x,max_x,nbin)

    arr=[]
    for p in peaks:
        arr.append(yval[p])
    # for y in arr:   
    #     lane.DisplayBins(x_val,y,'red')
    # lane.find_peaks()

    lanes =lane.DetectLanes(data[index,0:4],1,50,arr,min_x,max_x,len(peaks))

    no_of_lanes_detected=lanes.shape[2]

    new_lanes=np.zeros(no_of_lanes_detected)

    for i in range(no_of_lanes_detected):
        l=lanes[:,:,i]
        
        l=[l[i,:] for i in range(len(l)) if l[i,0]!=0 and l[i,1]!=0]
        
        l=np.array(l).reshape(-1,4) 

        plt.plot(l[:,0],l[:,1],color='blue')
        
        
    # im.render_lanes_on_image(lanes,rgb, calib, w,h,filename)

    fitted_lane=fit_polynomial(lanes,peaks)

    for d in fitted_lane:
        plt.plot(d[:,0],d[:,1],color='red')
        
    im.render_lanes_on_image(fitted_lane,rgb, calib, w,h,filename)

    # plt.show()











