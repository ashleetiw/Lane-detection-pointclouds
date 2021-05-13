#!/usr/bin/env python3

import copy
import numpy as np
import pandas as pd
import pyproj as pj
import pcl
import math
from pyproj import Transformer
from numpy.linalg import norm
from numpy.polynomial import polynomial as P
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from shapely import wkt
import utm 
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D

class Lidar:
    # def __init(self):
    #     #  projections setup
    #     self.wgs=pj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic
    #     self.bng=pj.Proj(init='epsg:3857')  # use a locally appropriate projected CRS

    def data_GPS(self,data_path):
        
        data=[]
        with open(data_path) as f:  
            line = f.readline()
            while line:
                d = line.split()
                data.append(d)
                line = f.readline()

        raw_data = np.array(data)
        #  converting raw data into tabular -readable form 
        pc = pd.DataFrame()
        pc["Latitude"] = raw_data[:,0]
        pc["Longitude"] = raw_data[:,1]
        pc["Altitude"] =  raw_data[:,2]
        pc["Intensity"] = raw_data[:,3]

        return pc


    def read_data(self,data_path):
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
        return data

    def visualize(self,data):
        plt.figure()
        plt.xlim(data['x'].min(),data['x'].max())
        plt.ylim(data['y'].min(),data['y'].max())
        plt.scatter(data['x'],data['y'],c=data['Intensity'])
        plt.show()
 
    

    def convert_to_xyz(self,lat, lon):
        # France zone 
        P = pj.Proj(proj='utm', zone=31, ellps='WGS84', preserve_units=False)
        x, y =P(lat,lon)

        # transformer = Transformer.from_crs('epsg:4269','epsg:4326',always_xy=True)
        points = list(zip(x,y))
        # coordsWgs = np.array(list(transformer.itransform(points)))
        
        return points

    def convert_to_latlon(self,x,y):
        lat, lon = pj.transform(self.bng, self.wgs, x, y)
        return lat,lon


    def mean_filter(self,pc):

        mean = pc["Intensity"].mean()
        std = pc["Intensity"].std()

        filtered_lanes = pc[pc["Intensity"] > mean + 1 * std]
        filtered_lanes = filtered_lanes[filtered_lanes["Intensity"] < mean + 7 * std ]
        
        return mean,std,filtered_lanes

    def find_road_plane(self,points):

        cloud = pcl.PointCloud_PointXYZI()
        cloud.from_array(points)
        
        print(len(points)/100)
        # find normal plane
        seg = cloud.make_segmenter_normals(ksearch=len(points)/100)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_normal_distance_weight(0.001)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.3)
        indices, model = seg.segment()

        cloud_plane = cloud.extract(indices, negative=False)        

        return cloud_plane.to_array(), np.array(indices)
        # return 0,0 

if __name__ == '__main__':
    l=Lidar()
    data=l.read_data("data_road_velodyne/training/velodyne/um_000051.bin")
    # l.visualize(data)
    mean,std,data=l.mean_filter(data)
  
    
    data=data.to_numpy()
    db = DBSCAN(eps=1, min_samples=10).fit_predict(data[:,0:2])

    p=pd.DataFrame()
    p['x']=data[:,0]
    p['y']=data[:,1]
    p['z']=data[:,2]
    p['Intensity']=data[:,3]
    p['label']=db
   
    p = p[p["label"]>-1 ]

    '''
     r = sqrt(x*x + y*y + z*z)
     phi = atan2(y,x)
     theta = acos(z,r)
    '''
    p['r'] = np.sqrt(p['x'] ** 2 + p['y'] ** 2+p['z'] ** 2)
    #  take only nearby lines into consideration 
    
    # p = p[p['r']<p['r'].mean()]

    # x_dummy=p['x'].to_numpy()
    # y_dummy=p['y'].to_numpy()
    # # print(type(x1))
    
    p['phi']=np.degrees(np.arctan(p['x']/p['y']))

    # print('max',abs(p['phi']).max())
    # print('min',abs(p['phi']).min())
    # print('mean ',abs(p['phi']).mean())

    # print('data before',p.shape[0],p.shape[1])    
    # p = p[abs(p['phi'])>50 ]
    # p=p.to_numpy()
    # print(p.shape[0],p.shape[1])
    f, ax = plt.subplots(2,1)
    ax[0].scatter(p['x'],p['y'],c=p['label'])
    ax[1].scatter(p['r'],p['phi'],c=p['label'])
    plt.show()


    
    # plt.xlim(data[road_plane_seg_idx,0].min(),data[road_plane_seg_idx,0].max())
    # plt.ylim(data[road_plane_seg_idx,1].min(),data[road_plane_seg_idx,1].max())
    # plt.scatter(data[road_plane_seg_idx,0],data[road_plane_seg_idx,1],c=data[road_plane_seg_idx,3])
    # plt.show()



  
    











