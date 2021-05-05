#!/usr/bin/env python3

import copy
import numpy as np
import pandas as pd
import pyproj as pj
import pcl
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

    def data(self,data_path):
        
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

    
    def get_trajectory_line(self,trajectory, min_x,min_y,min_z):
        trajectory_df = self.data(trajectory)
        trajectory_df["Latitude"] =pd.to_numeric(trajectory_df["Latitude"] )
        trajectory_df["Longitude"]=pd.to_numeric(trajectory_df["Longitude"])
        trajectory_df["Altitude"]=pd.to_numeric(trajectory_df["Altitude"])
        trajectory_df["Intensity"]=pd.to_numeric(trajectory_df["Intensity"])

        coords=self.convert_to_xyz(trajectory_df["Latitude"],trajectory_df["Longitude"])

        x=[]
        y=[]
        # plt.figure()
        for i in range(len(coords)):
            # print(coords[i][0],coords[i][1])
            x.append(coords[i][0])
            y.append(coords[i][1])

           
        # # #  data nomalization 
        # # x=x-min_x 
        # # y=y-min_y 
           
        #  not convertred z but saving xy instead of Latitde and long 
        # trajectory_df[["Latitude", "Longitude", "Altitude", "Intensity"]].to_csv("./data/trajectory.xyz", index=False)
        line = "LINESTRING("
        for i in range(len(x)):
             line = line + str(x[i]) + " " + str(y[i]) + ", "
        line = line[:-2] + ")"
        trajectory_line = wkt.loads(line)
        # print(trajectory_line)
        
        return trajectory_line,x,y
    


  
    def filter_by_trajectory_line(self,x,y,p):
        x_filter=[]
        y_filter=[]
        f = np.poly1d(p)
        for i in range(len(x)):
            ypred=f(x[i])
            # p3=np.asarray(x[i],y[i])
            # d = abs((a * x[i] + b * y[i] + c)) / (np.sqrt(a * a + b * b)) 
            # print("Perpendicular distance is", d)
            # d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
            # print(abs(ypred-y[i]))
            if abs(ypred-y[i])<20:
                x_filter.append(x[i])
                y_filter.append(y[i])

        return x_filter,y_filter


    def find_road_plane(self,points):

        cloud = pcl.PointCloud_PointXYZI()
        cloud.from_array(points.astype('float32'))
     
        # find normal plane
        seg = cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_normal_distance_weight(0.001)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.3)
        indices, model = seg.segment()

        cloud_plane = cloud.extract(indices, negative=False)        

        return cloud_plane.to_array(), np.array(indices)
                    

if __name__ == '__main__':
    l=Lidar()
    data=l.data("data/pointcloud.fuse")
    # print(data)
    data["Latitude"] =pd.to_numeric(data["Latitude"] )
    data["Longitude"]=pd.to_numeric(data["Longitude"])
    data["Altitude"]=pd.to_numeric(data["Altitude"])
    data["Intensity"]=pd.to_numeric(data["Intensity"])
    coords= l.convert_to_xyz(data.iloc[:, 0],data.iloc[:, 1])
    # print(coords)
    x=[]
    y=[]
    # plt.figure()
    for i in range(len(coords)):
        # print(coords[i][0],coords[i][1])
        x.append(coords[i][0])
        y.append(coords[i][1])

    x=np.asarray(x)
    y=np.asarray(y)

    data['x']=x
    data['y']=y

    # print(data)

    # plt.savefig('/home/ashlee/person_project/lidar/lane-detection/rawdata.png')
   
# #     # #  filters for point clSoud 
    mean,std,data=l.mean_filter(data)

    # plt.xlim(data['x'].min(),data['x'].max())
    # plt.ylim(data['y'].min(),data['y'].max())
    # plt.scatter(data['x'],data['y'],c=data['Intensity'])
    # plt.show()
    
    #  converting into pointcloud format  xyzI 
    p=pd.DataFrame()
    p['x']=data['x']
    p['y']=data['y']
    p['z']=data["Altitude"]- data["Altitude"].min()
    p['I']=data['Intensity']

    #  convert dataframe to numpy 
    p=p.to_numpy()

    road_plane, road_plane_idx = l.find_road_plane(p)
    road_plane_flatten = road_plane[:,0:2]

    # cluster road plane, and find the road segment
    db = DBSCAN(eps=1, min_samples=100).fit_predict(road_plane_flatten)
    largest_cluster_label = stats.mode(db).mode[0]
    largest_cluster_points_idx = np.array(np.where(db == largest_cluster_label)).ravel()

    road_plane_seg_idx = road_plane_idx[largest_cluster_points_idx]
    
    # print(len(road_plane_seg_idx),len(road_plane))

#     # print(p[road_plane_seg_idx, :])
#     # p=np.asarray(p)
    plt.xlim(p[road_plane_seg_idx,0].min(),p[road_plane_seg_idx,0].max())
    plt.ylim(p[road_plane_seg_idx,1].min(),p[road_plane_seg_idx,1].max())
    plt.scatter(p[road_plane_seg_idx,0],p[road_plane_seg_idx,1],c=p[road_plane_seg_idx,3])
   
    trajectory_line,xlane,ylane=l.get_trajectory_line("data/trajectory.fuse",0,0,0)
   
    # plt.figure()
    # plt.xlim(data['x'].min(),data['x'].max())
    # plt.ylim(data['y'].min(),data['y'].max())
    # plt.scatter(data['x'],data['y'])
    # plt.plot(xlane,ylane, linestyle = 'dotted',color="red")
    

    

    p1=(xlane[0],ylane[0])
    p2=(xlane[8],ylane[8])
    p1=np.asarray(p1)
    p2=np.asarray(p2)
    coef = np.polyfit(xlane, ylane, 1)
    A = coef[0]
    B = coef[1]
    # C = A*xlane[0] + B*ylane[0]
    
    # print(f)
    # plt.plot(x,f(x),color="green")
    # plt.show()

# #     # print(A,B,C)
    x_filter,y_filter=l.filter_by_trajectory_line(p[road_plane_seg_idx,0],p[road_plane_seg_idx,1],coef)
#     print("Intensity - Filtered points: ", len(x_filter))
#     print("Intensity - Original points: ", len(x))
#     print("Intensity - Reduction to %:  ", len(x_filter)/len(x))

# #     pc=pd.DataFrame()
# #     pc["x"]=x_filter
# #     pc["y"]=y_filter

    plt.scatter(x_filter,y_filter,color="red")
# #     # plt.savefig('/home/ashlee/Videos/after_trajectoryline_filter.png')
    plt.show()

# #     # X =(x_filter,y_filter)
# #     # X = StandardScaler().fit_transform(X)

# #     # db = DBSCAN(eps=0.006, min_samples=1).fit(X)
# #     # labels = db.labels_
# #     # print(labels)
# #     # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# #     # n_noise_ = list(labels).count(-1)

# #     # # pc["Group"] = labels
# #     # print(pc.head())
# #     # print(len(x_filter),len(y_filter),len(labels))
# # # cluster_df = lanes_df[["Easting", "Northing", "Altitude", "Group"]]
# # # cluster_df = cluster_df[cluster_df["Group"]>=0]
# # # cluster_df.to_csv("./data/clustering.xyz",index=False)
# # # cluster_df.head()
    











