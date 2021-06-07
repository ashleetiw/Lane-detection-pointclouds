#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pcl


class Lidar:

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
        return data,pointcloud

    def visualize_lidar_data(self,data):
        plt.figure()
        plt.xlim(data['x'].min(),data['x'].max())
        plt.ylim(data['y'].min(),data['y'].max())
        plt.scatter(data['x'],data['y'],c=data['Intensity'])
        plt.show()
 
    
    def mean_filter(self,pc):

        mean = pc["Intensity"].mean()
        std = pc["Intensity"].std()
        meanz = pc["z"].min()
        stdz = pc["z"].std()
        filtered_lanes = pc[pc["Intensity"] > mean - 1 * std]
        #  remove z as well
        filtered_lanes = filtered_lanes[filtered_lanes["z"] > meanz + stdz ]
        return filtered_lanes

    # running RANSAC Algo
    def find_road_plane(self,points):
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
        seg.set_distance_threshold(0.3)
        indices, model = seg.segment()
        cloud_plane = cloud.extract(indices, negative=False)
        return cloud_plane.to_array(), np.array(indices)

    def running_mean(self,x, N):
        """ x == an array of data. N == number of samples per average """
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)       



class Image:
    def read_calib_file(self,filepath):
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


    
    def project_velo_to_cam2(self,calib):
        P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
        R_ref2rect = np.eye(4)
        R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
        R_ref2rect[:3, :3] = R0_rect
        P_rect2cam2 = calib['P2'].reshape((3, 4))
        proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
        return proj_mat


    def project_to_image(self,points, proj_mat):
        """
        Apply the perspective projection
        Args:
            pts_3d:     3D points in camera coordinate [3, npoints]
            proj_mat:   Projection matrix [3, 4]
        """
        num_pts = points.shape[1]
        points = proj_mat @ points
        
        points[:2, :] /= points[2, :]
        # points[:2, :]=np.nan_to_num(points[:2, :]) 
        return points[:2, :]

    # def convert_kitti_bin_to_pcd(binFilePath):
    #     size_float = 4
    #     list_pcd = []
    #     with open(binFilePath, "rb") as f:
    #         byte = f.read(size_float * 4)
    #         while byte:
    #             x, y, z, intensity = struct.unpack("ffff", byte)
    #             list_pcd.append([x, y, z])
    #             byte = f.read(size_float * 4)
    #     np_pcd = np.asarray(list_pcd)
    #     pcd = pcl.PointCloud()
    #     pcd = open3d.utility.Vector3dVector(np_pcd)
    #     return pcd


    def render_lidar_on_image(self,pts_velo, img, calib, img_width, img_height,label):
        # projection matrix (project from velo2cam2)
        proj_velo2cam2 = self.project_velo_to_cam2(calib)

        # # apply projection
        pts_2d = self.project_to_image(pts_velo.transpose(), proj_velo2cam2)

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
        ax.scatter(imgfov_pc_pixel[0], imgfov_pc_pixel[1],s=3)
        # ax.label()
        # print(len(label[inds]))
        plt.yticks([])
        plt.xticks([])


        return imgfov_pc_pixel[0], imgfov_pc_pixel[1],pts_velo[inds,2],inds


  
    def render_lanes_on_image(self,data,img, calib, img_width, img_height,figg):
        """
        Overlay lane lines on the original frame
        :param: Plot the lane lines if True
        :return: Lane with overlay
        """

        print('data in lane_image fucntion',len(data))
        proj_velo2cam2 = self.project_velo_to_cam2(calib)
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        
        
        # for i in range(data.shape[2]):
        #     d=data[:,:,i]
        for d in data:
            pts_2d = self.project_to_image(d.transpose(), proj_velo2cam2)
            inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] > 0) &
                        (pts_2d[1, :] < img_height) & (pts_2d[1,:]>0)  )[0]

            # print(inds)

            # Filter out pixels points
            imgfov_pc_pixel = pts_2d[:, inds]

            # Retrieve depth from lidar
            imgfov_pc_velo = d[inds, :]
            # imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
            imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
            # Create a figure. Equal aspect so circles look circular  
            # Show the image
            ax.imshow(img)
            ax.scatter(imgfov_pc_pixel[0],imgfov_pc_pixel[1],color='coral')
        
        plt.savefig('result_umm/'+figg+'.png')
        
        # return imgfov_pc_pixel[0], imgfov_pc_pixel[1]

    











