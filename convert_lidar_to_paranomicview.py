#!/usr/bin/env python3
'''
   Helper method to visulaize lidar point cloud data to bird's eye view
'''
import numpy as np
import matplotlib.pyplot as plt

class overlay:

    def load_from_bin(self,bin_path):
        '''  reads data '''
        obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        # ignore reflectivity info
        return obj[:,:3]


    def normalize_depth(self,val, min_v, max_v):
        """ 
        print 'nomalized depth value' 
        nomalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
        """
        return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)

    def normalize_val(self,val, min_v, max_v):
        """ 
        print 'nomalized depth value' 
        nomalize values to 0-255 & close distance value has low value.
        """
        return (((val - min_v) / (max_v - min_v)) * 255).astype(np.uint8)

    def in_h_range_points(self,m, n, fov):
        """ extract horizontal in-range points """
        return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), \
                            np.arctan2(n,m) < (-fov[0] * np.pi / 180))

    def in_v_range_points(self,m, n, fov):
        """ extract vertical in-range points """
        return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), \
                            np.arctan2(n,m) > (fov[0] * np.pi / 180))

    def fov_setting(self,points, x, y, z, dist, h_fov, v_fov):
        """ filter points based on h,v FOV  """
        
        if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
            return points
        
        if h_fov[1] == 180 and h_fov[0] == -180:
            return points[self.in_v_range_points(dist, z, v_fov)]
        elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
            return points[self.in_h_range_points(x, y, h_fov)]
        else:
            h_points = self.in_h_range_points(x, y, h_fov)
            v_points = self.in_v_range_points(dist, z, v_fov)
            return points[np.logical_and(h_points, v_points)]

    def velo_points_2_pano(self,points, v_res, h_res, v_fov, h_fov, depth=False):

        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # project point cloud to 2D point map
        x_img = np.arctan2(-y, x) / (h_res * (np.pi / 180))
        y_img = -(np.arctan2(z, dist) / (v_res * (np.pi / 180)))

        """ filter points based on h,v FOV  """
        x_img = self.fov_setting(x_img, x, y, z, dist, h_fov, v_fov)
        y_img = self.fov_setting(y_img, x, y, z, dist, h_fov, v_fov)
        dist = self.fov_setting(dist, x, y, z, dist, h_fov, v_fov)

        x_size = int(np.ceil((h_fov[1] - h_fov[0]) / h_res))
        y_size = int(np.ceil((v_fov[1] - v_fov[0]) / v_res))
        
        # shift negative points to positive points (shift minimum value to 0)
        x_offset = h_fov[0] / h_res
        x_img = np.trunc(x_img - x_offset).astype(np.int32)
        y_offset = v_fov[1] / v_res
        y_fine_tune = 1
        y_img = np.trunc(y_img + y_offset + y_fine_tune).astype(np.int32)

        if depth == True:
            # nomalize distance value & convert to depth map
            dist = self.normalize_depth(dist, min_v=0, max_v=120)
        else:
            dist = self.normalize_val(dist, min_v=0, max_v=120)

        # array to img
        img = np.zeros([y_size + 1, x_size + 1], dtype=np.uint8)
        img[y_img, x_img] = dist
        
        return img

# velo_points = load_from_bin('data_road_velodyne/training/velodyne/umm_000011.bin')

