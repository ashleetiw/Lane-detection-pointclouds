#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math 

class Lane:
    def peak_intensity_ratio(self,ptCloud,bin_size):  
        
        y=ptCloud[:,1]
        min_y=math.ceil(y.min())
        max_y=math.ceil(y.max())
        y_val=np.linspace(min_y,max_y,bin_size)
        avg_intensity=[]
        ymean=[]
        for i in range(len(y_val)-1):
            index=self.get_index_inrange(y,y_val[i],y_val[i+1])
            intensity_sum=0
            for j in index:
                intensity_sum+=ptCloud[j,3]

            avg_intensity.append(intensity_sum)
            ymean.append((y_val[i]+y_val[i+1])/2)
        
        plt.plot(ymean,avg_intensity,'--k')
        return ymean,avg_intensity

    def get_index_inrange(self,arr,start,end):
        ind=[i for i in range(len(arr)) if arr[i]>=start and arr[i]<end]
        return ind

    def find_peaks(self,a):
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

    def ransac_polyfit(self,x, y, order=2, n=20, k=100, t=0.1, d=100, f=0.8):
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

    def DisplayBins(self,x_val,y,color):
        y_val=[y]*len(x_val)
        plt.plot(x_val,y_val,c=color)

    

    def DetectLanes(self,data,hbin,vbin, start,min_x,max_x,num_lanes):
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
                # plt.scatter(data[inds,0],data[inds,1],c='yellow')

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
            