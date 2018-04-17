#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:45:14 2018

@author: dkilanga
"""

import numpy as np
import os, sys
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd

def read_usps_data(filedata,filelabels):
    data = []
    labels = []
    data = pd.read_csv(filedata, sep=",", header=None)
    data = data.as_matrix()
    with open(filelabels, "r") as f:
        for line in f:
            labels.append([line.strip()])

    return data, labels

def read_data_from_file(fileName):
    """"Function used to parse files, it returns 2 lists.
    The first list has 10 items, which are arrays of shape 32x32xN.
    Each of the 10 arrays corresponds to data for 1 of the 10 digits. Note that N
    is variable depending on how much data is available for each digit in a file.
    The Second List corresond to 10 1xN arrays of labels which match the data. For instance.
    For instance data_list[0] will return all the data available for digit 0, and labels_list[0] will
    return the corresponding labels.
    data_list[0][:,:,0] will return the first 32x32 example of 0 data and labels_list[0][0] will return the corresponding label"""
    if not os.path.isfile(fileName):
        print('Could not open file:', fileName)
        return None
    count = 0
    n_ind = 0
    row_ind = 0
    with open(fileName, "r") as f:
        for line in f:
            # Remove trailing '\n' and collapse excess spaces
            line = (' '.join(line.strip().split())).split()
            if count == 3:
                w = int(line[-1])
            elif count == 4:
                h = int(line[-1])
            elif count == 8:
                num = int(line[-1])
                data = np.empty((h,w,num),dtype = int)
                labels = np.empty((num), dtype = int)
            elif count > 20:
                data_line = [int(v) for v in line[0]]
                if len(data_line) == 32:
                    data[row_ind,:,n_ind] = np.array(data_line)
                    indx = np.where(data[row_ind,:,n_ind] == 0)
                    data[row_ind,indx,n_ind] = -1
                    row_ind += 1
                else:
                    labels[n_ind] = data_line[0]
                    row_ind = 0
                    n_ind += 1
            count += 1
    
    indx_0 = np.where(labels == 0)
    labels_0 = labels[indx_0]
    data_0 = data[:,:,indx_0]
    shape_0 = data_0.shape
    data_0 = data_0.reshape((shape_0[0], shape_0[1], shape_0[3]))
    
    indx_1 = np.where(labels == 1)
    labels_1 = labels[indx_1]
    data_1 = data[:,:,indx_1]
    shape_1 =data_1.shape
    data_1 = data_1.reshape((shape_1[0], shape_1[1], shape_1[3]))
        
    indx_2 = np.where(labels == 2)
    labels_2 = labels[indx_2]
    data_2 = data[:,:,indx_2]
    shape_2 = data_2.shape
    data_2 = data_2.reshape((shape_2[0], shape_2[1], shape_2[3]))
    
    indx_3 = np.where(labels == 3)
    labels_3 = labels[indx_3]
    data_3 = data[:,:,indx_3]
    shape_3 = data_3.shape
    data_3 = data_3.reshape((shape_3[0], shape_3[1], shape_3[3]))
    
    indx_4 = np.where(labels == 4)
    labels_4 = labels[indx_4]
    data_4 = data[:,:,indx_4]
    shape_4 = data_4.shape
    data_4 = data_4.reshape((shape_4[0], shape_4[1], shape_4[3]))
    
    indx_5 = np.where(labels == 5)
    labels_5 = labels[indx_5]
    data_5 = data[:,:,indx_5]
    shape_5 = data_5.shape
    data_5 = data_5.reshape((shape_5[0], shape_5[1], shape_5[3]))
    
    indx_6 = np.where(labels == 6)
    labels_6 = labels[indx_6]
    data_6 = data[:,:,indx_6]
    shape_6 = data_6.shape
    data_6 = data_6.reshape((shape_6[0], shape_6[1], shape_6[3]))
    
    indx_7 = np.where(labels == 7)
    labels_7 = labels[indx_7]
    data_7 = data[:,:,indx_7]
    shape_7 = data_7.shape
    data_7 = data_7.reshape((shape_7[0], shape_7[1], shape_7[3]))
    
    indx_8 = np.where(labels == 8)
    labels_8 = labels[indx_8]
    data_8 = data[:,:,indx_8]
    shape_8 = data_8.shape
    data_8 = data_8.reshape((shape_8[0], shape_8[1], shape_8[3]))
    
    indx_9 = np.where(labels == 9)
    labels_9 = labels[indx_9]
    data_9 = data[:,:,indx_9]
    shape_9 = data_9.shape
    data_9 = data_9.reshape((shape_9[0], shape_9[1], shape_9[3]))
    data_list = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9]
    labels_list = [labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_7, labels_8, labels_9]      
    
    return data_list, labels_list

class vertex:
    """Vertex object for the ising model.
    Depending on where the node is located it can have 2,3 or 4 neighbors.
    Vertices (Nodes) at corners have 2 neighbors, vertices at the edges have 3 neighbors, and
    vertices anywhere else has 4 neighbors. However, Vertices (Nodes) are not always connected to neighbors.
    The forward_backward step will determine to which neighbor a vertex has to be connected. List self.neighbors will be 
    populated with id number of neighbor vertices. List self.edges will be populated with 0's or 1's.
    For instance, vertex 1 would have self.vertex_id = 1, self.neighbors = [2,33], and self.edges = [0,0] 0r [0,1] or [1,0]
    or [1,1]. If self.edges = [0,0], then there is no edge neither between 1 and 2 nor between 1 and 33.
    self.edges = [0,1] means there is no edge between 1 and 2, but there is an edge between 1 and 33
    self.edges - [1,1] meand there is an edge between 1 and 2 as well as an edge between 1 and 33"""
    def __init__(self, vertex_id):
        self.neighbors = []
        self.edges = []
        self.vertex_id = vertex_id
        self.set_edges = []
    
    def add_neighbor(self, neighboor):
        self.neighbors += neighboor
    
    def add_edges(self, edge):
        self.edges += edge
        
    def update_edge(self,val,indx):
        self.edges[indx] = val
        
    def get_neighbors(self):
        return self.neighbors
    def get_edges(self):
        return self.edges
    
    def expand_set_edges(self):
        self.set_edges.append(1)
    
    def reduce_set_edges(self):
        self.set_edges.pop()
        
    def set_edges_length(self):
        return len(self.set_edges)

class Graph:
    """Graph object in which different methods are implementated learn the structure of the graph"""
    def __init__(self, data):
        """Takes as input an array of dimensions 32x32xN
        self.vertices will be populated with 1024 vertices (32x32=1024)
        self.reshape_data will contain the same data that came in as input, but the array will be reshape to be
        Nx1024
        self.THETA will contain all the weights i.e. (theta_r for each individual node, and theta_rt between node r and
        neighboring node t. self.THETA[r,r] = theta_r and self.THETA[r,t] = theta_rt. Note that self.THETA[r,t] will end up
        being equal to self.THETA[t,r] because it is the same edge seen from different vertices.
        Some portions of the forward_backward algorithm require to have to theta at current time as well as theta at previous
        time step, so self.THTA_previous will play the role of theta at previous time step"""
        self.data = data
        #self.dataShape = self.data.shape
        self.dataShape = (16,16,600)
        self.vertices = []
        #self.reshaped_data = np.empty((self.dataShape[2], self.dataShape[1]*self.dataShape[0]), dtype=int)
        self.reshaped_data = data
        self.THETA = np.zeros((self.dataShape[0]*self.dataShape[0],self.dataShape[0]*self.dataShape[0]))
        self.THETA_previous = np.zeros((self.dataShape[0]*self.dataShape[0],self.dataShape[0]*self.dataShape[0]))
#        self.EDGE = np.zeros((self.dataShape[2],self.dataShape[2]))
        for i in range(0,256):
            self.vertices.append(vertex(i))
        self.topNodes = []
        for i in range(1,15):
            self.topNodes.append(i)
        self.leftNodes = []
        for i in range(16,240,16):
            self.leftNodes.append(i)
        self.rightNodes = []
        for i in range(31,255,16):
            self.rightNodes.append(i)
        self.bottomNodes = []
        for i in range(241,255):
            self.bottomNodes.append(i)
        
#        for i in range(self.dataShape[2]):
#            for j in range(self.dataShape[1]):
#                self.reshaped_data[i,(j*32):(j*32)+32] = self.data[j,:,i]
        
        for vert in self.vertices:
            if vert.vertex_id == 0:
                vert.add_neighbor([1,16])
                vert.add_edges([0,0])
            elif vert.vertex_id == 15:
                vert.add_neighbor([14,31])
                vert.add_edges([0,0])
            elif vert.vertex_id == 240:
                vert.add_neighbor([224,241])
                vert.add_edges([0,0])
            elif vert.vertex_id == 255:
                vert.add_neighbor([239,254])
                vert.add_edges([0,0])
            elif vert.vertex_id in self.topNodes:
                vert.add_neighbor([vert.vertex_id-1, vert.vertex_id+1, vert.vertex_id+16])
                vert.add_edges([0,0,0])
            elif vert.vertex_id in self.leftNodes:
                vert.add_neighbor([vert.vertex_id-16, vert.vertex_id+1, vert.vertex_id+16])
                vert.add_edges([0,0,0])
            elif vert.vertex_id in self.rightNodes:
                vert.add_neighbor([vert.vertex_id-16, vert.vertex_id-1, vert.vertex_id+16])
                vert.add_edges([0,0,0])
            elif vert.vertex_id in self.bottomNodes:
                vert.add_neighbor([vert.vertex_id-16, vert.vertex_id-1, vert.vertex_id+1])
                vert.add_edges([0,0,0])
            else:
                vert.add_neighbor([vert.vertex_id-16, vert.vertex_id-1, vert.vertex_id+1, vert.vertex_id+16])
                vert.add_edges([0,0,0,0])
    def loss_theta(self, v):
        """Implementation of loss function...
        Ended up not using this method. Can be deleted, but keeping it until 100% sure that it
        is not needed"""
        interim_sum = np.zeros((self.dataShape[2],1))
        for n_vert_id in v.get_neighbors():
            interim_sum += self.THETA[v.vertex_id,n_vert_id]*np.multiply(self.reshaped_data[:,v.vertex_id], self.reshaped_data[:,n_vert_id]).reshape((-1,1))
        interim_sum_ = interim_sum + self.THETA[v.vertex_id,v.vertex_id]*self.reshaped_data[:,v.vertex_id].reshape((-1,1))
        interim_exp = np.exp(interim_sum_)
        interim_log = np.log(np.add(1,interim_exp))
        interim_loss = np.subtract(interim_log, self.THETA[v.vertex_id,v.vertex_id]*self.reshaped_data[:,v.vertex_id])
        interim_loss = np.subtract(interim_loss,interim_sum)
        real_loss = np.divide(np.sum(interim_loss), self.dataShape[2])
        
        return real_loss
    
    def loss_edges(self, v, theta):
        """Implementation of loss function used to evaluate loss when edges are added or removed.
        v is the vertex at which the algorithm is trying to either add or remove edges connecting to
        neighbors
        theta contains eiter 3,4 or 5 elements depending on what type of node v is.
        if v is corner node, then theta has 3 element theta[0,0] = theta_r while theta[0,1] and 
        theta[0,2] correspond to theta_rt for each of the neighbors"""
        interim_sum = np.zeros((self.dataShape[2],1))
        for indx, n_vert_id in enumerate(v.get_neighbors()):
            interim_sum += theta[0,indx+1]*np.multiply(self.reshaped_data[:,v.vertex_id], self.reshaped_data[:,n_vert_id]).reshape((-1,1))
        interim_sum_ = interim_sum + theta[0,0]*self.reshaped_data[:,v.vertex_id].reshape((-1,1))
        interim_exp = np.exp(interim_sum_)
        interim_log = np.log(np.add(1,interim_exp))
        interim_loss = np.subtract(interim_log, theta[0,0]*self.reshaped_data[:,v.vertex_id])
        interim_loss = np.subtract(interim_loss,interim_sum)
        real_loss = np.divide(np.sum(interim_loss), self.dataShape[2])
        
        return real_loss
        
    
    def minimize_loss_theta(self,v, step):
        """Helper function used to update theta by minimizing the loss with respect to theta i.e. (argmin of loss with theta
        as the arg) To minimize loss, we have to take the derivative of loss with resprect to theta_r
        and with respect to theta_rt. Since the derivative did not yield a closed form, gradient
        descent was performed for 150,000 iterations. theta_r is always updated, but theta_rt is
        updated only if there is an edge between vertex r and its neighbor t. This to insure that this
        is the case, the update step * gradient is multiplied by v.get_edges()[i] because if there is
        an edge between verted r and its first neighbor, then v.get_edges()[0] will be 1, and the 
        update will take place, but if there is no edge, the v.get_edges()[0] will be 0, and
        theta_rt will not be update. It will stay at 0."""
        num_neighbors = len(v.get_neighbors())
        for i in range(150000):
            interim_sum = np.zeros((self.dataShape[2],1))
            for n_vert_id in v.get_neighbors():
                interim_sum += self.THETA[v.vertex_id,n_vert_id]*np.multiply(self.reshaped_data[:,v.vertex_id], self.reshaped_data[:,n_vert_id]).reshape((-1,1))
            interim_sum += self.THETA[v.vertex_id,v.vertex_id]*self.reshaped_data[:,v.vertex_id].reshape((-1,1))
            interim_exp = np.exp(interim_sum)
            num_1 = np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)), interim_exp)
            den = np.add(1,interim_exp)
            ratio_1 = np.divide(num_1,den)
            adj_ratio_1 = np.subtract(ratio_1,self.reshaped_data[:,v.vertex_id].reshape((-1,1)))
            grad_r = np.divide(np.sum(adj_ratio_1),self.dataShape[2])

            
            num_2 = np.multiply(np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[0]].reshape((-1,1))),interim_exp)
            ratio_2 = np.divide(num_2,den)
            adj_ratio_2 = np.subtract(ratio_2,np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[0]].reshape((-1,1))))
            grad_rt_2 = np.divide(np.sum(adj_ratio_2),self.dataShape[2])
            
            num_3 = np.multiply(np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[1]].reshape((-1,1))),interim_exp)
            ratio_3 = np.divide(num_3,den)
            adj_ratio_3 = np.subtract(ratio_3,np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[1]].reshape((-1,1))))
            grad_rt_3 = np.divide(np.sum(adj_ratio_3),self.dataShape[2])
            
            if num_neighbors >= 3:
                num_4 = np.multiply(np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[2]].reshape((-1,1))),interim_exp)
                ratio_4 = np.divide(num_4,den)
                adj_ratio_4 = np.subtract(ratio_4,np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[2]].reshape((-1,1))))
                grad_rt_4 = np.divide(np.sum(adj_ratio_4),self.dataShape[2])
            
            if num_neighbors == 4:
                num_5 = np.multiply(np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[3]].reshape((-1,1))),interim_exp)
                ratio_5 = np.divide(num_5,den)
                adj_ratio_5 = np.subtract(ratio_5,np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,v.get_neighbors()[3]].reshape((-1,1))))
                grad_rt_5 = np.divide(np.sum(adj_ratio_5),self.dataShape[2])
                
            self.THETA[v.vertex_id,v.vertex_id] -= step*grad_r
            self.THETA[v.vertex_id,v.get_neighbors()[0]] -= v.get_edges()[0]*step*grad_rt_2
            self.THETA[v.vertex_id,v.get_neighbors()[1]] -= v.get_edges()[1]*step*grad_rt_3
            if num_neighbors >=3:
                self.THETA[v.vertex_id,v.get_neighbors()[2]] -= v.get_edges()[2]*step*grad_rt_4
            if num_neighbors ==4:
                self.THETA[v.vertex_id,v.get_neighbors()[3]] -= v.get_edges()[3]*step*grad_rt_5

##################################################################################################
# These are just a couple of print statements for debbuging purposes            
        print("For Vertex ", v.vertex_id)
        print(self.THETA[v.vertex_id,v.vertex_id], self.THETA[v.vertex_id,v.get_neighbors()[0]], self.THETA[v.vertex_id,v.get_neighbors()[1]])
        if num_neighbors >= 3:
            print(self.THETA[v.vertex_id,v.get_neighbors()[2]])
        if num_neighbors ==4:
            print(self.THETA[v.vertex_id,v.get_neighbors()[3]])
        # This return is not necessary, it was used while debbuging. Will be deleted later    
        return interim_exp
#################################################################################################    
    def minimize_loss_edge(self,v,nb_vert_id,indx,step,mode):
        """Helper function used to update theta by minimizing loss with respect to edge i.e. (argmin 
        of loss with respect to 'j'. One of its input is mode. if mode is 0, then it serves
        in the minimization right below the first while statement in the pseudo code, and if mode is 1,
        it serves in the minimization right under the second while statement in the pseudo code"""
        interim_theta = np.zeros((1,len(v.get_edges())+1))
        interim_theta[0,0] = self.THETA_previous[v.vertex_id,v.vertex_id]
        num_neighbor = len(v.get_edges())
        for i in range(1,num_neighbor+1):
            interim_theta[0,i] = self.THETA_previous[v.vertex_id,v.get_neighbors()[i-1]]
        if mode == 1:
            interim_theta[0,indx+1] = 0
        interim_theta_prev = np.array(interim_theta)
        for i in range(150000):
            interim_sum = np.zeros((self.dataShape[2],1))
            for n_vert_id in v.get_neighbors():
                interim_sum += interim_theta[0,indx+1]*np.multiply(self.reshaped_data[:,v.vertex_id], self.reshaped_data[:,n_vert_id]).reshape((-1,1))
            interim_sum += interim_theta[0,0]*self.reshaped_data[:,v.vertex_id].reshape((-1,1))
            interim_exp = np.exp(interim_sum)
            num_1 = np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)), interim_exp)
            den = np.add(1,interim_exp)
            ratio_1 = np.divide(num_1,den)
            adj_ratio_1 = np.subtract(ratio_1,self.reshaped_data[:,v.vertex_id].reshape((-1,1)))
            grad_1 = np.divide(np.sum(adj_ratio_1),self.dataShape[2])
#            print "Gradient Edges 1 ", grad_1
            interim_theta[0,0] = interim_theta[0,0] - step*grad_1
            
            num_2 = np.multiply(np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,nb_vert_id].reshape((-1,1))),interim_exp)
            ratio_2 = np.divide(num_2,den)
            adj_ratio_2 = np.subtract(ratio_2,np.multiply(self.reshaped_data[:,v.vertex_id].reshape((-1,1)),self.reshaped_data[:,nb_vert_id].reshape((-1,1))))
            grad_2 = np.divide(np.sum(adj_ratio_2),self.dataShape[2])
#            print "Gradient Edges 2  ", grad_2
            interim_theta[0,indx+1] = interim_theta[0,indx+1] - step*grad_2
#            print "Loss ", self.loss_edges(v,interim_theta)
        
        return (interim_theta, interim_theta_prev)
            
        
    
    def forward_backward(self, v, step, epsilon, nu):
        while True:
            theta = []
            theta_prev = []
            loss_edges = []
            edges = []
            for indx, n_vert_id in enumerate(v.get_neighbors()):
                if v.get_edges()[indx] != 1:
                    new_theta, prev_theta = self.minimize_loss_edge(v,n_vert_id,indx,step,0)
                    theta.append(new_theta)
                    theta_prev.append(prev_theta)
                    edges.append(n_vert_id)
            for i in range(len(theta)):
                loss_edges.append(self.loss_edges(v,theta[i]))
            print(loss_edges)
            min_loss_indx = np.argmin(loss_edges)
            add_edge_to_vert = edges[min_loss_indx]
            v.update_edge(1,v.get_neighbors().index(add_edge_to_vert))
            print("Active Edges Forward ", v.get_edges())
            v.expand_set_edges()
            
            loss_prev = self.loss_edges(v,theta_prev[min_loss_indx])
            delta_f = loss_prev - loss_edges[min_loss_indx]
            print("delta F ", delta_f, "epsilon ", epsilon)
            if delta_f <= epsilon:
                break
            np.copyto(self.THETA_previous, self.THETA) 
            self.minimize_loss_theta(v, step)
            
            while True:
                theta_r = []
                theta_r_prev = []
                loss_edges_r = []
                edges_r = []
                for indx, n_vert_id in enumerate(v.get_neighbors()):
                    if v.get_edges()[indx] != 0:
                        new_theta, prev_theta = self.minimize_loss_edge(v,n_vert_id,indx,step,1)
                        theta_r.append(new_theta)
                        theta_r_prev.append(prev_theta)
                        edges_r.append(n_vert_id)
                if len(theta_r) == 0:
                    break
                for i in range(len(theta_r)):
                    loss_edges_r.append(self.loss_edges(v,theta_r[i]))
                min_loss_indx = np.argmin(loss_edges_r)
                loss_prev = self.loss_edges(v,theta_r_prev[min_loss_indx])
                delta_b = loss_edges_r[min_loss_indx] - loss_prev
                print("delta B ", delta_b, "nu * DF ", nu*delta_f)
                if (delta_b > nu*delta_f) or (v.set_edges_length() == 0):
                    break
                remove_edge_to_vert = edges[min_loss_indx]
                v.update_edge(0,v.get_neighbors().index(remove_edge_to_vert))
                print("Active Edges Backward ", v.get_edges())
                v.reduce_set_edges()
                self.THETA[v.vertex_id,remove_edge_to_vert] = 0
                self.minimize_loss_theta(v, step)
                np.copyto(self.THETA_previous, self.THETA)
            
        return theta
    
    def train(self, indices):
#        interim_exp = self.minimize_loss_theta(self.vertices[0],0.01)
        for vertex in self.vertices[indices[0]:indices[1]]:
            theta = self.forward_backward(vertex,0.01,95.0,0.4)
        print("###########################It Ended####################################")
        return self.THETA
        
#        self.THETA[v.vertex_id,v.vertex_id] = self.THETA[v.vert_id,v.vert_id] - ( step * (1/self.dataShape[2]) *    ) 
        
#        for vert in self.vertices:
#            print len(vert.neighboors)
#        print self.reshaped_data.shape
#        print ''.join(map(str,self.reshaped_data[0,0:32]))
#        print ''.join(map(str, self.reshaped_data[0,32:64]))
            
            

#fileName = "optdigits-orig.tra"
#train_data, train_labels = read_data_from_file(fileName)



#for i in range(10):
#    print train_data[i].shape, train_labels[i]

#for i in range(32):
# print ''.join(map(str,train_data[0][i,:,0]))

#for i in range(1933,1923, -1):
#    print train_labels[0,i]

#ising_model = Graph(train_data[0]).train()
data, labels = read_usps_data("tr_X.txt","tr_y.txt")
data[data == 0] = -1

    
#ising_model = Graph(data[:600]).train()

ising_model = Graph(data[:600,:])
#ising_model.train([[0,64], [64,128],[128,192], [192,256]])
results = Pool(3).map(ising_model.train, [[0,64], [64,128],[128,192], [192,256]])

np.savetxt("model_ups_1.csv", results[0], delimiter=",")
np.savetxt("model_ups_2.csv", results[1], delimiter=",")
np.savetxt("model_ups_3.csv", results[2], delimiter=",")
np.savetxt("model_ups_4.csv", results[3], delimiter=",")
