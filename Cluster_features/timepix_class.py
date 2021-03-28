import numpy as np
import math
import datetime
import pandas as pd
import h5py
import json
import glob
import os
import tables
from cluster_class import Cluster

# Timepix class
class Timepix:
    def __init__(self, logname, dirname):
        print('kacer')
        # Constructor
        self.columns = ['time', 'volume', 'size', 'height', 'energy per pixel', 'lower decile', 'upper decile', 'median', 'hull width', 'hull height', 'hull area', 'hull occupancy', 'linearity', 'eigen ratio', 'SCHR']
        self.lucid_columns = ['time', 'frame', 'track', 'volume', 'size', 'height', 'average_neighbours', 'radius', 'diameter', 'density', 'line_residual', 'width', 'curvature_radius', 'circle_residual']
        self.logname = logname
        self.dirname = dirname
        self.Open_log()
        self.Properties()
        self.frame_list = self.Open_data()
        print('data is read')
        self.data_cluster = self.Clusterfy()
        print('data is featured')
        self.Save_to_hdf()
        print('data is saved in ' + self.measurement_name + '.hdf')

    def Open_log(self):
        # Opens logbook file with detail parameters of the measurement
        with open(self.logname) as file:
            data = json.load(file)
            self.measurement_name = data['measurement name']
            self.timepix = data['timepix']
            self.bias = data['bias voltage']
            self.acquisition_time = data['acquisition time']
            self.calibration = data['calibration']
            self.threshold = data['threshold']
            self.frequency = data['frequency']
            self.calibration_file = data['calibration file']

    def Properties(self):
        # Determines the properties of chip (mass and width) based on the manufacture number
        if (self.timepix == "H08-W0276" or "I08-W0276"):
            self.mass = 2.3008477184E-04
            self.width = 300
        elif (self.timepix == "C08-W0276"):
            self.mass = 3.8347461973E-04
            self.width = 500
        else:
            self.mass = 0
            self.width = 0

    def Open_data(self):
        # Opens all .clog files in a directory, slices data into frames and saves data to a matrix
        out_list = []
        token = self.dirname + '/*.clog'
        for filepath in glob.iglob(token):
            with open(filepath) as file:
                filename = os.path.basename(filepath)
                lines = file.readlines()
                clusters = []
                for line in lines:
                    if (line.find('Frame') != -1):
                        if clusters != []:
                            out_list.append((filename, frame_number, frame_time, acquisition_time, clusters))
                        clusters = []
                        words = line.split()
                        frame_number = int(words[1])
                        time = words[2].replace('(', '').replace(',','')
                        frame_time = float(time)
                        acquisition_time = float(words[3])
                        #print(frame_number, frame_time, acquisition_time)
                    else:
                        line = line.replace('[', '').replace(']','').replace(',', '')
                        words = line.split()
                        clusters.append(words)
        return out_list

    def Clusterfy(self):
        # Processes the .clog data and creates objects of Cluster class
        cluster_list = []
        for filename, frame_number, frame_time, acquisition_time, clusters in self.frame_list:
            for idx, i in enumerate(clusters):
                cluster = Cluster(i, self.mass, self.width, filename, frame_number, idx, frame_time, acquisition_time)
                cluster_list.append(cluster)
        return cluster_list

    def Save_to_hdf(self):
        # Save Clusters to .hdf file
        hdf_file = self.measurement_name + '.hdf'
        lst = []
        with h5py.File(hdf_file, "a") as data_file:
            grp = data_file.require_group(self.timepix)
            grp.attrs['bias'] = self.bias
            grp.attrs['acquisition_time'] = self.acquisition_time
            grp.attrs['calibration'] = self.calibration
            grp.attrs['threshold'] = self.threshold
            grp.attrs['frequency'] = self.frequency
            grp.attrs['calibration_file'] = self.calibration_file
            grp.attrs['mass'] = self.mass
            grp.attrs['width'] = self.width
            #total = len(self.data_cluster)
            for counter, i in enumerate(self.data_cluster):
                #print(counter + '/' + total)
                idx = i.get_id()
                size = i.get_size()
                
                if size > 1:                 
                    image_grp = grp.require_group('image')
                    image = i.get_image()
                    dset_image = image_grp.create_dataset(idx, data=image)

                    bin_image_grp = grp.require_group('binary_image')
                    binary_image = i.get_binary_image()
                    binary_image = np.asarray(binary_image, dtype=np.bool)
                    dset_bin_image = bin_image_grp.create_dataset(idx, data=binary_image)

                    skeleton_grp = grp.require_group('skeleton')
                    skeleton = i.get_skeleton()
                    dset_skeleton = skeleton_grp.create_dataset(idx, data=skeleton)

                lst.append(i.get_lucid_features())
                
            data = np.array(lst)
            #df = pd.DataFrame(lst, columns=self.lucid_columns)
            #print(df)
            grp.create_dataset('dataset_lucid', data=data)
            grp.create_dataset('dataset_columns', data=np.asarray(self.lucid_columns, dtype='S'))

    def return_timepix(self):
        return self.timepix

    def return_dirname(self):
        return self.dirname

    def return_mass(self):
        return self.mass

    def return_width(self):
        return self.width
