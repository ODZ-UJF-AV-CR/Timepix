import glob
from datetime import datetime
import os
import numpy as np
import pandas as pd
from .Cluster import *

def go_through_files(path, extension):
    temp = os.path.join(path, '*.' + extension)
    print(temp)
    files = glob.glob(temp)
    return files

def read_multiple_files(dirname):
    out = []
    extension = '*.clog'
    dirname = os.path.join(dirname, extension)
    for file in glob.glob(dirname, recursive=True):
        #print(file)
        temp = read_clog(file)
        #print(temp)
        out.extend(temp)
    return out

def read_clog(filename):
    # Opens .clog file, slices data into frames and saves data to a matrix
    cluster_list = []
    with open(filename) as f:
        for line in f:
            if (line.find('Frame') != -1):
                frame = line
            else:
                line = line.replace('[', '').replace(']','').replace(',', '')
                words = line.split()
                x = list(map(int, words[0::3]))
                y = list(map(int, words[1::3]))
                energy = list(map(float, words[2::3]))
                cluster = list(zip(x, y, energy))
                if cluster != []:
                    cluster_list.append((frame, cluster))
    return cluster_list

def read_clog_tpx3(filename):
    # Opens .clog file, slices data into frames and saves data to a matrix
    cluster_list = []
    with open(filename) as f:
        for line in f:
            if (line.find('Frame') != -1):
                frame = line
            else:
                line = line.replace('[', '').replace(']','').replace(',', '')
                words = line.split()
                x = list(map(int, words[0::4]))
                y = list(map(int, words[1::4]))
                energy = list(map(float, words[2::4]))
                cluster = list(zip(x, y, energy))
                if cluster != []:
                    cluster_list.append((frame, cluster))
    return cluster_list

def get_properties(timepix):
    # Determines the properties of chip (mass and width) based on the manufacture number
    if (timepix == "H08-W0276" or "I08-W0276"):
        mass = 2.3008477184E-04
        width = 300
    elif (timepix == "C08-W0276"):
        mass = 3.8347461973E-04
        width = 500
    else:
        mass = 0
        width = 0
    return mass, width

def to_dataframe(data, tokens, cal_matrice):
    dic = {'frame': 0, 'unix': 0, 'shutter': 0, 'volume': 0,
           'height': 0, 'size': 0, 'x_max': 0, 'y_max': 0,
           'Tx': 0, 'Ty': 0, 'volume_2': 0, 'height_2': 0}
    lst = []
    for frame, cluster in data:
        x, y, energy = list(zip(*cluster))
        x = list(x)
        y = list(y)
        energy = list(energy)
        if ('frame' in tokens or 'unix' in tokens or 'shutter' in tokens):
            dic['frame'], dic['unix'], dic['shutter'] = frame_to_words(frame)
        if ('volume' in tokens or 'height' in tokens or 'size' in tokens or 'x_max' in tokens or 'y_max' in tokens):
            dic['volume'], dic['height'], dic['size'], dic['x_max'], dic['y_max'] = get_vhs(x, y, energy)
        if ('Tx' in tokens or 'Ty' in tokens):
            dic['Tx'], dic['Ty'] = centroid_weighted(x, y, energy)
        if ('volume_2' in tokens or 'height_2' in tokens):
            dic['volume_2'], dic['height_2'] = hot_pixel_excluded(energy)
        if ('height_TOT' in tokens):
            dic['height_TOT'] = decalibrate_height(x, y, energy, cal_matrice)

        temp = [dic[i] for i in tokens]
        lst.append(tuple(temp))
    df = pd.DataFrame(lst, columns=tokens)
    return df

def pixelize(df, x, y):
    df_x = df[df['x_max']==x]
    pixel = df_x[df_x['y_max']==y]
    return pixel

def pixelize_to_hdf(df, path, calibration_matrice, distance, bias, attribute):
    for i in range(256):
        #print('Row: ', i)
        df_x = df[df['x_max']==i]
        directory = os.path.join(path, 'row_' + str(i))
        if not os.path.exists(directory):
            os.makedirs(directory)
        for j in range(256):
            file = os.path.join(directory, str(i) + 'x' + str(j) + '.h5')
            pixel = df_x[df_x['y_max']==j]
            a, b, c, t = get_perpixel_coeffs(i, j, calibration_matrice)
            pixel['height_TOT'] = decalibrate_pixel(pixel['height'], a, b, c, t)
            save_table_to_hdf(pixel, file, [a, b, c, t], distance, bias, attribute)

def pixelize_to_image(df):
    arr = np.zeros((256, 256))
    for x in range(256):
        for y in range(256):
            pixel = pixelize(df, x, y)
            mean = len(pixel['volume'])
            #print(mean)
            arr[x, y] = mean
    return arr

def save_table_to_hdf(df, file, calibration, distance, bias, attribute):
    [a, b, c, t] = calibration
    with h5py.File(file, 'a', libver='latest') as hdf:
        if 'A' not in hdf.attrs.keys():
            hdf.attrs['A'] = a
            hdf.attrs['B'] = b
            hdf.attrs['C'] = c
            hdf.attrs['T'] = t

        if distance == None:
            group = hdf.require_group('undefined_dist_data')
        else:
            group = hdf.require_group(str(distance))
        if attribute == None:
            name = str(bias) + '_V'
        else:
            name = str(bias) + '_V_' + attribute
        dset = group.create_dataset(name, data=df)
        dset.attrs['BIAS'] = bias
        if distance != None:
            dset.attrs['DISTANCE'] = distance
        if attribute != None:
            dset.attrs['ATTRIBUTE'] = attribute

def frame_to_words(string):
    data = string.split(' ')
    frame_i = data[1]
    shutter = data[3]
    data[2] = data[2].replace("(", "")
    unix_time = data[2].replace(",", "")
    return frame_i, unix_time, shutter

def read_csv(filename):
    return pd.read_csv(filename)

def perpixel_calibration_matrice(path):
    # Save matrice to memory
    clog_file = os.path.join(path, 'A.txt')
    try:
        A = pd.read_csv(clog_file, sep=' ', header=None, index_col=None)
    except:
        print('Calibration file ' + clog_file + ' could not be found.')
    A = np.array(A)
    A = A.transpose()
    clog_file = os.path.join(path, 'B.txt')

    try:
        B = pd.read_csv(clog_file, sep=' ', header=None, index_col=None)
    except:
        print('Calibration file ' + clog_file + ' could not be found.')
    B = np.array(B)
    B = B.transpose()
    clog_file = os.path.join(path, 'C.txt')

    try:
        C = pd.read_csv(clog_file, sep=' ', header=None, index_col=None)
    except:
        print('Calibration file ' + clog_file + ' could not be found.')
    C = np.array(C)
    C = C.transpose()
    clog_file = os.path.join(path, 'T.txt')

    try:
        T = pd.read_csv(clog_file, sep=' ', header=None, index_col=None)
    except:
        print('Calibration file ' + clog_file + ' could not be found.')
    T = np.array(T)
    T = T.transpose()

    return [A, B, C, T]

def get_perpixel_coeffs(x, y, matrice):
    [A, B, C, T] = matrice
    #print(x, y, matrice)
    a = float(A[x][y])
    b = float(B[x][y])
    c = float(C[x][y])
    t = float(T[x][y])
    return a, b, c, t

def calibrate_data(data, cal_matrix):
    data_out = []
    for frame, cluster in data:
        energy_out = []
        x_out, y_out, _ = list(zip(*cluster))
        for x, y, tot in cluster:
            a, b, c, t = get_perpixel_coeffs(x, y, cal_matrix)
            if (a == 0.0 or b == 0.0 or c == 0.0):
                break
            energy = calibrate_pixel(tot, a, b, c, t)
            #if energy >= 5000:
                #print(cluster)
                #print(energy, tot, a, b, c, t)
            energy_out.append(energy)
        cluster_out = list(zip(x_out, y_out, energy_out))
        if cluster_out != []:
            data_out.append((frame, cluster_out))
    return data_out

def decalibrate_height(x, y, energy, cal_matrix):
    idx = np.argmax(energy)
    en_max = np.max(energy)
    x_max = x[idx]
    y_max = y[idx]
    a, b, c, t = get_perpixel_coeffs(x_max, y_max, cal_matrix)
    TOT = decalibrate_pixel(en_max, a, b, c, t)
    return TOT

def recalibrate_data(data, old_cal_matrix, new_cal_matrix):
    data_out = []
    for frame, cluster in data:
        energy_out = []
        x_out, y_out, _ = list(zip(*cluster))
        for x, y, energy in cluster:
            a, b, c, t = get_perpixel_coeffs(x, y, old_cal_matrix)
            energy_tot = decalibrate_pixel(energy, a, b, c, t)
            a, b, c, t = get_perpixel_coeffs(x, y, new_cal_matrix)
            energy_new = calibrate_pixel(energy_tot, a, b, c, t)
            #print(x, y, energy, energy_tot, energy_new)
            energy_out.append(energy_new)
        cluster_out = list(zip(x_out, y_out, energy_out))
        if cluster_out != []:
            data_out.append((frame, cluster_out))
    return data_out

def save_clog(data, file):
    with open(file, 'w') as f:
        last_frame = 'keine'
        for frame, cluster in data:
            #print(frame, cluster)
            if frame != last_frame:
                f.write(frame)
                last_frame = frame
            cluster = str(cluster)
            cluster = cluster.replace("[", "")
            cluster = cluster.replace("]", "")
            cluster = cluster.replace("(", "[")
            cluster = cluster.replace(")", "]")
            #string = ' '.join(cluster)
            f.write(cluster + '\n')

def calculate_dose(df):
    electron_charge = 1.60218E-19
    mass = 0.1398e-3
    df['dose'] = ((electron_charge * 1000 * df['volume']) / mass) * 1e+06
    return df

def time_series(df, frequency, time=0):
    if time == 0:
        start = min(df['unix'])
        stop = max(df['unix'])
    elif (len(time)==2):
        start = time[0]
        stop = time[1]
        start = datetime.timestamp(datetime.strptime(start, "%Y-%m-%d %H:%M:%S"))
        stop = datetime.timestamp(datetime.strptime(stop, "%Y-%m-%d %H:%M:%S"))
        print(start, stop)
    else:
        print('Time interval in a wrong format.')

    df['date'] = pd.to_datetime(df['unix'], unit='s')

    df.index = df['unix']
    mask = (df.index >= start) & (df.index <= stop)
    df = df.loc[mask]
    df.index = df['date']

    f = {'volume':[list_volume, 'count', 'mean', 'min', 'max', 'sum', 'std'], 'height':['mean', 'min', 'max', 'std'], 'size':['mean', 'min', 'max', 'std']}
    temp = df.groupby(pd.Grouper(freq=frequency)).agg(f)

    #print(temp)
    #print('Time series data were created.')
    return temp

def list_volume(x):
    return list(x)

def calculate_dead_time(df):
    unix = df['unix'].to_numpy().astype('float')
    shutter = df['shutter'].to_numpy().astype('float')
    return 1 - (np.sum(shutter) / (unix[-1] - unix[0] + shutter[-1]))

