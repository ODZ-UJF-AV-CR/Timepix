import timepix_module as tpx
import numpy as np
import pandas as pd
import os

def main():
    filename = 'data_example.clog'
    calibration_path = 'H08-W0276_new'

    tokens = ['unix', 'shutter', 'volume', 'height', 'size', 'x_max', 'y_max']

    cal_matrix = tpx.perpixel_calibration_matrice(calibration_path)

    #data = read_multiple_files(dirname)
    data = tpx.read_clog(filename)
    data = tpx.calibrate_data(data, cal_matrix)

    df = tpx.to_dataframe(data, tokens, cal_matrix)
    dead_time = tpx.calculate_dead_time(df)

    print(dead_time)

main()
