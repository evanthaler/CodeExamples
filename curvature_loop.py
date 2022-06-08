import numpy as np
import random
import gc
import matplotlib.pyplot as plt
import rasterio as rio
import os

#os.chdir('X:/backup/projects/midwest/dems/lidar-las-files/weston-prairie/las-dataset/')
dem_name = r'C:\Users\361045\Documents\projects\ngee\ncalm_data\Teller_North\t47_hillslope\t47hillslope_dtm3m_v2.tif'

src = rio.open(dem_name)
data = src.read(1)

with rio.open(dem_name) as src:
    data_ras = src.read()
    ras_meta = src.profile

wdsz=np.array([15])
cs=3.
outprefix='t47'
outdir = r'C:\Users\361045\Documents\projects\ngee\ncalm_data\Teller_North\t47_hillslope\curvatureRasters\\'
for w in wdsz:
    window = w#oddnumbers only
    dx = cs #meters
    #data = np.loadtxt(dem_name, skiprows=6)
    data[data==ras_meta['nodata']] = np.nan
    
    curvature = np.zeros_like(data)
    x = np.linspace(-dx * float((window - 1) /2), dx * float((window - 1) /2), window)
    y = np.linspace(-dx * float((window - 1) /2), dx * float((window - 1) /2), window)
    X = np.zeros((window) ** 2)
    Y = np.zeros((window ) ** 2)
    
    int_XY = 0
    for x_window in x:
        for y_window in y:
            X[int_XY] = x_window 
            Y[int_XY] = y_window
            int_XY += 1
    print(range(((window - 1) //2),data.shape[0] - ((window - 1) //2)))
    for x_corner in range(((window - 1) //2),data.shape[0] - ((window - 1) //2)):
    
        print( str(int(float(x_corner) /  float(data.shape[0] - ((window - 1) /2) - 1) * 1000)/10.) +'% done')
        for y_corner in range(((window - 1) //2),data.shape[1] - ((window - 1) //2)):
            proceed = 1
            for x_window in range(-((window - 1) //2),1+((window - 1) //2)):
                for y_window in range(-((window - 1) //2),1+((window - 1) //2)):
                    if np.isnan(data[x_corner+ x_window][y_corner + y_window]):
                         proceed = 0   
            if proceed == 1:
                data_window = np.zeros((window ) ** 2)
                int_XY = 0
                for x_window in range(-((window - 1) //2),1+((window - 1) //2)):
                    for y_window in range(-((window - 1) //2),1+((window - 1) //2)):
                        data_window[int_XY] = data[x_corner+ x_window][y_corner + y_window]
                        int_XY += 1
                A = np.array([X**2 * Y **2, X** 2 * Y, X * Y ** 2, X**2, Y **2, X * Y, X, Y, X ** 0 * Y ** 0 + 1]).T
                B = data_window
    
                coeff, r, rank, s = np.linalg.lstsq(A, B,rcond=-1)
                curvature[x_corner][y_corner] = - 2.0 * (coeff[3] + coeff[4])
            # you are the best    
    
    ras_transform = ras_meta["transform"]
    ras_crs = ras_meta["crs"]
    # View spatial attributes
    #ras_transform, ras_crs
    #These steps are only necessary here if the input raster isn't already 1 band and the data aren't already float64
    # Change the count or number of bands from 4 to 1
    ras_meta['count'] = 1
    # Change the data type to float rather than integer
    ras_meta['dtype'] = 'float64'
    
    output_filename = outdir+outprefix+'_curv_'+str(int(dx*w))+'m.tif'
    curvature =curvature.astype('float64')
    with rio.open(output_filename, 'w', **ras_meta) as dst:
        dst.write(curvature, 1)

