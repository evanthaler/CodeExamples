from osgeo import gdal
import pandas as pd
import numpy as np
import os,glob

def bilinear_interp(x,y,data):
    #nothing needs to be changed in this function--unless you find an error!
    #x and y are given as float indices coordinates. WTF is that?
    #x_lower is the x-index integer just under x
    #y_lower is the y-index integer just under y
    #x_upper is the x-index integer just above x
    #y_upper is the y-index integer just above y
    x_lower = int(x)
    y_lower = int(y)
    x_upper = x_lower + 1
    y_upper = y_lower + 1

    #out of range test. Interpolation is not possible is the index is out of range. This will return the edge value.
    if x_lower < 0:
            x_lower = 0
            x_upper = 1
    if y_lower < 0:
            y_lower = 0
            y_upper = 1
    if x_upper >= data.shape[0]:
            x_lower = data.shape[0] - 2
            x_upper = data.shape[0] - 1
    if y_upper >= data.shape[1]: 
            y_lower = data.shape[1] - 2
            y_upper = data.shape[1] - 1
    
    #We first interpolate in the x-direction for y_lower and y_upper. You can do this in the y-direction first, but it still be the same.
    step_1 = (data[x_upper,y_lower] - data[x_lower,y_lower]) / 1.0 * (x - float(x_lower)) + data[x_lower,y_lower]
    step_2 = (data[x_upper,y_upper] - data[x_lower,y_upper]) / 1.0 * (x - float(x_lower)) + data[x_lower,y_upper]

    #We take the two interpolations along y_lower and y_upper and we then interpolate in the x-direction.
    step_3 =  (step_2 - step_1) / 1.0 * (y - float(y_lower)) + step_1

    #step 3 is the bilinear interpolated value!
    return step_3

def point_value(raster, pos):

  '''
  raster: raster dataset in a format that gdal can injest (I typically try to stick with tiffs)
  pos: 1D array with pos[0]=x-coordinate and pos[1]=y-coordinate
  '''
  gdata = gdal.Open(raster)
  gtr = gdata.GetGeoTransform()
  data = gdata.ReadAsArray().astype(float)
  gdata = None
  
  x1 = float((pos[0] - gtr[0])/gtr[1])
  y1 = float((pos[1] - gtr[3])/gtr[5])
  interpolated_value = bilinear_interp(y1,x1,data)
  #return data[y1,x1] ##can use this line instead if you don't want to use the bilinearly interpolated points
  return interpolated_value


os.chdir(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\rasters_3m')
rasList=glob.glob('*.tif')
posFile = pd.read_csv(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\PFShapefile\PF_GroundTruth_noFeatures.csv')
posFile=posFile.dropna()
x,y=posFile.X,posFile.Y


colname=[];val=[]
for r in np.arange(0,len(rasList)):
    ras=rasList[r]
    colname.append(rasList[r][:-4])
    param=[]
    for p in np.arange(0,len(x)):
        param.append(point_value(ras,(x[p],y[p])))
    val.append(param)
    
    
for c in np.arange(0,len(colname)):
    posFile[colname[c]]=val[c]
    
    
    
posFile.to_csv('./ClassifiedValuesWithParams.csv')

    
    
    