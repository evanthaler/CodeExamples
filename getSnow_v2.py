import numpy as np
import os,glob
from osgeo import gdal
import ogr, osr
##Setting environmental variables for GDAL--might not be necessary for some users
##These lines can probably be commented out on most machines
os.environ['PROJ_LIB'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share'
import matplotlib.pyplot as plt
from scipy.signal import argrelmin,argrelmax,find_peaks
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from scipy.ndimage.filters import convolve1d

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def getSnowBlue (wd,inras,outras):
    #Change to directory where images are located
    os.chdir(wd)
    
   
    #open image
    tif = gdal.Open(inras)
    #get image metadata
    meta = tif.GetMetadata()
    if (tif.RasterCount >1): #here we are making sure that tif is a true color image.
        #read in band 1 (change number in GetRasterBand() to read in different bands)
        blue = tif.GetRasterBand(1)
        #get nodata value for the band
        nodat = blue.GetNoDataValue()
        if nodat is None:
            nodat=0.
        #read band values into array
        blue = np.array(blue.ReadAsArray()).astype(float)
        blue[blue==nodat]=np.nan
        ##remove nans from blue array and flatten to 1d
        bluef =blue[~np.isnan(blue)].ravel()
        #need to smooth array to remove noise--this will help us find the true peaks
        bluef_sm = smooth(bluef,window_len=50)
        #bluef_sm = np.sort(bluef_sm)
        
        
        # bluef_kde = gaussian_kde(bluef_sm)
        # bluebins = np.linspace(np.min(bluef),np.max(bluef),50)
        # bkdepdf = bluef_kde(bluebins)
        
        # plt.figure(figsize=(3,3))
        # plt.plot(bluebins,bkdepdf,'-k')
        # plt.xlabel('blue');plt.ylabel('kde')
        # plt.show()
        


        #Find the index of the first local minimum greater than the mean value
        #lmin=argrelmax(bluehist[0])[0][0] ##using binned histogram values
        if (bluef_sm[bluef_sm>np.mean(bluef_sm[bluef_sm>0])][0]) > (bluef_sm[bluef_sm>np.mean(bluef_sm[bluef_sm>0])][-1]):
            print('inverted array: So, we need to grab the last value in the array ')
            lmin=argrelmin(bluef_sm[bluef_sm>np.mean(bluef_sm[bluef_sm>0])])[0][-1]
        #Index blue at local min--this will be the threshold for the image.
        #bluethresh = bluehist[1][lmin+1] ##using binned histogram values
            bluethresh = bluef_sm[bluef_sm>np.mean(bluef_sm)][lmin]
            print(bluethresh)
        else:
            lmin=argrelmin(bluef_sm[bluef_sm>np.mean(bluef_sm[bluef_sm>0])])[0][0] 
            bluethresh = bluef_sm[bluef_sm>np.mean(bluef_sm)][lmin]    
            print(bluethresh)
            
        #generate an array of zeros with the input raster dimensions
        snowBLUE=np.zeros(blue.shape)
        #set all pixels predcited to be snow to a value of 1
        snowBLUE=np.where((blue>bluethresh),snowBLUE+1,snowBLUE)
        
    
    
        #######
        #Below are a bunch of gdal functions to create the output file
        #There shouldn't be a need to change any of the lines
        #######
        
        geotransform = tif.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        cols = tif.RasterXSize
        rows = tif.RasterYSize
        
    
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(outras, cols, rows, 1, gdal.GDT_Float32,["COMPRESS=LZW"])
        outband=outRaster.GetRasterBand(1)
        
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRaster.SetGeoTransform(geotransform)
        outRaster.SetProjection(tif.GetProjection())
        #outband = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(0)
        outband.WriteArray(snowBLUE)
        
        
        outband.FlushCache()
        outRaster=None
        tif=None
    else:
        print('this is a single band raster and probably not a true color image--moving on to next raster')

#Set working directory
wd=r'C:\Users\361045\Desktop\snowTest'
#Get list of tif files in working directory
flist=glob.glob(wd+'\\'+'*.tif')#for macOS/linux,might need to change '\\' to '/'
for f in flist: ##we're going to loop through the tifs in the directory and calculate snow on the true color images
    
    inras=f
    #generate path for output file...here we are just throwing the new snow rasters in the working directory
    outras=f[:-4]+'_snow.tif'
    getSnowBlue(wd,inras,outras)






