from osgeo import gdal
import glob,os
import numpy as np

'''
Function for calculating the soil organic carbon index (SOCI) from Thaler et al., 2019 (SSSAJ)

'''

def get_soci(wd,inras,outras):
    '''
    

    Parameters
    ----------
    wd : string
        full path to directory with images
    inras : geotiff
        input image as geotiff
    outras : geotiff
        output SOCI raster

    Returns
    -------
    None.

    '''
    tif = gdal.Open(inras)
    #get image metadata
    meta = tif.GetMetadata()
    if (tif.RasterCount >1): #here we are making sure that tif is a true color image.
        #read in band 1 (change number in GetRasterBand() to read in different bands)
        blue = tif.GetRasterBand(1)
        green= tif.GetRasterBand(2)
        red = tif.GetRasterBand(3)
        #get nodata value for the band
        nodat = blue.GetNoDataValue()
        if nodat is None:
            nodat=0.
        #read band values into array
        blue = np.array(blue.ReadAsArray()).astype(float)
        green =  = np.array(green.ReadAsArray()).astype(float)
        red =  = np.array(red.ReadAsArray()).astype(float)
 

        soci = blue/(red*green)



            

        
    
    
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
    outras=f[:-4]+'soci.tif'
    getSnowBlue(wd,inras,outras)
