from osgeo import gdal
from gdalconst import GA_ReadOnly
import os
import numpy as np

##These two lines are only necessary if your machine has pain in the ass permissions like mine
os.environ['PROJ_LIB'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share'


def clip_resample(mskRas,inRas,warpRas,outRas,outRas2):

    '''
    Here I'm generating three output rasters. The first is a reprojection of the input raster to the projection of the mask raster. 
    The second is the square clip of a raster, and the third clips to raster to only real value pixels.
    Then, I delete the first and second rasters.
    THIS IS A REALLY INEFFECIENT WAY TO CLIP TO NODATA PIXELS. PLEASE UPDATE THE SCRIPT IF YOU DECIDE TO MAKE IT MORE EFFICIENT.


    Parameters
    ----------
    maskRas : raster
        full path to the mask raster (this raster will be the dimension and resolution of the desired output)
    inRas : raster
        full path to the raster which will be clipped
    outRas : raster
        full path to desired output raster (include extension (.tif)). This is a file that will be deleted after it is created.
        I have just made this raster to further clip the nodata regions to match the mask file
    warpRas : raster
        full path to desired output warp raster (include extension (.tif)). This is a file that will be deleted after it is created.
        I have just made this raster to further clip the nodata regions to match the mask file
        
    outRas2: raster
        full path to the desired output raster(include extension (.tif)). This is the final output file that has is further masked to the
        nodata values of the input mask. This setp allows output rasters that are non-rectangular.
        
        

    Returns
    -------
    None.

    '''
    
    ##This is the raster that'll be the mask (clipping extent and correct resolution)
    maskRas = gdal.Open(mskRas, GA_ReadOnly)
    meta = maskRas.GetMetadata() 
    projection=maskRas.GetProjectionRef()
    geoTransform = maskRas.GetGeoTransform()
    ##Get upper left x value
    minx = geoTransform[0]
    ##Get lower right x value
    maxx = minx + geoTransform[1] * maskRas.RasterXSize
    ##Get upper left y value
    maxy = geoTransform[3]
    ##Get lower right y value
    miny = maxy + geoTransform[5] * maskRas.RasterYSize
    
    maskBand = maskRas.GetRasterBand(1)
    #get nodata value for the band
    nodat = maskBand.GetNoDataValue()
    if nodat is None:
        nodat=0.
    #read band values into array
    maskBand = np.array(maskBand.ReadAsArray()).astype(float)
    #Flatten array for easy comparison with input raster
    mskbandFlatten = maskBand.ravel()
    
    
    
    ###The input here is the raster to be converted to the mask coordinate system and clipped
    
    output_warp = warpRas
    
    ##This is the function to reproject the input raster into the mask raster coordinate system
    warp = gdal.Warp(output_warp,inRas,dstSRS=projection)
    
    
    ##This is the main function to do the clipped and resampling
    out=gdal.Translate(outRas,warp,format='GTiff',projWin=[minx,maxy,maxx,miny],outputSRS=projection,xRes=geoTransform[1],yRes=np.positive(geoTransform[-1]))
    
    outband =out.GetRasterBand(1)
    #outband band values into array
    outband = np.array(outband.ReadAsArray()).astype(float)
    #Flatten array for easy comparison with flattened mask raster
    outbandFlatten = outband.ravel()
    #Set values to nan where mask band is nodata
    outbandFlatten[mskbandFlatten==nodat]=np.nan
    #reshape the output array to the mask raster shape
    outbandFlatten= outbandFlatten.reshape(outband.shape)
    
    #######
    #Below are a bunch of gdal functions to create the output file
    #There shouldn't be a need to change any of the lines
    #######
    
    pixelWidth = geoTransform[1]
    pixelHeight = geoTransform[5]
    cols = maskRas.RasterXSize
    rows = maskRas.RasterYSize
    

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(outRas2, cols, rows, 1, gdal.GDT_Float32,["COMPRESS=LZW"])
    outband=outRaster.GetRasterBand(1)
    
    outRaster.SetGeoTransform((minx, pixelWidth, 0, maxy, 0, pixelHeight))
    outRaster.SetGeoTransform(geoTransform)
    outRaster.SetProjection(maskRas.GetProjection())
    #outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(0)
    outband.WriteArray(outbandFlatten)
    
    
    outband.FlushCache()
    
    #Close all of the open raster files
    maskRas=None
    data=None
    out=None
    outRaster=None
    warp=None
    
    #Here we are going to delete that first output file that we don't really need anymore
    os.remove(outRas)
    os.remove(warpRas)
    
####Run the function on some rasters
mskRas = r'C:/Users/361045/Documents/projects/ngee/imagery/t47/may242021/20210524_214038_27_222b_3B_AnalyticMS_SR_harmonized_clip.tif'
inRas = r'C:/Users/361045/Documents/projects/ngee/arcticDEM/sites/t47_elev.tif'
outRas = r'C:\Users\361045\Desktop\outTest.tif'
warpRas = r'C:\Users\361045\Desktop\WarpTest.tif'
outRas2 = r'C:\Users\361045\Desktop\outTest2.tif'
#clip_resample(mask,input,output1,output2)
clip_resample(mskRas,inRas,warpRas,outRas,outRas2)