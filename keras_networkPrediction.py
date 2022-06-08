from osgeo import gdal,gdalnumeric
import pandas as pd
import os,glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasClassifier

os.environ['PROJ_LIB'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share'

os.chdir(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\rasters_3m')
flist = glob.glob('*.tif')


newRasterfn1 = './PF_predictions/CNN_prediction.tif'

'''
Below are a lot of GDAL functions for getting the nodata value from the input rasters
and generating a new output raster with predicted PF values. 
'''

tif = gdal.Open(flist[0])
#get image metadata
band = tif.GetRasterBand(1)
bandarr = band.ReadAsArray()
nodat = band.GetNoDataValue()
geotransform = tif.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
cols = tif.RasterXSize
rows = tif.RasterYSize


driver = gdal.GetDriverByName('GTiff')
outRaster = driver.Create(newRasterfn1, cols, rows, 1, gdal.GDT_Float32,["COMPRESS=LZW"])
outband=outRaster.GetRasterBand(1)

outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
outRaster.SetGeoTransform(geotransform)
outRaster.SetProjection(tif.GetProjection())
outband = outRaster.GetRasterBand(1)

###
'''
First we will read in the rasters of the full study location
We will extract the data from each raster into a 1-D array and set the nodata values to np.nan
'''
snow=gdalnumeric.LoadFile(glob.glob('*snow*.tif')[0]).astype('float32').flatten();#snow[snow==nodat]=np.nan

chm = gdalnumeric.LoadFile(glob.glob('*chm*.tif')[0]).astype('float32').flatten();#chm[chm==nodat]=np.nan

da=gdalnumeric.LoadFile(glob.glob('*da*.tif')[0]).astype('float32').flatten();#da[da==nodat]=np.nan

aspect=gdalnumeric.LoadFile(glob.glob('*aspect*.tif')[0]).astype('float32').flatten();#aspect[aspect==nodat]=np.nan

elev = gdalnumeric.LoadFile(glob.glob('*elev*.tif')[0]).astype('float32').flatten();#elev[elev==nodat]=np.nan

slope=gdalnumeric.LoadFile(glob.glob('*slope*.tif')[0]).astype('float32').flatten();#slope[slope==nodat]=np.nan

tri=gdalnumeric.LoadFile(glob.glob('*tri*.tif')[0]).astype('float32').flatten();#tri[tri==nodat]=np.nan
tpi=gdalnumeric.LoadFile(glob.glob('*tpi*.tif')[0]).astype('float32').flatten();#tri[tri==nodat]=np.nan
curv = gdalnumeric.LoadFile(glob.glob('*curv*.tif')[0]).astype('float32').flatten();#curv[curv==nodat]=np.nan



'''
Now we need to combine the raster data into a dataframe
We will be later use the model to predict PF extent based on these data
'''
df_fullData = pd.DataFrame({'chm':chm,'aspect':aspect,'elev':elev,'tpi':tpi,'slope':slope,'tri':tri,'curv':curv,'da':da,'snow':snow})
df_fullData=df_fullData.dropna()

X_FULL=df_fullData[['chm','aspect','elev','slope','curv']]
#X_FULL=df_fullData[['chm','aspect','elev','tpi','slope','tri','curv','snow','da']]

##############################################
'''
Now we need to load in the training data set, which has the same 
parameters as those in the raster dataframe, but we also have a binary column with
PF presence/absence. 
'''
training_df = pd.read_csv(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\rasters_3m\ClassifiedValuesWithParams.csv')
target = training_df['PF']
#training_data= training_df[['chm','aspect','elev','tpi','slope','tri','curv','da','snow']]
training_data= training_df[['chm','aspect','elev','slope','curv']]


X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2,random_state=109) # 70% training and 30% test

'''
Now we will train and test the model using the training and target datasets
'''

# MLP with manual validation set


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# split into input (X) and output (Y) variables
X = training_data
Y = target
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
# create model
model = Sequential()
model.add(Dense(12, input_dim=len(training_data.columns), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=1)


cnn = model.predict(X_FULL)
cnn_pred=cnn.reshape(bandarr.shape)

outband.SetNoDataValue(-9999)
outband.WriteArray(cnn_pred)


outband.FlushCache()
outRaster=None
tif=None