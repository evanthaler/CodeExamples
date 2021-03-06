from osgeo import gdal,gdalnumeric
import pandas as pd
import os,glob
import numpy as np
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF,RationalQuadratic
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

os.environ['PROJ_LIB'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\361045\Anaconda3\envs\pygeo\Library\share'


def MLClassify(wd,outras,trainCSV,method='ERF'):
    '''
    

    Parameters
    ----------
    wd : string
        Full path to directory where input rasters are stores
    outras : string
        path and name of output prediction raster
    method : string, optional
        Classification method for prediction. 'ERF' = Extra random forest; 'SVM' = support vector machine; 
        'GPC' = gaussian processes; 'ADA' = adaptive boost 
        The default is 'ERF'.
    trainCSV: string
        Full path to csv file with training data. This file will have data from each of the rasters in the wd

    Returns
    -------
    None.

    '''
    os.chdir(wd)
    flist = glob.glob('*.tif')
    
    
    newRasterfn1 = outras[:-4]+'_'+method+'.tif'
    
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
    aspect=gdalnumeric.LoadFile(glob.glob('*aspect*.tif')[0]).astype('float32').flatten();aspect[aspect==nodat]=np.nan
    snow=gdalnumeric.LoadFile(glob.glob('*snow*.tif')[0]).astype('float32').flatten();snow[np.isnan(aspect)]=np.nan
    chm = gdalnumeric.LoadFile(glob.glob('*chm*.tif')[0]).astype('float32').flatten();chm[np.isnan(aspect)]=np.nan
    
    elev = gdalnumeric.LoadFile(glob.glob('*elev*.tif')[0]).astype('float32').flatten();elev[np.isnan(aspect)]=np.nan
    slope=gdalnumeric.LoadFile(glob.glob('*slope*.tif')[0]).astype('float32').flatten();slope[np.isnan(aspect)]=np.nan
    curv = gdalnumeric.LoadFile(glob.glob('*curv*.tif')[0]).astype('float32').flatten();curv[np.isnan(aspect)]=np.nan
    ndvi = gdalnumeric.LoadFile(glob.glob('*ndvi*.tif')[0]).astype('float32').flatten(); ndvi[np.isnan(aspect)]=np.nan
    
    ##Add a variable that is just random noise---this will help us understand if the model is capturing
    ##physically meaningful correlations
    rand = np.random.rand(len(ndvi))
    
    
    '''
    Now we need to combine the raster data into a dataframe
    We will later use the model to predict PF extent based on these data
    '''
    df_fullData = pd.DataFrame({'chm':chm,'aspect':aspect,'elev':elev,'slope':slope,'curv':curv,'snow':snow,'ndvi':ndvi,'rand':rand})
    df_fullDataNoDat=df_fullData.dropna()
    
    X_FULL=df_fullDataNoDat[['chm','aspect','elev','slope','curv','snow','ndvi']]
    
    ##############################################
    '''
    Now we need to load in the training data set, which has the same 
    parameters as those in the raster dataframe, but we also have a binary column with
    PF presence/absence. 
    '''
    training_df = pd.read_csv(trainCSV)
    training_df = training_df.dropna()
    target = training_df['PF']
    ##Add a variable that is just random noise---this will help us understand if the model is capturing
    ##physically meaningful correlations
    trnRand = np.random.rand(len(target))
    training_df['rand']=trnRand
    
    training_data= training_df[['chm','aspect','elev','slope','curv','snow','ndvi']]
    
    
    X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.3) # 70% training and 30% test
    
    '''
    Now we will train and test the model using the training and target datasets
    '''
    clf_erf = ExtraTreesClassifier(n_estimators=100)
    clf_erf.fit(X_train, y_train)
    y_pred_erf=clf_erf.predict(X_test)
    print("ExtraRF Accuracy: ",metrics.accuracy_score(y_test,y_pred_erf))
    #feature_imp = pd.Series(clf_erf.feature_importances_,index=training_data.columns).sort_values(ascending=False)
    objects = list(training_data.columns)
    
    
    
    #Impurity based importance
    importances = clf_erf.feature_importances_
    forest_importances = pd.Series(importances, index=objects)
    std = np.std([tree.feature_importances_ for tree in clf_erf.estimators_], axis=0)
    
    fig, ax = plt.subplots(figsize=(5,5))
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    # plt.savefig(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\Figs\featureImportance.jpg',dpi=300)
    
    
    # ##Permutation based importance
    # result = permutation_importance(
    #     clf_erf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    # )
    
    # forest_importances = pd.Series(result.importances_mean, index=objects)
    # fig, ax = plt.subplots(figsize=(5,5))
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # plt.ylim(bottom=0)
    # fig.tight_layout()
    # plt.show()
    
    #Create a svm Classifier
    clf_svm = svm.SVC(kernel="linear") # Linear Kernel
    #Train the model using the training sets
    clf_svm.fit(X_train, y_train)
    svm_pred = clf_svm.predict(X_test)
    print('SVM accuracy: ',metrics.accuracy_score(y_test,svm_pred))
    
    
    
    ##GaussianProcess Classifier
    
    rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpc = GaussianProcessClassifier(kernel=rbf)
    
    gpc.fit(X_train, y_train)
    gpc_pred =gpc.predict(X_test)
    gpc.score(X_train,y_train)
    print('GPC accuracy: ',metrics.accuracy_score(y_test,gpc_pred))
    
    
    ##adaptive boost classifier
    ada= AdaBoostClassifier(n_estimators=100)
    ada.fit(X_train,y_train)
    ada_pred = ada.predict(X_test)
    adascores = cross_val_score(ada, X_train, y_train, cv=5)
    print('ADA accuracy: ',adascores.mean())
    
    '''
    Now let's use the model to predict PF using the full raster dataset
    change the model before .predict(X_FULL) to use different ML model
    '''
    if method == 'ERF':
        classifier = clf_erf
    elif method == 'SVM':
        classifier = clf_svm
    elif method == 'GPC':
        classifier = gpc
    elif method == 'ADA':
        classifier = ada
        
    y_full_pred=classifier.predict(X_FULL)
    
    
    '''
    Now we can write the prediction raster to a new tif
    First we need to fill in the places in y_full_pred where we removed nans from the original input dems, which were compiled
    into df_fullData. we'll just be adding NaNs where the NaNs were in the original input data.
    '''
    
    
    def add_nans(x,y):
        lst = []
        index = 0
        for val in y:
            if np.isnan(val):
                lst.append(np.nan)
            else:
                lst.append(x[index])
                index +=1
    
        return np.array(lst)
    
    #need to reshape prediction array to match the raster domain
    y_fullpredResize = add_nans(y_full_pred,aspect)
    y_full_pred=y_fullpredResize.reshape(bandarr.shape)
    
    
    outband.SetNoDataValue(-9999)
    outband.WriteArray(y_full_pred)
    
    
    outband.FlushCache()
    outRaster=None
    tif = None
    
    
MLClassify(r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\rasters_3m',
           r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\rasters_3m\PF_predictions\PF_predictions.tif',
           r'C:\Users\361045\Documents\projects\ngee\machineLearningData\t47\rasters_3m\ClassifiedValuesWithParams.csv',
           'ERF')


'''
 MLClassify(wd,outras,trainCSV,method='ERF')

 Parameters
 ----------
 wd : string
     Full path to directory where input rasters are stored
 outras : string
     path and name of output prediction raster
 method : string, optional
     Classification method for prediction. 'ERF' = Extra random forest; 'SVM' = support vector machine; 
     'GPC' = gaussian processes; 'ADA' = adaptive boost 
     The default is 'ERF'.
 trainCSV: string
     Full path to csv file with training data. This file will have data from each of the rasters in the wd

 Returns
 -------
 None.

'''

