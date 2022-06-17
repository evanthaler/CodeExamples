import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def bootstrap_linear(x,y,numiter,pct,vals):
    '''
    bootstrap_linear(x,y,numiter,pct,vals):
        x: array of values
        y: array of values with len(x)
        numiter: number of bootstrap iterations
        pct: fraction of values to grab each iteration (as decimal)
        vals: tuple of 5 lists for storing linregress output (this doesn't need to be changes)
    '''
    
    n_size = np.int(len(x)*pct)
    
    
    plt.figure(figsize=(5,5))
    
    for i in range(numiter):
        # prepare train and test sets
        #x_train,x_test,y_train,y_test=train_test_split(soci_scaled,hznclass,test_size=n_size,random_state=np.random.RandomState())
        idx = np.random.choice(np.arange(len(x)),n_size, replace=True)
        x_samp = x[idx]
        y_samp = y[idx]
        
        #define regression parameters for subset
        slope,intercept,rval,pval,stderr= linregress(x_samp,y_samp)
        ##Store regression parameters for subset
        #sl.append(slope);intcp.append(intercept);r.append(rval**2);p.append(pval);sderr.append(stderr)
        vals[0].append(slope);vals[1].append(intercept);vals[2].append(rval**2);
        vals[3].append(pval);vals[4].append(stderr)
        #plot subset
        plt.plot(x_samp,slope*x_samp+intercept,'-',color='gray',alpha=0.1)
    
    plt.plot(np.sort(x),np.mean(vals[0])*np.sort(x)+np.mean(vals[1]),'-r',linewidth=2)
    plt.ylabel('y');plt.xlabel('x')
    plt.show()



############################
## RUN FUNCTION############
#############################

df = pd.read_csv(r'path\to\file.csv')
df=df.dropna()
df=df.reset_index()

#Here I make a tuple to store the output of linregress for each iteration
#0th list = slopes;1st list = intercepts; 2nd list = r-squared values; 3rd list = p values; 4th list = standard errors of the slopes
##There must be a better way to do this############# 

vals_to_save=([],[],[],[],[])
# in this example, I am generating regressions for df.X and df.Y
bootstrap_linear(df.X,df.,500,0.20,vals_to_save)