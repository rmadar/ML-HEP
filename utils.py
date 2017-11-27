import numpy             as     np
import matplotlib.pyplot as     plt
import matplotlib        as     mpl
from   sklearn.metrics   import roc_curve

def compare_4top_ttV_distributions(data_4top,data_ttV,variables,myfigsize=(20,20)):
    plt.figure(figsize=myfigsize)
    i=0
    for var in variables:
        if (i>9):
            break
        i=i+1
        plt.subplot(3,3,i)
        plt.hist(data_4top[var], bins=100, histtype='step', linewidth=2.5,label='tttt')
        plt.hist(data_ttV[var] , bins=100, histtype='step', linewidth=2.5,label='ttV' )
        plt.legend()
        plt.xlabel(var)
    return


def plot_perf_randomforest_vs_ntree(rf_regre,trainX,trainY,testX,testY):

    plt.figure(figsize=(15,7.5))
    for sX, sY, label in [(trainX, trainY, 'Train'), (testX, testY, 'Test')]:
        predictions = np.zeros(len(sX))
        curve_rms   = []
        curve_mean  = []

        predictions = np.array( [tree.predict(sX) for tree in rf_regre.estimators_] )
        for i in range(0,rf_regre.n_estimators):
            if (i==0):
                Ypred=predictions[0]
            else:
                prediction_i = predictions[0:i]
                Ypred = np.average(prediction_i,axis=0)

            curve_rms .append( np.sqrt(np.var(Ypred-sY)) )
            curve_mean.append( np.abs(np.average(Ypred-sY)) )

        plt.subplot(221)
        plt.plot(curve_mean,label=label)
        plt.xlabel('Number of Trees')
        plt.ylabel('Biais $|\mu\,[Y-f(X)]|$')
        plt.legend()

        plt.subplot(222)
        plt.plot(curve_rms,label=label)
        plt.ylabel('Precision $\sigma\,[Y-f(X)]$')
        plt.xlabel('Number of Trees')
        plt.semilogx()
        plt.legend()

    return;


def plot_prediction_vs_truth(regressor,testX,testY):

    ypred = regressor.predict(testX)
    plt.figure(figsize=(15,5))
    
    plt.subplot(131)
    plt.xlabel('Y')
    plt.hist(testY, bins=np.linspace(0,6000,60), histtype='step', linewidth=2.5, label='Truth')
    plt.hist(ypred, bins=np.linspace(0,6000,60), histtype='step', linewidth=2.5, label='Prediction')
    plt.legend()

    plt.subplot(132)
    plt.xlabel('Y_{pred} - Y_{truth}')
    plt.hist(ypred-testY, bins=100, histtype='step', linewidth=2.5)

    plt.subplot(133)
    plt.xlabel('Y_{truth}')
    plt.ylabel('Y_{pred} - Y_{truth}')
    plt.plot(ypred,ypred-testY,'o')

    return


def plot_prediction_ndim(regressor,testX,testY,trainX,trainY,nvar=10):

    ypred=regressor.predict(testX)
    plt.figure(figsize=(15,15))

    i=0
    for varname in testX.columns[:nvar-1]:
        i=i+1
        x=testX[varname]
        plt.subplot(3,3,i)
        plt.plot(trainX[varname],trainY,'.',alpha=0.7, label='Data')
        plt.plot(testX[varname],ypred,'ro',alpha=0.1,label='Model')
        plt.ylabel('Y')
        plt.xlabel(varname)
        if (i==1):
            plt.legend()

    return


def plot_roc_curves(Xsig, Xbkg, variables, regressors):
    """
    Xsig      : dataframe object containing signal events
    Xbkg      : dataframe object containing background events
    variables : array of variable name that will be use to plot ROC curve
    regressors: array of [reg_method,name] where reg_method is regression and name is its legend name
    """ 
    # Prepare the full dataset
    sig_labelled = Xsig
    sig_labelled['isSig'] = True
    bkg_labelled = Xbkg
    bkg_labelled['isSig'] = False
    X = pd.concat( [sig_labelled,bkg_labelled] )

    # Produce the plots
    plt.figure(figsize=(10,8))
    for var in variables:
        fake,eff,_= roc_curve(X['isSig'],X[var])
        plt.plot(fake,eff,label=var)
    
    Xeval=X.drop('isSig',axis=1)
    Xeval.head()
    for reg,name in regressors:
        fake,eff,_= roc_curve(X['isSig'],reg.predict(Xeval))
        plt.plot(fake,eff,label=name)
    
    plt.xlabel('ttV efficiency')
    plt.ylabel('4top efficiency')
    plt.legend()
    
    return
