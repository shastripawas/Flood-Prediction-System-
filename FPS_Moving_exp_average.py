import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/home/pawas/Downloads/rainfall in india 1901-2015.csv')
S=pd.unique(df['SUBDIVISION'])
St=np.array(S)
#Applying a loop for all the states present in the dataset
for i in range(np.size(St)):
    #Cleaning the data and initialising variables
    df1=df[df['SUBDIVISION']==St[i]]                #obtaining dataset of a particular state
    df1=df1.dropna()                                #dropping the null values
    L=df1['ANNUAL']
    count=0
    Y=np.array(L)
    B=df1['YEAR']
    X=np.array(B)
    m=np.size(Y)

    # Function to generate an exponential fit curve using alpha as a parameter
    def exp_smoothing(x,y,alpha):
        m=np.size(x)
        y_pred=np.zeros(m)
        y_pred[0]=y[0]
        for i in range(1,m):
            y_pred[i]=alpha*x[i]+(1-alpha)*y[i-1]
        return y_pred

    # Extending the above curve to predict value for the given year
    def exp_smoothing_extn(x,y,year,alpha):
        m=np.size(x)
        X=[i for i in range(2016,(year+1))]
        X=np.array(X)
        X=np.concatenate((x,X))
        y_pred=np.zeros(np.size(X))
        y_pred[0:m]=exp_smoothing(x,y,alpha)
        for i in range(m,np.size(y_pred)):
            y_pred[i]=alpha*X[i]+(1-alpha)*y_pred[i-1]
        return y_pred
    # RMS Error for the obtained curve
    def exp_smooth_error(x,y,alpha):
        n=np.size(y)
        y_pred=exp_smoothing(x,y,alpha)
        err=(np.sum((y_pred-y)**2)/np.size(y))**(0.5)
        return err
    #Finding alpha with minimum rms error
    def find_alpha(x,y):
        dict={exp_smooth_error(x,y,alpha):alpha for alpha in np.arange(0.0,1.0,0.01)}
        L=[exp_smooth_error(x,y,alpha) for alpha in np.arange(0.0,1.0,0.01)]
        x=min(L)
        return dict[x]
    alpha=find_alpha(X,Y)


    #Finding the rolling mean
    mean=np.zeros(m)
    mean[0]=Y[0]
    for j in range(m):
        count=count+Y[j]
        mean[j]=count/(j+1)
    err=(np.sum((mean-Y)**2))/m
    print('Training set error for moving average ' + St[i]+ '  %f' %((err)**0.5))
    print('Training set error for exponential smothing ' + St[i]+ '  %f' %((exp_smooth_error(X,Y,alpha))))

    #Predictions for the year 2021
    x1=2021
    si=x1-2015
    In_pre=np.zeros(si)
    In_pre[0]=mean[j]
    for l in range(1,si):
        count=count+In_pre[l-1]
        In_pre[l]=count/(j+l+1)
    print('The predicted rainfall for the year 2021 in ' + St[i]+ ' using moving average is: %0.2f' %(In_pre[si-1]))
    Y2=exp_smoothing_extn(X,Y,2021,alpha)
    print(('The predicted rainfall for the year 2021 in ' + St[i]+ ' using exponential average is: %0.2f' %(Y2[-1])))
    print('\n')

    #Plotting the curves
    L2=[i for i in range(2016,2022)]
    L2=np.array(L2)
    X1=np.concatenate((X,L2))
    Y1=np.concatenate((mean,In_pre))
    plt.plot(X,Y,label='Actual data')
    plt.plot(X1,Y2,label="exp_alpha=%f" %alpha)
    plt.plot(X1,Y1,label='Rolling Mean')
    plt.xlabel('YEAR')
    plt.ylabel('ANNUAL RAINFALL')
    plt.title(St[i])
    plt.legend()
    plt.show()
