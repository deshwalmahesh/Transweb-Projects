import numpy as np
import pandas as pd



def transform_data(data,t_type=0,m='min'):
    '''
    A t_type to transform data in Gaussian which is not Gaussian in nature using different techniques
    param:
        data: input data in the form of numpy array or pandas series
        t_type: transformation type 
                options:   int {
                                0: Square Root
                                1: Normalization
                                2: Sigmoid
                                3: Cube Root
                                4: Normalized Cube Root
                                5: Log
                                6: Log Max Root
                                7: Normalized Log
                                8: Normalized Log Max Root
                                9: Hyperbolic Tangent
                                10: 
                            }
    out:
        transformed data 
    '''

    def normalize_(data):
        upper = data.max()
        lower = data.min()
        return (column - lower)/(upper-lower)
    
    def sigmoid_(data):
        e = np.exp(1)
        return 1/(1+e**(-data))

    def log_(data):
        if data.min()>0:
            return np.log(data)
        else:
            return np.log(data+1)
    
    
    if t_type==0:
        return np.sqrt(data)

    if t_type==1:
        return normalize_(data) # normalize
    
    elif t_type==2:
        return sigmoid_(data) # sigmoid
    
    elif t_type==3:
        return data**(1/3) # cube root

    elif t_type==4:
        return normalize_(data**(1/3)) # normalized cube root

    elif t_type==5:
        return log_(data) # log

    elif t_type==6:
        return data**(np.log(data.max())) # log-max-root 

    elif t_type==7:
        return normalize_(log_(data)) # normalized log

    elif t_type==8:
        return normalize_(data**(np.log(data.max()))) # normalized log-max-root

    elif t_type==9:
        return np.tanh(data) # hyperbolic tangent
    
    elif t_type==10:
        return data.rank(method=m).apply(lambda x: (x-1)/len(data)-1)

    else:
        print('No Suitable t_type Specified. Returning Data')
        return(data) 
    

print(transform_data([1,2,3],t_type=2))
