import math
import numpy as np

''' INITIALIZING WEIGHTS '''

alpha = 0.3
epochs = 1000

w0 =  0.3640010671864496
#w0 is the bias , here we consider the bias to be 1 whose weight is random
w1 = 0.4532866437112887


print("--Initial weights--\n ")
print("bias = ",w0,"w1 = ", w1)

del_w0 = 1
del_w1 = 1


''' SPECIFYING TRAINING DATA '''

train_data_temp = [[1, 0], [1, 1]]
train_data = np.asarray(train_data_temp)

# NOT Output Function is given by the following
op_f = [1,0]


op = np.asarray(op_f)

# y = w0 + w1*x1 

''' TRAINING PROCESS '''

for i in range(epochs):
    j = 0
    for x in train_data:
        res = w0*x[0] + w1*x[1] 

        if (res >= 0.5):
            act = 1
        else:
            act = 0
        
        # act = 1/(1+math.exp(-x))

        err = op[j] - act

        del_w0 = alpha*x[0]*err
        del_w1 = alpha*x[1]*err
      
        w0 = w0 + del_w0
        w1 = w1 + del_w1
      

        j = j + 1
        # print("epoch ",i+1,"error = ",err)
        # print(del_w0,del_w1,del_w2)

print("\n--Final weights-- ")
print("bias = ",w0,"w1 = ",w1)

''' SPECIFYING TESTING DATA AND TESTING PROCESS '''

test_data = [[0.2], [ 0.3], [0.4], [0.9], [0.7], [0.56], [0.97], [0.5123]]
test_op = [1, 1, 1, 0, 0, 0, 0, 0]
total = len(test_data)
correct = 0

for i in range(len(test_data)):
    temp = w0 + w1*test_data[i][0] 
    #this is the binary sigmoidal activation function 
    # ans = 1/(1 + math.exp(-temp))

    #the bipolar sigmoidal activation function
    # ans = (1-math.exp(-temp))/(1+math.exp(-temp))

    #the code depicts only the step activation function which must be included
    
    if ( temp >= 0.5 ):
        ans = 1
    else:
        ans = 0

    print("\nx1 - ",test_data[i][0])
    print("Predicted - ",ans,",Actual - ",test_op[i])

    if ( ans == test_op[i] ):
        correct = correct + 1

print ( "\nAccuracy - ", correct/total)