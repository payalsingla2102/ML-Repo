#importing the python libraries
import numpy as np
import math 

" --- INITIALIZING THE VALUES ---"
ip = [0.7, 0.8]
bias_val = 1
bias_weight = 0.9
ip_weight = [0.2, 0.3]
print("--inputs--")
print("x1 = ", ip[0], ",x2 = ", ip[1])
print("\n--weights of inputs--")
print("w1 = ", ip_weight[0],",w2 = ", ip_weight[1])
print("\n--Bias and weight--")
print("bias  = ", bias_val," ,weight = ", bias_weight)



#defining the activation functions

def bin_sigm(x):
    y= 1/(1+math.exp(-x))
    return y

# defining the bipolar sigmoidal
def bip_sigm(x):
     y = (1- math.exp(-x))/(1+ math.exp(-x))
     return y


def perceptron(ip, bias_val, bias_weight, ip_weight):
    #calculating the weight * input
    sum_ip = 0
    for i in range (len(ip)):
        sum_ip = sum_ip + ip[i]*ip_weight[i]

    # y = weight * input + bias*bais_weight
    per_out = sum_ip + bias_val *bias_weight

    #caluculaton for binary sigmoidal activation function
    y_bin = bin_sigm(per_out)

    #calculation for bipolar sigmoidal activation function
    y_bip = bip_sigm(per_out)
    print("\n--Value with Binary sigmoidal function--")
    print("y = ",y_bin)
    print("\n--Value with Bipolar sigmoidal function--")
    print("y = ", y_bip)


    print("threshold  = 0.7")
    if y_bip >= 0.7:
        #y_bin>=0.7
        out2= 1
        #out1 =1
    else:
        out2 = 0
        #out1 =0

    
    print("\n--OUTPUT OF THE NETWORK--")
    print("with binary output= ",out2 )
   # print("with binary output= ",out1 )


# calling the function
perceptron(ip, bias_val, bias_weight, ip_weight)












