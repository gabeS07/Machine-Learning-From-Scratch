# Dependencies
import numpy as np
import random
import matplotlib.pyplot as plt
# create plot
fig, ax = plt.subplots()
# Data (chosen arbitrarily)

features = np.array([1,1.5,2,2.5,3,3.5,4,4.5]) 
labels = np.array([1, 2, 2, 3, 3, 4, 4, 5])
ax.scatter(features, labels)
ax.set_xlabel('Features')
ax.set_ylabel('Labels')
# Square Trick

def square_trick(bias, weight, feature, label, learning_rate):

    y_pred = bias +weight*feature
    bias += learning_rate * (label - y_pred)
    weight += learning_rate *feature * (label - y_pred)
    return bias, weight

# Linear regression method

def linear_regression(features, labels, learning_rate, epochs):
   
    # start with random weight and bias
    bias = random.random()
    weight = random.random()
   
    # training loop
    for epoch in range(epochs):
        i = random.randint(0, len(features)-1) # choose random point
        x = features[i] # get feature and label of point
        y = labels[i]
        bias, weight = square_trick(bias, weight, x, y, learning_rate) # apply sqaure trick
    
    return bias, weight

bias, weight = linear_regression(features, labels, learning_rate=0.01, epochs=1000) # train model 
                                                                                    # to get weight and bias

# Prediction function with new weight and bias
def predict(feature):
    return bias + weight * feature

# plot the line of best fit
ax.plot(features, predict(features), color='red') 

plt.show()