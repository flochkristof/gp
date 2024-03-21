import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from GP.models.RLSSGPModel import RLSSGPModel
from GP.models.GRADSGPModel import GRADSGPModel
from GP.models.SGPModel import SGPModel
import time

# Define tranining and testing datasets

train_x=np.random.rand(50, 4)
train_y=np.sin(train_x[:, 0]) + np.cos(train_x[:, 1]) + np.arctan(train_x[:, 2]) + np.exp(train_x[:, 3]) + torch.randn(train_x.shape[0]).numpy() * 0.2

# phase 1 - test data identicat to the training data 
test_x1=np.random.rand(50, 4)
test_y1=np.sin(test_x1[:, 0]) + np.cos(test_x1[:, 1]) + np.arctan(test_x1[:, 2]) + np.exp(test_x1[:, 3])


# phase 2 -  test data with some variables zeroed out to test how much the model forgets
test_x2=np.random.rand(100, 4)
test_x2[:,0]=np.zeros(100)
test_x2[:,1]=np.zeros(100)
#test_x2[:,2]=np.zeros(50)
#test_x2[:,3]=np.zeros(50)
test_y2=np.sin(test_x2[:, 0]) + np.cos(test_x2[:, 1]) + np.arctan(test_x2[:, 2]) + np.exp(test_x2[:, 3])

# phase 3 - information rich test data again
test_x3=np.random.rand(100, 4)
test_y3=np.sin(test_x3[:, 0]) + np.cos(test_x3[:, 1]) + np.arctan(test_x3[:, 2]) + np.exp(test_x3[:, 3])


# phase 4 - change the test data to see how the model adapts
test_x4=np.random.rand(400, 4)
test_y4=.7*np.sin(test_x4[:, 0]) + np.cos(test_x4[:, 1]-0.2) + np.arctan(test_x4[:, 2]) - np.exp(test_x4[:, 3])


test_x=np.concatenate((test_x1, test_x2, test_x3, test_x4), axis=0)
test_y=np.concatenate((test_y1, test_y2, test_y3, test_y4), axis=0)

# generate a number of GP models
# a GP is defined by three main parameters
# - confidence_level: confidence level for the recursive update
# - forgetting_factor: forgetting factor for the recursive update
# - batch_size: number of data points to be used for the batch update
LS_GP_datas = [{"num_of_inducing": 20, # update without retraining
             "confidence_level": 1,
             "forgetting_factor": 0.96,
             "batch_size": 5},
             {"num_of_inducing": 20, # update without retraining
             "confidence_level": 1,
             "forgetting_factor": 0.95,
             "batch_size": 5},
             {"num_of_inducing": 20, # update without retraining
             "confidence_level": 1,
             "forgetting_factor": 0.9,
             "batch_size": 10}
             ]

GRAD_GP_datas = [{"num_of_inducing": 20, # update without retraining
             "retrain_iter": 1,
             "batch_size": 10},
             {"num_of_inducing": 20, # update without retraining
             "retrain_iter": 1,
             "batch_size": 5},
             {"num_of_inducing": 10, # update without retraining
             "retrain_iter": 5,
             "batch_size": 1},
             {"num_of_inducing": 25, # update without retraining
             "retrain_iter": 5,
             "batch_size": 5}
             ]

GPs=[]

# reference SGP for comparison
GP=SGPModel(data_x=train_x, data_y=train_y, num_of_inducing=30, training_iter=200)
GP.train_model()
GPs.append(GP)

# pretrain the models
for i, GP_data in enumerate(LS_GP_datas):
    sGP=RLSSGPModel(data_x=train_x, data_y=train_y, num_of_inducing=GP_data["num_of_inducing"], training_iter=200)
    sGP.train_model()
    sGP.configure_update_method(batch_size=GP_data["batch_size"], confidence_level=GP_data["confidence_level"], forgetting_fator=GP_data["forgetting_factor"])
    GPs.append(sGP)


# pretrain the models
for i, GP_data in enumerate(GRAD_GP_datas):
    sGP=GRADSGPModel(data_x=train_x, data_y=train_y, num_of_inducing=GP_data["num_of_inducing"], training_iter=200)
    sGP.train_model()
    sGP.configure_update_method(batch_size=GP_data["batch_size"], retrain_iter=GP_data["retrain_iter"])
    GPs.append(sGP)

means = [[] for _ in range(len(GPs))]
vars = [[] for _ in range(len(GPs)+1)]
time_req = [[] for _ in range(len(GPs)+1)]


# run the estimation and compare the models

for i in range(test_x.shape[0]):
    for j, GP in enumerate(GPs):
        t0 = time.time()        

        # do prediction
        #if j==0:
        #    mean, var = GP.eval_gp(test_x[i,:].reshape(1,-1))
        #else:
        mean, var = GP.predict(test_x[i,:].reshape(1,-1))
        
        # save results
        means[j].append(mean.item())
        vars[j].append(var.item())

        # update the model
        if j!=0:
            GP.add_to_batch(test_x[i,:].reshape(1,-1), test_y[i])

        time_req[j].append(time.time()-t0)

# display some results
print("====== RESULTS ======")

for j, GP in enumerate(GPs):
    print(GP)
    print(f"Calculation times -- mean:{np.mean(time_req[j]):.5f} s, min:{np.min(time_req[j]):.5f} s, max:{np.max(time_req[j]):.5f} s")
    print(f"Predictions -- MSE: {np.mean((np.array(means[j])-test_y)**2):.5f}, maximal error: {np.max(np.abs(np.array(means[j])-test_y)):.5f}")
    print(f"Variance -- mean: {np.mean(vars[j]):.5f}, maximal variance: {np.max(vars[j]):.5f}")
    print("---------------------")


# plot the results
plt.figure()
for j, GP in enumerate(GPs):
    plt.plot(means[j], "-", label=str(GP))
plt.plot(test_y, label="Reference")
plt.legend()
plt.title("Predictions")
plt.show(block=False)

# plot the errors
plt.figure()
for j, GP in enumerate(GPs):
    plt.plot(np.abs(np.array(means[j])-test_y), "-", label=str(GP))
plt.legend()
plt.title("Errors")
plt.show(block=False)

## plot the variance
plt.figure()
for j, GP in enumerate(GPs):
    plt.plot(vars[j], "-", label=str(GP))
plt.legend()
plt.title("Variance")
plt.show(block=False)

# plot the calculation times
plt.figure()
for j, GP in enumerate(GPs):
    plt.plot(time_req[j], "-", label=str(GP))
plt.legend()
plt.title("Calculation times")
plt.show(block=True)
