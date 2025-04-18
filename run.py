import mlp.mlp_model as mlp
import gendata.gendata as gd
import nn.utils as utils
import numpy as np
import matplotlib.pyplot as plt


INPUNT_DIM = 2


# model = mlp.MLP()

# model.add(mlp.Heaviside(INPUNT_DIM, 2))

givemedata = gd.ClusteredClasses2D()

data = givemedata.getdata()
print(data)

training_data, testing_data = utils.split_data_8_2(data)

model.train(training_data)
model.evaluate(testing_data)

plt.scatter(training_data[0, :], training_data[1, :], c=training_data[2, :])
plt.show()