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

