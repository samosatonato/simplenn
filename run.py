import miniflow as mf


"""
This is a simple example of how to use the MiniFlow library to create a fully self contained neural network.
"""

""" Create an empty workbench """
workbench = mf.Workbench()
# Creates empty module container


""" Add tools to the workbench """
workbench.add(mf.layers.Dense(2, 2))
workbench.add(mf.activations.ReLU())
workbench.add(mf.layers.Dense((2, 2), 'softmax'))
# Adds modules - layers and activations to the container


""" Build the machine """
machine = workbench.build()
# Builds the neural network model using the provided modules


""" Preprocess the input data """
refinery = mf.Refinery()
x, y = None
# Preprocess the data


""" Train the model """
lossfunction = mf.lossfunctions.MSE()
optimizer = mf.optimizers.SGD(learning_rate=0.01)
hypertuner = mf.hypertuner.HyperTuner(optimizer, lossfunction)

machine.learn((x, y), lossfunction, optimizer, hypertuner)


""" Evaluate the model """
evaluator = mf.evaluator.Accuracy()
machine.evaluate()


""" Inference """
g = machine.predict(x)
