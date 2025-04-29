import miniflow as mf
from miniflow import regularizers as reg
import gendata as gd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Initializing necessary components.

dataproc = mf.dataprocessor.DataProcessor()

optimizer = mf.optimizers.SGD(learning_rate=0.001, momentum=0.1) 
lr_scheduler = mf.callbacks.LearningRateScheduler('step', optimizer) # Learning rate scheduler
early_stopper = mf.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

loss = mf.lossfunctions.CategoricalCrossEntropyLoss()

encoder = mf.dataprocessor.OneHotEncoder1D()

scaler = mf.dataprocessor.StandardScaler()


"""
Data generation.
    Options:
    patterns: 'blobs', 'spirals', 'moons', 'circles'
    split: if the data should be split into x, y
"""
gd_generator = gd.gendata.ClusteredClasses2D(pattern='blobs', labels_n=3,
                                         sample_size=(3000, 3000, 3000), # Total desired samples
                                         seed=1, split=True)
x_all, y_all = gd_generator.getdata()

"""
Data loading (optional).
"""
"""
# Ignores first line.
PATH_DATASET1 = None
dataproc.load_dataset(PATH_DATASET1)
"""

# Split the data. Using sklearn.
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.5, random_state=42, stratify=y_all)


# Plot initial data
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Training Data")
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', s=10) # smaller points
plt.xlabel("x1")
plt.ylabel("x2")

plt.subplot(2, 2, 2)
plt.title("Test Data")
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='viridis', edgecolors='k', s=10)
plt.xlabel("x1")
plt.ylabel("x2")

# Define your model HERE...
model = mf.model.Sequential()
model.add(mf.layers.Input(x_train.shape[1]))
model.add(mf.layers.SimpleDense(16, 'sigmoid', reg.L2Regularizer(0.001)))
model.add(mf.layers.SimpleDense(8, 'sigmoid', reg.L2Regularizer(0.001)))
model.add(mf.layers.SimpleDense(3, 'softmax')) # Assuming 2 classes

# Build the model using specified optimizer and loss
model.build(optimizer=optimizer, loss=loss)

# Preprocess data
x_train = scaler(x_train)

y_train_encoded = dataproc.encode(encoder, y_train)
x_train_shuffled, y_train_encoded_shuffled = dataproc.shuffle(x_train, y_train_encoded)

x_train_fin, x_val_fin = x_train_shuffled[:int(len(x_train)*0.9)], x_train_shuffled[int(len(x_train)*0.9):]
y_train_fin, y_val_fin = y_train_encoded_shuffled[:int(len(y_train)*0.9)], y_train_encoded_shuffled[int(len(y_train)*0.9):]


# Set training parameters
epochs = 100
batch_size = 8
batches = dataproc.batchify(x_train_fin, y_train_fin, batch_size=batch_size)

model.train() # Set model to training mode
# Main training loop
early_stopper.load_model(model)
for epoch in range(epochs):
    lr_scheduler.on_epoch_begin(epoch)
    for x_batch, y_batch in batches:
        g_batch = model.predict(x_batch) # Forward pass (training mode)
        loss(g_batch, y_batch)           # Calculate loss and delta_net
        gradients = model.gradients(loss.delta_net) # Backpropagate using delta_net
        optimizer.step(gradients)         # Update weights
    model.untrain() # Set to evaluation mode
    val_predictions = model.predict(x_val_fin)
    val_acc = np.mean(val_predictions.argmax(axis=1) == y_val_fin.argmax(axis=1))
    if early_stopper.on_epoch_end(epoch, val_acc):
        break
    model.train()
model.untrain() # Set model to prediction mode

predictions = model.predict(x_test) # Forward pass (prediction mode)
predicted_classes = predictions.argmax(axis=1)

# Evaluate
accuracy = np.mean(predicted_classes == y_test)
print(f"Test Accuracy: {accuracy:.0%}")

# Plot predictions
plt.subplot(2, 2, 3)
plt.title("Model Predictions on Test Data")
plt.scatter(x_test[:, 0], x_test[:, 1], c=predicted_classes, cmap='viridis', edgecolors='k', s=10)
plt.xlabel("x1")
plt.ylabel("x2")

plt.tight_layout()
plt.show()

