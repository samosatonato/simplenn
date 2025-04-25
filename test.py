# Example Fix in main script:
import miniflow as mf
from miniflow import regularizers as reg
import gendata as gd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # Or manual split

# 1. Generate ONE dataset
gd_generator = gd.gendata.ClusteredClasses2D(pattern='blobs', labels_n=2,
                                         sample_size=(200, 200), # Generate total desired samples
                                         seed=1, split=True)
x_all, y_all = gd_generator.getdata()

# 2. Split the dataset
# Using sklearn (recommended)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.5, random_state=42, stratify=y_all)

# --- Rest of your main script ---
dtproc = mf.dataprocessor.DataProcessor()

# Plot initial data (optional: plot all or just train/test)
plt.figure(figsize=(10, 8)) # Adjusted size

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

# Model Definition...
model = mf.model.Sequential()
model.add(mf.layers.Input(x_train.shape[1]))
model.add(mf.layers.SimpleDense(8, 'sigmoid', reg.L2Regularizer(0.001)))
model.add(mf.layers.SimpleDense(6, 'sigmoid', reg.L2Regularizer(0.001)))
model.add(mf.layers.SimpleDense(2, 'softmax')) # Assuming 2 classes for spiral

optimizer = mf.optimizers.SGD(learning_rate=0.003) # Instantiate optimizer
loss = mf.lossfunctions.CategoricalCrossEntropyLoss()
encoder = mf.dataprocessor.OneHotEncoder1D()

model.build(optimizer=optimizer, loss=loss) # Pass instance

scaler = mf.dataprocessor.StandardScaler()
x_train = scaler(x_train)
x_test = scaler(x_test)

# Encode training labels ONLY
y_train_encoded = dtproc.encode(encoder, y_train)
# **Important:** y_test remains as class indices for evaluation comparing with argmax

x_train_shuffled, y_train_encoded_shuffled = dtproc.shuffle(x_train, y_train_encoded)

epochs = 100
model.train() # Set model to training mode
for epoch in range(epochs):
    batch_size = 1
    batches = dtproc.batchify(x_train_shuffled, y_train_encoded_shuffled, batch_size=batch_size)
    for x_batch, y_batch in batches:
        g_batch = model.predict(x_batch) # Forward pass (training mode)
        loss(g_batch, y_batch)           # Calculate loss and delta_net
        gradients = model.gradients(loss.delta_net) # Backpropagate using delta_net
        optimizer.step(gradients)         # Update weights

model.untrain() # Set model to prediction mode
predictions = model.predict(x_test) # Forward pass (prediction mode)
predicted_classes = predictions.argmax(axis=1)

# Evaluate (compare predicted indices with original test indices)
accuracy = np.mean(predicted_classes == y_test) # Simpler accuracy calculation
print(f"Test Accuracy: {accuracy:.0%}")

# Plot predictions
plt.subplot(2, 2, 3)
plt.title("Model Predictions on Test Data")
plt.scatter(x_test[:, 0], x_test[:, 1], c=predicted_classes, cmap='viridis', edgecolors='k', s=10)
plt.xlabel("x1")
plt.ylabel("x2")

plt.tight_layout()
plt.show()