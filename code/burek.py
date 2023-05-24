from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

# Define the model
model = Sequential()
model.add(Dense(93, input_dim=283, activation='relu'))  # Hidden layer
model.add(Dense(83, activation='relu'))  # Output layer

# Generate the model diagram
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

