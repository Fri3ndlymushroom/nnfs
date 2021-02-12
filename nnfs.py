
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# ====================================[Dense layer]====================================



class Layer_Dense:
    # Layer initializialisierung
    def __init__(self, n_inputs, n_neurons):
        # Zufällige gewichtie und biases werden erzeugt
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # np.zeros füllt ein array mit der länge n_neurons mit nullen
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Die Inputs werden gespeichert
        self.inputs = inputs
        # Die outputs werden gespeichert
        # Mit np.dot werdn die inputs zuerst alle multipliziert dann werden die produkte addiert und am ende der Bias hinzugefügt
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # die Transposeten Inputs werden mit dem Resultat der Ableitung der activation function (dvalues) verarbeitet
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# ====================================[ReLU activation]====================================


class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Input values werden sich gemerkt
        self.inputs = inputs

        # Output values werden berechnet.
        # bei der relu funktion werden alle werte die kleiner als null sind null alle grösseren sich selbst
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Weil hier die Variable veränder werden muss wird eine Kopie angefertigt
        self.dinputs = dvalues.copy()

        # Wenn die Werte Negativ sind ist der Output null
        self.dinputs[self.inputs <= 0] = 0

# ====================================[Softmax activation]====================================


class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Input values werden sich gemerkt
        self.inputs = inputs

        # Unnormalisierte werte berechnen
        # np.max holt den grössten wert aus einem Array
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # für jedes Sample Nomralisieren
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Erstellt ein Array in der Form von dvlaues mit nullen gefüllt
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # output Array abflachen
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix aus output berechnen
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            # sample weise den gradient berechnen
            # zum Array hinzu addieren
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# ====================================[SGD optimizer]====================================


class Optimizer_SGD:
    # Optimizer inizialisieren
    # die learning rate 1 ist der default für diesen optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # einmal vor jedem parameter update aufrufen
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Paramter updaten
    def update_params(self, layer):
        # wenn momentum genutzt wird
        if self.momentum:
            # wenn der layer keine momentum array hat sollen diese mit nullen erzeugt werden
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # wenn es keinen weight momentum hat hat es auch für bias noch keinen
                layer.bias_momentums = np.zeros_like(layer.biases)
            # gewichts update wird mit momentum erzeugt
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # bias update wird mit momentum erzeugt
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (wie vor momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                layer.dweights
            bias_updates = -self.current_learning_rate * \
                layer.dbiases
        # Gewichte und bias mit momentum oder Vanilla updaten
        layer.weights += weight_updates
        layer.biases += bias_updates

    # nach jedem parameter update ausführen
    def post_update_params(self):
        self.iterations += 1

# ====================================[Adagrad optimizer]====================================


class Optimizer_Adagrad:
    # optimizer initialisieren
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    # einmal vor jedem parameter update aufrufen

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # parameter updaten
    def update_params(self, layer):
        # wenn der layer keine cache arrays hat sollen welche mit nullen erzeugt werden
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # cache mit quadrierten gradienten updaten
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        # Vanilla SGD parameter update + normalisierung
        # mit square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights(np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases(np.sqrt(layer.bias_cache) + self.epsilon)

    # einmal nach jedem parameter update aufrufen
    def post_update_params(self):
        self.iterations += 1

# ====================================[RMSprop optimizer]====================================


class Optimizer_RMSprop:
    # Optimizer initialisieren
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # einmal vor jedem parameter update aufrufen
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # parameter updaten
    def update_params(self, layer):
        # wenn der layer keine cache arrays hat sollen welche mit nullen erzeugt werden
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # cache mit quadrierten gradienten updaten
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2
        # Vanilla SGD parameter update + normalisierung
        # mit square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights(np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases(np.sqrt(layer.bias_cache) + self.epsilon)
    # einmal nach jedem parameter update aufrufen

    def post_update_params(self):
        self.iterations += 1

# ====================================[Adam optimizer]====================================


class Optimizer_Adam:
    # Optimizer initialisieren
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # einmal vor jedem parameter update aufrufen
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    # parameter updaten

    def update_params(self, layer):
        # wenn der layer keine cache arrays hat sollen welche mit nullen erzeugt werden
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        # momentums mit momentanen gradien ten updaten
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # korrigiertes momentum bekommen
        # self.iteration ist 0 beim ersten durchgang
        # aber es braucht 1 desswegen + 1
        weight_momentums_corrected = layer.weight_momentums(
            1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums(
            1 - self.beta_1 ** (self.iterations + 1))
        # cache mit quadrierten gradienten updaten
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # korrigiertes cache bekommen
        weight_cache_corrected = layer.weight_cache(
            1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache(
            1 - self.beta_2 ** (self.iterations + 1))
        # Vanilla SGD parameter update + normalisierung
        # mit square rooted cache
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected(
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected(
                np.sqrt(bias_cache_corrected) + self.epsilon)
    # einmal nach jedem parameter update aufrufen

    def post_update_params(self):
        self.iterations += 1

# ====================================[Common loss class]====================================


class Loss:
    # berechnet die data losses
    # mit dem model output und den ground trouth values
    def calculate(self, output, y):
        # sample losses berechnen
        sample_losses = self.forward(output, y)
        # mean loss berechnen
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

# ====================================[Cross-entropy loss]====================================


class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # anzahl samples in einem batch
        samples = len(y_pred)
        # Clip data um division durch null zu verhindern
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # wahrscheinlichkeit für target value
        # nur categorial labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - nur für one hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    # Backward pass

    def backward(self, dvalues, y_true):
        # anzahl samples
        samples = len(dvalues)
        # anzahl labels in jedem sample
        # sie werden im ersten sample gezählt
        labels = len(dvalues[0])
        # wenn die labels sparse sind sollen sie in einen one hot encoded vector verwandelt werden
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # gradient berchenen
        self.dinputs = -y_true / dvalues
        # gradient normalisieren
        self.dinputs = self.dinputs / samples


# ====================================[Activation_Softmax_Loss_CategoricalCrossentropy]====================================
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # erzeugt activation und loss function objekte
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    # Forward pass

    def forward(self, inputs, y_true):
        # Output layer’s activation function
        self.activation.forward(inputs)
        # output bestimmen
        self.output = self.activation.output
        # loss berechnen und zurück geben
        return self.loss.calculate(self.output, y_true)
    # Backward pass

    def backward(self, dvalues, y_true):
        # anzahl samples
        samples = len(dvalues)
        # wenn lables one hot encoded sind sollen sie in discrete values umgewandelt werden
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # kopieren um sicher bearbeiten zu können
        self.dinputs = dvalues.copy()
        # gradient berechnen
        self.dinputs[range(samples), y_true] -= 1
        # gradient normalisieren
        self.dinputs = self.dinputs / samples


# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)
# Create Softmax classifier’s combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# Create optimizer
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)
# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()





