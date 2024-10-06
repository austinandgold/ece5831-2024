import numpy as np
import multilayer_perceptron as mlp

MLP = mlp.MultiLayerPreceptron()
MLP.init_network()
y = MLP.forward(np.array([7.0, 2.0]))

print(y)