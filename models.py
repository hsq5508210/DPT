from numba import cuda, int64, int32
import numpy as np

class _NN:
    def __init__(self, layer, activate):
        """
        list :param layer: list of neuron's number of each layer.
        str :param activate: type of the activate function.
        """
        self.layer = self.getLayers(layer)
        self.activate = activate
    def getLayers(self, layer):
        """
        :param layer: argument list of hidden layer objects.
        :return: list of the layers objects.
        """
        L = []
        m = len(layer) - 1
        for i in range(m):
            L.append(_layer(layer[i], layer[i + 1]))
        return L
    def feedforward(self):
        activate = self.activate

class _layer:
    def __init__(self, inputNum, outputNum):
        """
        :param inputNum: neuron number of input layer.
        :param outputNum: neuron number of output layer.
        """
        self.outNodesNum = outputNum
        self.inNodesNum =  inputNum
        self.W_mat = np.random.rand(self.outNodesNum, self.inNodesNum)
    def creatLayer(self):
        """
        generate the layer.
        :return: layer.
        """
        layer = 0
        return layer
class inputLayer:
    """

    """
    def __init__(self, inputWise):
        ""

class outputLayer:
    """

    """
    def __init__(self, outputWise):
        ""

class compute:
    """

    """
class mul(compute):
    """

    """
class matmul(mul):
    def __init__(self, A, B):
        """
        np.array :param A: a mat
        np.array :param B: a mat
        """
        self.A = A
        self.B = B
    def mul_C(self):
        """
        compute with cpu.
        :return: result.
        """
        return np.dot(self.A, self.B)
    @cuda.jit
    def mul_G(self):
        """"""
a = _NN([2, 2, 9], "y")

print(a.layer[0].W_mat, '\n\n', a.layer[1].W_mat)
