from numba import cuda, jit, int64, int32, float32
import numpy as np

class _NN:

    def __init__(self, layer, activate):
        """
        list :param layer: list of neuron's number of each layer.
        str :param activate: type of the activate function.
        """
        self.layer = self.getLayers(layer)
        self.activate = activate
        self.output = 0
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

    def getInput(self, inputData):
        return inputData
    @jit
    def feedForward(self, inputData):
        activate = self.activate
        everyLayers = self.layer
        W = []
        m = len(everyLayers)
        input = self.getInput(inputData)
        for layer in everyLayers:
            W.append(layer.W_mat)
        x = 0
        for i in range(m):
            w = W[i]
            if i == 0: x = input
            x = self.act(self.runLayer(x, w))
        return x
    def runLayer(self, x, w):
        return matmul(w, x).val

    def act(self, x):
        m = x.shape[0]
        if self.activate == 'sigmoid':
            for i in range(m):
                x[i] = sigmoid(x[i]).val
        return x
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
    def __init__(self, target = 'cpu'):
        """

        str :param target: Run on the gpu or not.
        """
        self.target = target
        if target == 'gpu':
            self.target = True
        else:
            self.target = False
class activate(compute):
    """

    """
    def __init__(self, target = 'cpu'):
        if target == 'cpu':
            self.TARGET = False
        else:
            self.TARGET = True

class sigmoid(activate):
    """

    """
    def __init__(self, x, target = False):
        self.x = float32(x)
        self.target = target
        if self.target == True:
            self.val = self.gpu_f()
        else:
            self.val = self.cpu_f()
    def cpu_f(self):
        """

        :return: sigmoid function result from cpu.
        """
        return float32(1 / (1 + np.exp(-self.x)))
    @cuda.jit
    def gpu_f(self):
        """

        :return:
        """
class mul(compute):
    """

    """
class matmul(mul):
    def __init__(self, A, B, target = False):
        """
        np.array :param A: a mat
        np.array :param B: a mat
        """
        self.A = A
        self.B = B
        self.target = target
        if self.target == True:
            self.val = self.mul_G()
        else:
            self.val = self.mul_C()
    def mul_C(self):
        """
        compute with cpu.
        :return: result.
        """
        return np.dot(self.A, self.B)
    @cuda.jit
    def mul_G(self):
        """"""
a = _NN([2, 4, 3], "sigmoid")
x = np.random.rand(2,1)
print(a.feedForward(x))

