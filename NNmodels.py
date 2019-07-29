from numba import cuda, jit, int64, int32, float32
import numpy as np
import getMINIST
import fileOp
from sympy import *
# np.set_printoptions(threshold='nan')
class _NN:

    def __init__(self, layer, input, label, activate, learningRate):
        """
        list :param layer: list of neuron's number of each layer.
        str :param activate: type of the activate function.
        """
        self.learningRate = learningRate
        self.layer = self.getLayers(layer)
        self.activate = activate
        self.inputData = input
        self.label = label
        self.loss = 0
        self.output = 0
    def getLayers(self, layer):
        """
        :param layer: argument list of hidden layer objects.
        :return: list of the layers objects.
        """
        L = []
        m = len(layer) - 1
        for i in range(m):
            L.append(_layer(layer[i], layer[i + 1], i))
        return L

    def getInput(self):
        return self.inputData
    @jit
    def feedForward(self):
        """
        execute the feed-forward propaganda at once.
        :return:
        """
        activate = self.activate
        everyLayers = self.layer
        W = []
        B = []
        m = len(everyLayers)
        input = self.getInput()
        for layer in everyLayers:
            # print("layer index is: ", layer.layerIdx, ", out shape is: ",
            #       layer.outNodesNum, ", W shape is :",
            #       layer.W_mat.shape, ' ', layer.bias.shape)
            W.append(layer.W_mat)
            B.append(layer.bias)
        x = 0
        for i in range(m):
            w = W[i]
            bias = B[i]
            if i == 0:
                x = input
            x = self.act(self.runLayer(w, x) + bias)
            self.layer[i].outputVal = x
        return x

    def runLayer(self, W, x):
        return matmul(W, x).val

    def act(self, x):
        """
        :param x: input variable.
        :return: activate function result.
        """
        m = x.shape[0]
        if self.activate == 'sigmoid':
            for i in range(m):
                x[i] = sigmoid(x[i]).val
        return x

    # def getDiff(self, x, fun):
    #     """
    #
    #     :param x: value of x.
    #     :param fun: function form.
    #     :return: diff value.
    #     """
    #     x, y = symbols('x, y')
    #     res =

    def getGrad(self, func, x_name, x_val):
        """

        :param func:
        :param x_name:
        :param x_val:
        :return:
        """
        return diff(func, x_name)

    def updateWeight(self, layerIdx, grad_B, grad_W):
        """"""
        learningRate = self.learningRate
        self.layer[layerIdx].W_mat = self.layer[layerIdx].W_mat - \
                                     (learningRate * grad_W)
        self.layer[layerIdx].bias = self.layer[layerIdx].bias - \
                                    (learningRate * grad_B)

    def getDelta(self, layerIndex, outputLayerMark):
        """"""

    @jit
    def backPropaganda(self):
        """
        execute the back propaganda procedure one time.
        """
        output = self.feedForward()
        lossGrad = output - self.label
        self.loss = (0.5 * (lossGrad**2)).mean()
        everyLayers = self.layer
        delta = []
        outputs = []
        for i in range(len(everyLayers) - 1, -1, -1):
            output = everyLayers[i].outputVal
            theLayer = everyLayers[i]

            nextLayer = everyLayers[i - 1]
            if i == len(everyLayers) - 1:
                delta.append(lossGrad * output * (1 - output))
                continue
            else:
                lastLayer = everyLayers[i + 1]
                delta.append(np.dot(lastLayer.W_mat.T, delta[-1]) * output * (1 - output))
            grad_B = delta[-1]
            grad_W = delta[-1] * theLayer.outputVal
            self.updateWeight(i, grad_B, grad_W)
    @jit
    def train(self, epoch):
        """"""
        m = (self.inputData.shape[0])
        x_train = self.inputData
        y_train = self.label
        for i in range(epoch):
            if epoch % 10 == 0:
                print(self.loss)
            for j in range(m):
                self.inputData = x_train[j]
                self.label = y_train[j]
                self.backPropaganda()



class _layer:
    def __init__(self, inputNum, outputNum, layerIdx):
        """

        int :param inputNum: neuron number of input layer.
        int :param outputNum: neuron number of output layer.
        int :param layerIdx: index of this layer.
        """
        self.outNodesNum = outputNum
        self.inNodesNum = inputNum
        self.layerIdx = layerIdx
        self.outputVal = 0
        self.W_mat = np.random.randn(self.outNodesNum, self.inNodesNum)
        self.bias = np.random.randn(self.outNodesNum, 1)
        self.grad = []

    # def creatLayer(self):
    #     """
    #     generate the layer.
    #     :return: layer.
    #     """
    #     layer = None
    #     return layer

    def getGrad(self, nodeVal, delta):
        """"""
        grad = 0
        # grad =
        delta *= grad
        return grad, delta
    def updateWeights(self):
        """"""

class node:
    def __init__(self, nodeIndex, output, allocateW):
        """

        :param nodeIndex: (i, j, k) 'i' is index of W[], 'j', 'k' is element index of mat W[i].
        :param output: the node output value.
        :param allocateW: W[i][j, k] value.
        """
        self.nodeIndex = nodeIndex
        self.output = output
        # self.input = None
        self.allocateW = allocateW
    def updateW(self, newVal):
        """"""
        self.allocateW = newVal


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

x = np.array([[1],
              [2],
              [1],
              [1],
              [0],
              [1]])
y = np.array([[1],
              [0],
              [0]])
# for i in range(100):
#     x.append(np.random.rand(2,1))
#     y.append(np.random.randint(0, 1, (2, 1)))
inputdata = getMINIST.load_train_images().reshape((60000, 784, -1))
label = getMINIST.load_train_labels()
print(inputdata.shape)
print(label.shape)
a = _NN([784, 512, 512, 128, 10], inputdata, label=label, activate="sigmoid", learningRate=0.001)
a.train(1000)





