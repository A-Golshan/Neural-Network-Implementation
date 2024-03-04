import numpy as np
import matplotlib.pyplot as plt
from Optimizer import Optimizer

from sklearn.metrics import accuracy_score
from time import time


class MLP:

    def __init__(self):
    
        self.layers = []
        self.compiled = False
        self.fitted = False

    def add_layer(self, units, activation):
        self.layers.append({'units': units, 'activation': activation})
        
    def compile(self, loss, optimizer: str='mini-batch adam'):
    
        self.compiled = True
        self.loss_fn = loss

        self.train_acc_hist = []
        self.test_acc_hist = []
        self.val_acc_hist = []

        self.train_loss_hist = []
        self.test_loss_hist = []
        self.val_loss_hist = []

        # random matrix for weight and bias
        self.W = np.array(
            [
                np.random.randn(
                    self.layers[i + 1]['units'], 
                    self.layers[i]['units']
                )
                for i in range(len(self.layers) - 1)
            ], 
            dtype='object'
        )
        self.b = np.array(
            [
                np.matrix(np.random.randn(self.layers[i + 1]['units'])).T for i in range(len(self.layers) - 1)
            ], 
            dtype='object'
        )

        self.optimizer = Optimizer(
            self.W, self.b, 
            self.__backpropagation, 
            solver=optimizer,
            beta_1=0.8, beta_2=0.9
        )
    
    def __calculate_metrics(self, data_train, data_test=None, data_val=None):
    
        y_hat = self.predict(data_train[0])

        train_cost = self.__loss(data_train[1], y_hat, type=self.loss_fn)
        train_acc = accuracy_score(np.argmax(data_train[1], axis=1), np.argmax(y_hat, axis=1))

        self.train_loss_hist.append(train_cost)
        self.train_acc_hist.append(train_acc)

        if data_test is not None:
        
            y_test = self.predict(data_test[0])
            test_cost = self.__loss(data_test[1], y_test, type=self.loss_fn)
            test_acc = accuracy_score(np.argmax(data_test[1], axis=1), np.argmax(y_test, axis=1))

            self.test_loss_hist.append(test_cost)
            self.test_acc_hist.append(test_acc)
                
        if data_val is not None:
        
            y_val = self.predict(data_val[0])
            val_cost = self.__loss(data_val[1], y_val, type=self.loss_fn)
            val_acc = accuracy_score(np.argmax(data_val[1], axis=1), np.argmax(y_val, axis=1))

            self.val_loss_hist.append(val_cost)
            self.val_acc_hist.append(val_acc)

    def fit(
        self, 
        x_train, y_train, 
        data_test=None, data_val=None, 
        iters: int=10, 
        eta: float=0.001, 
        batch_size: int=32
    ):
    
        if not self.compiled:
            raise Exception("The Neural Network has not compiled yet!")

        start = time()
        
        k = 0
        i = 0
        
        while True:
            
            i += 1
            k += self.optimizer.step(
                x_train, 
                y_train, 
                eta, 
                self.__calculate_metrics, 
                data_test, data_val
            )

            if i % 25 == 0:

                log_train = f'Epoch {i}/{iters} - Train Loss : {self.train_loss_hist[-1]:.4f} - Train ACC : {self.train_acc_hist[-1]:.4f}'
                log_test = ''
                log_val = ''

                if data_test is not None:

                    log_test = f' - Test Loss : {self.test_loss_hist[-1]:.4f} - Test ACC : {self.test_acc_hist[-1]:.4f}'
                    
                if data_val is not None:

                    log_val = f' - Val Loss : {self.val_loss_hist[-1]:.4f} - Val ACC : {self.val_acc_hist[-1]:.4f}'

                print(log_train, log_test, log_val)
            if k > iters:
                break
        self.fitted = True

        end = time()

        print(f'Training Time : {(end - start):.4f} sec.')

    def predict(self, X):
        if not self.compiled:
            raise Exception("The Neural Network has not compiled yet!")
        
        _, _, Y = self.__feedforward(X)
        return np.array(Y.T)

    def visualize(self, X, y):

        if not self.fitted:
            raise Exception("The Neural Network has fitted yet!")

        plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
        plt.show()

        tx = np.linspace(X[:,0].min(), X[:,0].max(), 100)
        ty = np.linspace(X[:,1].min(), X[:,1].max(), 100)
        XX, YY = np.meshgrid(tx, ty)
        ZZ = np.empty(XX.shape)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                ZZ[i, j] = np.argmax(self.predict([[XX[i, j], YY[i, j]]]))

        plt.contourf(XX, YY, ZZ, alpha=0.4, cmap='RdBu')
        plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
        plt.show()
    
    def plot_history(self):
    
        legends = ['Train']
        plt.plot(range(len(self.train_acc_hist)), self.train_acc_hist, color='red')
        
        if len(self.val_acc_hist) != 0:
            legends.append('Validation')
            plt.plot(range(len(self.val_acc_hist)), self.val_acc_hist, color='green')
            
        if len(self.test_acc_hist) != 0:
            legends.append('test')
            plt.plot(range(len(self.test_acc_hist)), self.test_acc_hist, color='blue')
            
        plt.title('ACC History')
        plt.legend(legends)
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.show()

        plt.plot(range(len(self.train_loss_hist)), self.train_loss_hist, color='red')
        if len(self.val_loss_hist) != 0:
            plt.plot(range(len(self.val_loss_hist)), self.val_loss_hist, color='green')
        if len(self.test_loss_hist) != 0:
            plt.plot(range(len(self.test_loss_hist)), self.test_loss_hist, color='blue')
            
        plt.title('Loss History')
        plt.legend(legends)
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
    
    def summary(self):
    
        print(self.layers)
        print(f'Loss : {self.loss_fn}')

    def __feedforward(self, x):
    
        a = np.matrix(x.copy()).T
        A = [a]
        Z = [[]]
        
        # calculate a and Z for each neuron
        for l in range(len(self.layers) - 1):
        
            z = self.W[l] @ a + self.b[l]
            Z.append(z)
            a = self.__activation(z, activation=self.layers[l + 1]['activation'])
            A.append(np.matrix(a))
            
        return A, Z, A[-1]
        
    def __backpropagation(self, X, Y, eta):
    
        # feed-forward step
        a, Z, _ = self.__feedforward(X)
        # calculate partial derivatives
        deltas = self.__delta(a, Z, Y)
        d_w = self.__d_C_W(a, deltas)
        d_b = self.__d_C_b(deltas)

        return d_w, d_b

    def __delta(self, a, Z, y):
    
        deltas = [[] for i in range(len(self.layers))]
        # delta for last layer using equation (1) in Assignment 6
        deltas[-1] = a[-1].T - y
        # delta for other layers using equation (2) in Assignment 6
        for l in reversed(range(1 ,len(self.layers) - 1)):

            temp = np.array(deltas[l + 1] @ self.W[l]) * self.__activation(Z[l], activation=self.layers[l]['activation'], der=True).T
            deltas[l] = np.matrix(temp)
            
        return deltas
    
    def __d_C_W(self, a, deltas):
    
        D = []
        # derivative of weights using equation (3) in Assignment 6
        for l in range(1, len(self.layers)):
            D.append(np.array(np.matrix(a[l - 1]) @ np.matrix(deltas[l])).T)
            
        return np.array(D, dtype='object')
        
    def __d_C_b(self, deltas):
    
        # derivative of biases using equation (4) in Assignment 6
        D = []
        for l in range(1, len(self.layers)):
            D.append(np.array(np.sum(deltas[l], axis=0).T))
        return np.array(D, dtype='object')

    def __activation(self, x, activation='sigmoid', der=False):
    
        if not der:
            if activation == 'linear':
                return x
            elif activation == 'sigmoid':
                return 1 / (1 + np.exp(-x))
            elif activation == 'softmax':
                return np.exp(x) / np.sum(np.exp(x), axis=0)
            else:
                return None
        else:
            if activation == 'linear':
                return 1
            elif activation == 'sigmoid':
                s = np.array(self.__activation(x, activation=activation))
                return s * (1 - s)
            else:
                return None

    def __loss(self, y_true, y_pred, type='mse'):
    
        if type == 'mse':
            return np.square(y_true - y_pred).mean(axis=1)
        elif type == 'rmse':
            return np.sqrt(np.square(y_true - y_pred).mean(axis=1))
        elif type == 'binary cross-entropy':
            pass
        elif type == 'categorical cross-entropy':
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
