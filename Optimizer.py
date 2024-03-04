import numpy as np

class Optimizer:
    def __init__(
        self, 
        W, 
        b, 
        Gradient_fn, 
        solver: str='adam',
        beta_1: float=0.9, 
        beta_2: float=0.999
    ):
    
        self.solver = solver
        self.Gradeint_fn = Gradient_fn
        self.K = 1
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.kappa_w = 0.001
        self.kappa_b = 0.001
        self.d_w_1 = 0
        self.d_b_1 = 0

        self.W = W
        self.b = b
        self.m_w, self.m_b, self.s_w, self.s_b = 0, 0, 0, 0
        
    def step(
        self, 
        X, y, eta, 
        calculate_metrics, 
        data_test, 
        data_val, 
        epsilon: float=1e-8
    ):
    
        k = 0
        if self.solver == 'adam':
            d_w, d_b = self.Gradeint_fn(X, y, eta)

            self.m_w = self.beta_1 * self.m_w + (1 - self.beta_1) * d_w
            self.m_b = self.beta_1 * self.m_b + (1 - self.beta_1) * d_b

            self.s_w = self.beta_2 * self.s_w + (1 - self.beta_2) * d_w ** 2
            self.s_b = self.beta_2 * self.s_b + (1 - self.beta_2) * d_b ** 2

            m_w_bar = self.m_w / (1 - self.beta_1 ** self.K)
            m_b_bar = self.m_b / (1 - self.beta_1 ** self.K)

            s_w_bar = self.s_w / (1 - self.beta_2 ** self.K)
            s_b_bar = self.s_b / (1 - self.beta_2 ** self.K)
                        
            d_w = m_w_bar / (s_w_bar ** 0.5 + epsilon)
            d_b = m_b_bar / (s_b_bar ** 0.5 + epsilon)

            self.__update_parameters(d_w, d_b, eta)

            k += 1

            calculate_metrics((X, y), data_test, data_val)
        elif self.solver == 'GD':
        
            d_w, d_b = self.Gradeint_fn(X, y, eta)

            self.__update_parameters(d_w, d_b, eta)
            k += 1
            calculate_metrics((X, y), data_test, data_val)

        elif self.solver == 'SGD':
        
            i_k = np.random.randint(X.shape[0])
            d_w, d_b = self.Gradeint_fn(X[i_k], y[i_k], eta)

            self.__update_parameters(d_w, d_b, eta)
            k += 1
            calculate_metrics((X, y), data_test, data_val)

        elif self.solver == 'mini-batch adam':
        
            Batches = self.__get_batches(X, y)
            for batch in Batches:
            
                d_w, d_b = self.Gradeint_fn(batch.x, batch.y, eta)
                ### mini batch adam ##
                self.m_w = self.beta_1 * self.m_w + (1 - self.beta_1) * d_w
                self.m_b = self.beta_1 * self.m_b + (1 - self.beta_1) * d_b

                self.s_w = self.beta_2 * self.s_w + (1 - self.beta_2) * d_w ** 2
                self.s_b = self.beta_2 * self.s_b + (1 - self.beta_2) * d_b ** 2

                m_w_bar = self.m_w / (1 - self.beta_1 ** self.K)
                m_b_bar = self.m_b / (1 - self.beta_1 ** self.K)

                s_w_bar = self.s_w / (1 - self.beta_2 ** self.K)
                s_b_bar = self.s_b / (1 - self.beta_2 ** self.K)
                        
                d_w = m_w_bar / (s_w_bar ** 0.5 + 0.001)
                d_b = m_b_bar / (s_b_bar ** 0.5 + 0.001)

                self.__update_parameters(d_w, d_b, eta)

                k += 1

                calculate_metrics((X, y), data_test, data_val)
                
        elif self.solver == 'mini-batch GD':
            Batches = self.__get_batches(X, y)
            for batch in Batches:
                d_w, d_b = self.Gradeint_fn(batch.x, batch.y, eta)
                
                self.__update_parameters(d_w, d_b, eta)

                k += 1

                calculate_metrics((X, y), data_test, data_val)
        return k
        

    def __update_parameters(self, d_w, d_b, eta):
    
        self.W -= eta * d_w
        self.b -= eta * d_b
        self.K += 1

    def __get_batches(self, X, y, batch_size: int=32):
    
        indices = np.array(range(len(y)))
        # shuffle data
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        L = X.shape[0]
        
        batches = []
        for index in range(0, L, batch_size):
        
            batch_x = X[index:min(index + batch_size, L)]
            batch_y = y[index:min(index + batch_size, L)]
            batches.append(Batch(batch_x, batch_y))
            
        return batches

class Batch:
    def __init__(self, x, y):
        self.x = x
        self.y = y
