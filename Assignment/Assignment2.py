import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



np.random.seed(0) #hep aynı randomlari uretmek icin

def polynomial(values, coeffs):
    # Coeffs are assumed to be in order 0, 1, ..., n-1
    expanded = np.column_stack([coeffs[i] * (values ** i) for i in range(0, len(coeffs))])
    print(expanded)
    return np.sum(expanded, axis=-1)

def polynomial_data(coeffs, n_data=100, x_range=[-1, 1], eps=0.1):
    x = np.random.uniform(x_range[0], x_range[1], n_data)
    poly = polynomial(x, coeffs)
    return x.reshape([-1, 1]), np.reshape(poly + eps * np.random.randn(n_data), [-1, 1])


# 1 + 0.5 * x - 0.5 x^2 - 0.2 x^3 - 0.W1 x^4
coeffs = [1, 0.5, -0.5, -0.2, -0.1]
X, y = polynomial_data(coeffs, 100, [90, 110], 200)
#print("A)",X.shape)
#print("B)",y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0) #egitim ve test olarak ikiye ayırdık  
#print("c)",X_train.shape)
#print("d)",y_train.shape)

scaler = StandardScaler() #olceklendirme (0 ort ve birim varyns)
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

class LRwithGradientDecent:
    def _init__(self, lr):
        """
        Initializes the class instance with the specified learning rate
        Parameters:
            lr (float): The learning rate used in gradient descent.
        """
        self.lr = lr
        self.weights = None
        self.bias = None
        self.loss_values = []
        self.X = None  # Input data
        self.y = None  # Actual values
       
        
    def initialize_parameters(self):
        if self.X.ndim ==1:
             self.W = 0
        else:
            self.W = self.W = np.random.randn(self.X.shape[-1]) * np.sqrt(2 / (self.X.shape[-1] + 1))

        self.b = 0
        
    def forward(self, X):

        Z=np.dot(X, self.weights) + self.bias
        return Z

    
    def compute_loss(self, preds, y):
        self.y=y
        self.preds=preds

        mse_loss = ((preds - y) ** 2).mean()
        self.loss_values.append(mse_loss)

        return mse_loss
        
       
    def backward(self, preds):

        d_loss = 2 * (preds - y) / len(preds) 

       
        self.dW = self.X.T.dot(d_loss)  
        self.db = d_loss.sum()  


    def update(self, learning_rate):

        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db

    def fit(self, X, y, n_iter, plot_cost=True):
        self.X = X
        self.y = y
        self.initialize_parameters()

        for n in range(int(n_iter)):
            preds = self.predict(X)
            loss = self.compute_loss(preds, y)
            self.backward(preds, y)
            self.update(learning_rate=0.01)  # Adjust the learning rate as needed

            self.loss_values.append(loss)

        if plot_cost:
            import matplotlib.pyplot as plt
            plt.plot(range(n_iter), self.loss_values)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.title("Cost Function Evolution")
            plt.show()
    
    def predict(self, X):
        preds = X.dot(self.weights) + self.bias
        return preds


model = LRwithGradientDecent(lr=33)


model.fit(X, y, n_iter=1000)

       
