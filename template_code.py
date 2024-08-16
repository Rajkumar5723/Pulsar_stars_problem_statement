import sys
import numpy as np
from cvxopt import solvers
import cvxopt.solvers                 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\RAJKUMAR\Desktop\Program\pulsar_stars.csv")
df.head()                                                          

X = df.drop('Class', axis=1)
y = df['Class']                                                                                     
X = X.to_numpy()
y = y.to_numpy()                                                                                    
y[y == 0] = -1                                                                                      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)           
mean_train = X_train.mean()                                                                         
std_train = X_train.std()


mean_train = X_train.mean(axis=0)
std_train = X_train.std(axis=0)

X_train = (X_train - mean_train) / std_train  
X_test = (X_test - mean_train) / std_train

class SVM(object):
    
    def linear_kernel(self, x1, x2):                                                            
        return np.dot(x1, x2)                                        

    def __init__(self, kernel_str='linear', C=1.0, gamma=0.1):                                 
        if kernel_str == 'linear':
            self.kernel = SVM.linear_kernel
        else:
            self.kernel = SVM.linear_kernel
            print('Invalid kernel string, defaulting to linear.')
        self.C = C
        self.gamma = gamma
        self.kernel_str = kernel_str
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        kernel_matrix = np.zeros((num_samples, num_samples))                                                    
        kernel_matrix = self.kernel(self, X, X.T)

        P = cvxopt.matrix(np.outer(y,y) * kernel_matrix)                                                    
        q = cvxopt.matrix(np.ones(num_samples) * -1)
        A = cvxopt.matrix(y, (1,num_samples)) * 1.
        b = cvxopt.matrix(0) * 1.
        G_upper = np.diag(np.ones(num_samples) * -1)
        G_lower = np.identity(num_samples)
        G = cvxopt.matrix(np.vstack((G_upper, G_lower)))
        h_upper = np.zeros(num_samples)
        h_lower = np.ones(num_samples) * self.C
        h = cvxopt.matrix(np.hstack((h_upper, h_lower)))

        solvers.options['show_progress'] = False                                                            
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)                                                      
        a = np.ravel(solution['x'])                                                                         
        support_vectors = a > 1e-4                                                                          
        ind = np.arange(len(a))[support_vectors]                                                            
        self.a = a[support_vectors]                                                                        
        self.support_vectors = X[support_vectors]
        self.y_support_vectors = y[support_vectors]
        

        self.b = 0                                                                                          
        for n in range(len(self.a)):
            self.b += self.y_support_vectors[n]
            self.b -= np.sum(self.a * self.y_support_vectors * kernel_matrix[ind[n],support_vectors])
        self.b /= len(self.a)

        if self.kernel_str == 'linear':                                                                    
            self.w = np.zeros(num_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.y_support_vectors[n] * self.support_vectors[n]
        else:
            self.w = None                                                                                   

    def predict(self, X):
        if self.kernel_str == 'linear': 
                                                                               
            y_predict = np.dot(X, self.w) + self.b
            return np.sign(y_predict)
        else:
            y_predict = np.sum(self.a * self.y_support_vectors * self.kernel(self, X, self.support_vectors.T), axis=1)  
            
            return np.sign(y_predict)
        

X_train = X_train[:800]
y_train = y_train[:800]
X_test = X_test[:200]
y_test = y_test[:200]

if __name__ == '__main__':

    input_data_one = sys.argv[1].strip()
    
    c_value = float(input_data_one)

    svm_linear = SVM('linear', C=c_value)
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    print(accuracy_score(y_test, y_pred_linear))