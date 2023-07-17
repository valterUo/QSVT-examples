from scipy.special import erf
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial.chebyshev import Chebyshev, chebinterpolate, chebval, chebtrim, chebadd, chebmulx

class QSVTPolynomial:

    def __init__(self, threshold, n_qubits, delta = 0.01, force_parity = 'odd'):
        self.threshold = threshold
        self.n_qubits = n_qubits
        self.delta = delta
        self.epsilon = delta/2
        self.force_parity = force_parity
        
        self.degree = np.ceil(np.sqrt(2**self.n_qubits)*np.log2(1/self.delta)).astype(int)

        if force_parity == 'even':
            if self.degree % 2 != 0:
                self.degree += 1
        elif force_parity == 'odd':
            if self.degree % 2 == 0:
                self.degree += 1

        self.mod_erf = lambda x : (1 - 2*self.epsilon)*erf(10*(self.threshold - x))
        self.cheb_polynomial = chebinterpolate(self.mod_erf, self.degree)
        #self.cheb_polynomial = chebtrim(self.cheb_polynomial, tol=1e-10)

    def get_threshold_polynomial(self):
        self.mod_erf_plus = lambda x : (1 - 2*self.epsilon)*erf(10*(self.threshold + x))
        self.cheb_polynomial_plus = chebinterpolate(self.mod_erf_plus, self.degree)
        const_poly = Chebyshev([- 1 + self.epsilon/4])
        sum_polynomial = self.cheb_polynomial + self.cheb_polynomial_plus + const_poly
        scaled_polynomial = []
        for i, coeff in enumerate(sum_polynomial.coef):
            if  self.force_parity == 'even':
                if i % 2 == 0:
                    scaled_polynomial.append((1/(1 + self.epsilon/4))*coeff)
                else:
                    scaled_polynomial.append(0)
            elif self.force_parity == 'odd':
                if i % 2 != 0:
                    scaled_polynomial.append((1/(1 + self.epsilon/4))*coeff)
                else:
                    scaled_polynomial.append(0)
        return np.array(scaled_polynomial)


    def get_polynomial(self):
        return self.cheb_polynomial
    
    def get_degree(self):
        return self.degree
    
    def get_threshold(self):
        return self.threshold
    
    def get_delta(self):
        return self.delta
    
    def get_epsilon(self):
        return self.epsilon
    
    def get_n_qubits(self):
        return self.n_qubits
    
    def get_mod_erf(self):
        return self.mod_erf

    def plot(self, polynomial):
        x = np.linspace(-1.0, 1.0, num = 100)
        plt.plot(x, self.mod_erf(x), label="erf")
        plt.plot(x, chebval(x, polynomial), label=f"degree={self.degree}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0, shadow=True)
        plt.tight_layout()
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.show()

