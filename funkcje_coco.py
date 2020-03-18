import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import ortho_group

################
# Functions optimum:
# f_opt = 0
################
# Definition of transformations used in functions definitions
# More information: https://coco.gforge.inria.fr/downloads/download16.00/bbobdocfunctions.pdf
################

def T_osz(x):
    x_estimate = np.zeros(len(x))
    c1 = np.zeros(len(x))
    c2 = np.zeros(len(x))
    for iter in range(0, len(x)):
        if x[iter] > 0:
            x_estimate[iter] = math.log10(math.fabs(x[iter]))
            c1[iter] = 10
            c2[iter] = 7.9
        elif x[iter] < 0:
            x_estimate[iter] = math.log10(math.fabs(x[iter]))
            c1[iter] = 5.5
            c2[iter] = 3.1
        else:
            c1[iter] = 5.5
            c2[iter] = 3.1

    x_transformed = np.zeros(len(x))
    for iter in range(0, len(x)):
        param = x_estimate[iter] + 0.049 + math.sin(c1[iter]*x_estimate[iter]) + math.sin(c2[iter]*x_estimate[iter])
        x_transformed[iter] = np.sign(x[iter]) * math.exp(param)

    return x_transformed

def T_osz_scalar(x):
    x_estimate = 0.0
    c1 = 0.0
    c2 = 0.0
    if x > 0:
        x_estimate = math.log10(math.fabs(x))
        c1 = 10
        c2 = 7.9
    elif x < 0:
        x_estimate = math.log10(math.fabs(x))
        c1 = 5.5
        c2 = 3.1
    else:
        c1 = 5.5
        c2 = 3.1

    param = x_estimate + 0.049 + math.sin(c1*x_estimate) + math.sin(c2*x_estimate)
    x_transformed = np.sign(x) * math.exp(param)

    return x_transformed

def T_asy(x, beta):
    x_transformed = x
    for iter in range(0, len(x)):
        if x[iter] > 0:
            x_transformed[iter] = x[iter] ** (1 + beta*(iter - 1)*math.sqrt(x[iter])/(len(x) - 1))
    return x_transformed

def diagonal_A(alpha, dim):
    A = np.zeros([dim, dim])
    for iter in range(0, dim):
        A[iter, iter] = alpha**(0.5 * (iter - 1)/(dim - 1))
    return A

def f_pen(x):
    sum = 0
    for iter in range(0,len(x)):
        sum += max(0, abs(x[iter]) - 5)**2
    return sum

def rand_vector(dim):
    vector = np.zeros(dim)
    for iter in range(0, dim):
        param = random.random()
        if (param < 0.5):
            vector[iter]= -1
        else:
            vector[iter]= 1
    return vector

################
# Test functions
# More information: https://coco.gforge.inria.fr/downloads/download16.00/bbobdocfunctions.pdf
# WARNING! - While adding additional function remember to put it on the fun_list
################

def sphere(x):
    return np.linalg.norm(x)**2

def ellipsoidal(x):
    sum=0
    z = T_osz(x)
    for iter in range(1,len(x)+1):
        sum += 10**(6 * (iter-1) / ( len(x)-1)) + z[iter-1]**2
    return sum

def rosenbrock(x):
    z = max(1, math.sqrt(len(x))/8) * x + 1
    sum=0
    for iter in range(0,len(x)-1):
        sum += 100 * (z[iter]**2 - z[iter + 1])**2 + (z[iter] - 1)**2
    return sum

def rastrigin(x):
    x_osz = T_osz(x)
    z = np.matmul(diagonal_A(10, len(x)), T_asy (x_osz, 0.2))
    sum = 0
    for iter in range(0, len(x)):
        sum += math.cos(2 * math.pi * z[iter])
    return 10 * (len(x) - sum) + np.linalg.norm(z)**2

def buche_rastrigin(x):
    s = np.zeros(len(x))
    z = np.zeros(len(x))
    sum1 = 0
    sum2 = 0
    x_osz = T_osz(x)
    for iter in range(0, len(x)):
        if x[iter] > 0 and (iter+1)%2 ==1 :
            s[iter] = 10 * 10**(0.5 * (iter)/(len(x) - 1))
        else:
            s[iter] = 10**(0.5 * (iter)/(len(x) - 1))
        z[iter] = s[iter] * x_osz[iter]
        sum1 += math.cos(2 * math.pi * z[iter])
        sum2 += z[iter]**2
    return 10 * (len(x) - sum1) + sum2 + 100 * f_pen(x)

def attr_sector(x):
    temp1 = np.matmul(ortho_group.rvs(len(x)), diagonal_A(10, len(x)))
    temp2 = np.matmul(temp1, ortho_group.rvs(len(x)))
    z = np.matmul(temp2, x)
    sum1 = 0
    s = np.zeros(len(x))
    for iter in range(0, len(x)):
        sum1 += (z[iter])**2
    sum1 = sum1**0.9
    return T_osz_scalar(sum1)

def slope(x):
    x_opt = rand_vector(len(x)) * 5
    s = np.zeros(len(x))
    z = np.zeros(len(x))
    sum = 0
    for iter in range(0, len(x)):
        s[iter] = 10**((iter)/(len(x) - 1)) * np.sign(x_opt[iter])
        if x_opt[iter]*x[iter] < 25:
            z[iter] = x[iter]
        else:
            z[iter] = x_opt[iter]
        sum += 5*abs(s[iter]) - s[iter]*z[iter]
    return sum

################
# Test functions list will be used in the other scripts
################

fun_list = [sphere, ellipsoidal, rosenbrock, rastrigin, buche_rastrigin, attr_sector, slope]
fun_name_list = ['sphere', 'ellipsoidal', 'rosenbrock', 'rastrigin', 'buche_rastrigin', 'attr_sector', 'slope']
fun_name_list_short = ['sphere', 'ellip.', 'rosen.', 'rastr.', 'buche.', 'a.sec.', 'slope.']
#fun_list = [sphere, ellipsoidal]
#fun_name_list = ['sphere','ellipsoidal']
#fun_name_list_short = ['sphere', 'ellip.']

################
# Test functions 3D plots
# Plots will be saved as .png files.
# WARNING! - Plots are shown one by one. The next plot will be seen after closing the previous one.
################
if __name__ == "__main__":
    for iter in range(len(fun_list)):
        fig1 = plt.figure()
        ax = fig1.gca(projection='3d')
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        Xlen = len(X)
        Ylen = len(Y)
        Z = np.zeros([Xlen,Ylen])
        for xind in range(0, len(X)):
            for yind in range(0, len(Y)):
                input = np.asarray([X[xind], Y[yind]])
                Z[xind,yind] = fun_list[iter](input)
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        plt.savefig(fun_name_list[iter] + ".png")
        plt.show(fig1)
