# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 00:02:07 2021

@author: nlewis
"""

#useful math functions and everything else
import numpy as np

#useful for setting up matrices
from scipy.sparse import spdiags
#for plotting our histograms and contours
import matplotlib.pyplot as plt

#for making cool animations


#Problem 6
# The purpose of this problem is to code up solutions for the transport equation using 
#different numerical methods. As such

### Step sizes for space (h) and time (k)
#we define lambda to be 1/2, but lambda = k/h so k = h/2. 
lambda_ = 1/2
h = 0.5**np.array([5,6,7])
k_ = h/2

### THIS CREATES THE SEQEUNCE OF STEPS FOR EACH ANALYSIS. Recall x_j = hj, 2pi/h = N+1
###NOTE: since N will not be an integer, we can just take int of this expression which
#will act as a floor function
### We use a list to save on memory.

#Boundaries for  space variable
x_0 = 0
x_N = 2*np.pi

#Boundaries for time variable
t_0 = 0
t_K = 1

##space variable gridpoints for j = 5, 6, 7
###time variable grid points for j = 5, 6, 7

x,t = [],[]
for j in range(len(h)):
    x.append( np.linspace(start = x_0, stop = x_N, num = int( ((x_N - x_0)/h[j]) - 1)) )
    t.append( np.linspace(start = t_0, stop = t_K, num = int( ((t_K - t_0)/k_[j]) - 1)) )


#How many gridpoints are in the space and time variables
N = np.array([len(x[0]), len(x[1]), len(x[2])])
K = np.array([len(t[0]), len(t[1]), len(t[2])])

## Define our initial condition here
def f(x):
    #let's us define a piecewise function
    tau = 2*np.pi
    #makes sure this is periodic for values not in the interval [0,2pi]
    x = x % tau
    y= np.piecewise(
        x,
        [(0 <= x)&(x <= np.pi),  (np.pi < x)&(x < 2*np.pi)     ],
        [lambda x: x, lambda x: 2*np.pi - x,0])
    return y

### Graph of f(x)
plt.plot(x[0],f(x[0]))

q = np.linspace(0,4*np.pi,1000)
plt.plot(q,f(q))
#self explanatory
plt.xlabel('x')
plt.ylabel('u(x,0)')
plt.title('Transport Equation Initial Condtion')



### UPWIND SCHEME
#v^(n+1)_j = v^n_j + lambda*(v^n_(j+1) - v^n_j)
#v^0_j = f_j
# Recall this can be written in matrix form
o1 = np.ones(N[0])
o2 = np.ones(N[1])
o3 = np.ones(N[2])
A = [spdiags( [lambda_*o1,(1-lambda_)*o1,lambda_*o1], (1-N[0],0,1), N[0], N[0] ).toarray(), 
     spdiags( [lambda_*o2,(1-lambda_)*o2,lambda_*o2], (1-N[1],0,1), N[1], N[1] ).toarray()
     , spdiags( [lambda_*o3,(1-lambda_)*o3,lambda_*o3], (1-N[2],0,1), N[2], N[2] ).toarray()]

### Makes it so we can do this for any given j = 5,6 or 7. Simplifies analysis
solutions0 = []
for j in range(len(N)):
    solutions0.append( np.zeros((N[j],K[j])) )
    
for k in range(3):
    #initializes conditions to start with
    solutions0[k][:,0] = f(x[k])
    for j in range(1,K[k]):
        #using matrix corresponding to steps, we sample. We then set initial equal to the old
        #so we can repeat this process (i.e. we use previous time to update future time)
        solutions0[k][:,j] = A[k].dot(solutions0[k][:,j-1])

f1 = solutions0[0]
f2 = solutions0[1]
f3 = solutions0[2]

plt.plot(x[0], f1[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Upwind Method for Transport Equation' +',' + 'h= ' + str(1/2**5))

plt.plot(x[1], f2[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Upwind Method for Transport Equation' +',' + 'h= ' + str(1/2**6))

plt.plot(x[2], f3[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Upwind Method for Transport Equation' +',' + 'h= ' + str(1/2**7))

plt.plot(x[0], f1[:,-1])
plt.plot(x[1], f2[:,-1])
plt.plot(x[2], f3[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Upwind Method for Transport Equation')
plt.legend(['h =' + str(1/2**5), 'h =' + str(1/2**6),'h =' + str(1/2**7)])

### NAIVE METHOD (USING CENTRAL DIFFERENCE)
#v^(n+1)_j = v^n_j + lambda*(v^n_(j+1) - v^n_(j-1))
#v^0_j = f_j

A_CD = [spdiags( [lambda_*o1,-lambda_*o1, o1, lambda_*o1,-lambda_*o1], (1-N[0],-1,0,1,N[0]-1), N[0], N[0]).toarray(), 
     spdiags( [lambda_*o2,-lambda_*o2, o2, lambda_*o2,-lambda_*o2], (1-N[1],-1,0,1,N[1]-1), N[1], N[1] ).toarray()
     , spdiags([lambda_*o3,-lambda_*o3, o3, lambda_*o3,-lambda_*o3], (1-N[2],-1,0,1,N[2]-1), N[2], N[2]).toarray()]

solutions1 = []
for j in range(len(N)):
    solutions1.append( np.zeros((N[j],K[j])) )
    
for k in range(3):
    #initializes conditions to start with
    solutions1[k][:,0] = f(x[k])
    for j in range(1,K[k]):
        #using matrix corresponding to steps, we sample. We then set initial equal to the old
        #so we can repeat this process
        solutions1[k][:,j] = A_CD[k].dot(solutions1[k][:,j-1])

f1_CD = solutions1[0]
f2_CD = solutions1[1]
f3_CD = solutions1[2]

plt.plot(x[0], f1_CD[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Central Method for Transport Equation' +',' + 'h= ' + str(1/2**5))

plt.plot(x[1], f2_CD[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Central Method for Transport Equation' +',' + 'h= ' + str(1/2**6))

plt.plot(x[2], f3_CD[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Central Method for Transport Equation' +',' + 'h= ' + str(1/2**7))

plt.plot(x[0], f1_CD[:,-1])
plt.plot(x[1], f2_CD[:,-1])
plt.plot(x[2], f3_CD[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Central Method for Transport Equation')
plt.legend(['h =' + str(1/2**5), 'h =' + str(1/2**6),'h =' + str(1/2**7)])

### LAX-FRIEDRICHS
# choose sigma  such that 2*sigma = lambda. In this setting, sigma = 1/4
#v^(n+1)_j = (I+kD_0)v^n_j + (1/4)*kh*(D_+D_-)v^n_j
#v^0_j = f_j

#for convenience. we can manipulate matrices without having to re-enter terms and whatnot
sigma1 = lambda_/2
c = np.array([(sigma1*lambda_- lambda_/2),(1-2*sigma1*lambda_), (sigma1*lambda_+ lambda_/2)])

A_LF = [spdiags( [c[2]*o1,c[0]*o1, c[1]*o1, c[2]*o1,c[0]*o1], (1-N[0],-1,0,1,N[0]-1), N[0], N[0]).toarray(), 
     spdiags( [c[2]*o2,c[0]*o2, c[1]*o2, c[2]*o2,c[0]*o2], (1-N[1],-1,0,1,N[1]-1), N[1], N[1]).toarray()
     , spdiags([c[2]*o3,c[0]*o3, c[1]*o3, c[2]*o3,c[0]*o3], (1-N[2],-1,0,1,N[2]-1), N[2], N[2]).toarray()]

solutions2 = []
for j in range(len(N)):
    solutions2.append( np.zeros((N[j],K[j])) )
    
for k in range(3):
    #initializes conditions to start with
    solutions2[k][:,0] = f(x[k])
    for j in range(1,K[k]):
        #using matrix corresponding to steps, we sample. We then set initial equal to the old
        #so we can repeat this process
        solutions2[k][:,j] = A_LF[k].dot(solutions2[k][:,j-1])

f1_LF = solutions2[0]
f2_LF = solutions2[1]
f3_LF = solutions2[2]

plt.plot(x[0], f1_LF[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Friedrichs Method for Transport Equation' +',' + 'h= ' + str(1/2**5))

plt.plot(x[1], f2_LF[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Friedrichs Method for Transport Equation' +',' + 'h= ' + str(1/2**6))

plt.plot(x[2], f3_LF[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Friedrichs Method for Transport Equation' +',' + 'h= ' + str(1/2**7))

plt.plot(x[0], f1_LF[:,-1])
plt.plot(x[1], f2_LF[:,-1])
plt.plot(x[2], f3_LF[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Friedrichs Method for Transport Equation')
plt.legend(['h =' + str(1/2**5), 'h =' + str(1/2**6),'h =' + str(1/2**7)])

### LAX-WINDROFF
#choose sigma such that 2*sigma = 1/lambda. In this setting, sigma = 1
#v^(n+1)_j = (I+kD_0)v^n_j + kh*(D_+D_-)v^n_j
#v^0_j = f_j

sigma2 = 1/(2*lambda_)
d = np.array([(sigma2*lambda_- lambda_/2),(1-2*sigma2*lambda_), (sigma2*lambda_+ lambda_/2)])

A_LW = [spdiags( [d[2]*o1,d[0]*o1, d[1]*o1, d[2]*o1,d[0]*o1], (1-N[0],-1,0,1,N[0]-1), N[0], N[0]).toarray(), 
     spdiags( [d[2]*o2,d[0]*o2, d[1]*o2, d[2]*o2,d[0]*o2], (1-N[1],-1,0,1,N[1]-1), N[1], N[1]).toarray()
     , spdiags([d[2]*o3,d[0]*o3, d[1]*o3, d[2]*o3,d[0]*o3], (1-N[2],-1,0,1,N[2]-1), N[2], N[2]).toarray()]

solutions3 = []
for j in range(len(N)):
    solutions3.append( np.zeros((N[j],K[j])) )
    
for k in range(3):
    #initializes conditions to start with
    solutions3[k][:,0] = f(x[k])
    for j in range(1,K[k]):
        #using matrix corresponding to steps, we sample. We then set initial equal to the old
        #so we can repeat this process
        solutions3[k][:,j] = A_LF[k].dot(solutions3[k][:,j-1])

f1_LW = solutions3[0]
f2_LW = solutions3[1]
f3_LW = solutions3[2]

plt.plot(x[0], f1_LW[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Wendroff Method for Transport Equation' +',' + 'h= ' + str(1/2**5))

plt.plot(x[1], f2_LW[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Wendroff Method for Transport Equation' +',' + 'h= ' + str(1/2**6))

plt.plot(x[2], f3_LW[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Wendroff Method for Transport Equation' +',' + 'h= ' + str(1/2**7))

plt.plot(x[0], f1_LW[:,-1])
plt.plot(x[1], f2_LW[:,-1])
plt.plot(x[2], f3_LW[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Lax-Wendroff Method for Transport Equation')
plt.legend(['h =' + str(1/2**5), 'h =' + str(1/2**6),'h =' + str(1/2**7)])

### BACKWARD EULER
#(I - kD_0)v^(n+1)_j = v^n_j
#v^0_j = f_j
A_BE = [spdiags( [-0.5*lambda_*o1,0.5*lambda_*o1, o1, -0.5*lambda_*o1,0.5*lambda_*o1], (1-N[0],-1,0,1,N[0]-1), N[0], N[0]).toarray(), 
     spdiags( [-0.5*lambda_*o2,0.5*lambda_*o2, o2, -0.5*lambda_*o2,0.5*lambda_*o2], (1-N[1],-1,0,1,N[1]-1), N[1], N[1] ).toarray()
     , spdiags( [-0.5*lambda_*o3,0.5*lambda_*o3, o3, -0.5*lambda_*o3,0.5*lambda_*o3], (1-N[2],-1,0,1,N[2]-1), N[2], N[2]).toarray()]

solutions4 = []
for j in range(len(N)):
    solutions4.append( np.zeros((N[j],K[j])) )
    

for k in range(3):
    #initializes conditions to start with
    solutions4[k][:,0] = f(x[k])
    C = np.linalg.inv(A_BE[k])
    for j in range(1,K[k]):
        #using matrix corresponding to steps, we sample. We then set initial equal to the old
        #so we can repeat this process
        solutions4[k][:,j] = C.dot(solutions4[k][:,j-1])

f1_BE = solutions4[0]
f2_BE = solutions4[1]
f3_BE = solutions4[2]

plt.plot(x[0], f1_BE[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Backwards-Euler Method for Transport Equation' +',' + 'h= ' + str(1/2**5))

plt.plot(x[1], f2_BE[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Backwards-Euler Method for Transport Equation' +',' + 'h= ' + str(1/2**6))

plt.plot(x[2], f3_BE[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Backwards-Euler Method for Transport Equation' +',' + 'h= ' + str(1/2**7))

plt.plot(x[0], f1_BE[:,-1])
plt.plot(x[1], f2_BE[:,-1])
plt.plot(x[2], f3_BE[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Backwards-Euler Method for Transport Equation')
plt.legend(['h =' + str(1/2**5), 'h =' + str(1/2**6),'h =' + str(1/2**7)])

### CRANK- NICHOLSON
# (I-0.5kD_0)v^(n+1)_j = (I+0.5kD_0)v^n_j
#v^0_j = f_j
theta = 1/2
#FOR CONVENIENCE (i.e. so I can still see what's written)
a = (1-theta)*lambda_/2
b = theta*lambda_/2

A_CN = [spdiags( [-a*o1,a*o1,o1,-a*o1,a*o1], (1-N[0],-1,0,1,N[0]-1), N[0], N[0]).toarray(), 
        spdiags( [-a*o2,a*o2,o2,-a*o2,a*o2], (1-N[1],-1,0,1,N[1]-1), N[1], N[1]).toarray(),
        spdiags( [-a*o3,a*o3,o3,-a*o3,a*o3], (1-N[2],-1,0,1,N[2]-1), N[2], N[2]).toarray()]



B_CN = [spdiags( [b*o1,-b*o1,o1,b*o1,-b*o1],  (1-N[0],-1,0,1,N[0]-1), N[0], N[0]).toarray(), 
        spdiags( [b*o2,-b*o2,o2,b*o2,-b*o2], (1-N[1],-1,0,1,N[1]-1), N[1], N[1]).toarray(),
        spdiags( [b*o3,-b*o3,o3,b*o3,-b*o3], (1-N[2],-1,0,1,N[2]-1), N[2], N[2]).toarray()]

solutions5 = []
for j in range(len(N)):
    solutions5.append( np.zeros((N[j],K[j])) )
    
for k in range(3):
    #initializes conditions to start with
    solutions5[k][:,0] = f(x[k])
    C = np.matmul(np.linalg.inv(A_CN[k]), B_CN[k])
    for j in range(1,K[k]):
        #using matrix corresponding to steps, we sample. We then set initial equal to the old
        #so we can repeat this process
        solutions5[k][:,j] = C.dot(solutions5[k][:,j-1])
        
f1_CN = solutions5[0]
f2_CN = solutions5[1]
f3_CN = solutions5[2]

plt.plot(x[0], f1_CN[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Crank-Nicholson Method for Transport Equation' +',' + 'h= ' + str(1/2**5))

plt.plot(x[1], f2_CN[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Crank-Nicholson Method for Transport Equation' +',' + 'h= ' + str(1/2**6))

plt.plot(x[2], f3_CN[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Crank-Nicholson Method for Transport Equation' +',' + 'h= ' + str(1/2**7))

plt.plot(x[0], f1_CN[:,-1])
plt.plot(x[1], f2_CN[:,-1])
plt.plot(x[2], f3_CN[:,-1])
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Crank-Nicholson Method for Transport Equation')
plt.legend(['h =' + str(1/2**5), 'h =' + str(1/2**6),'h =' + str(1/2**7)])


### Error calculations in h norm
def h_norm(x,y):
    norm = np.linalg.norm(x-y)
    return norm

### Theoreitcal soln
def u(x,t):
    #let's us define a piecewise function
    tau = 2*np.pi
    z = x+t
    z = z % tau
    y= np.piecewise(
        z,
        [(0 <= z)&(z <= np.pi),  (np.pi <= z)&(z <= 2*np.pi)     ],
        [lambda z: z, lambda z: 2*np.pi - (z), 0])
    return y
plt.plot(x[2],u(x[2],1))
plt.xlabel('x')
plt.ylabel('u(x,1)')
plt.title('Analytic Solution for Transport Equation')

### UPWIND, CENTRAL, LAX-WENDROFF, LAX-FRIEDRICHS, BACKWARDS EULER, CRANK-NICHOLSON
    
Upwind_error = h_norm(f3[:,-1],u(x[2],1))
Central_error = h_norm(f3_CD[:,-1],u(x[2],1))
Lax_Friedrichs_error = h_norm(f3_LF[:,-1],u(x[2],1))
Lax_Wendroff_error = h_norm(f3_LW[:,-1],u(x[2],1))
Backwards_Euler_Error = h_norm(f3_BE[:,-1],u(x[2],1))
Crank_Nicholson_Error = h_norm(f3_CN[:,-1],u(x[2],1))

