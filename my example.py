#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
np.seterr(invalid="ignore")


# In[2]:


D = 0.1      #seperation of two adjacent wells  
L1 = 2       #width of the first well
L2 = 1       #width of the second well
gamma1  = 100  #absolute value of the depth of the first well / unitless 
gamma2  = 80 #absolute value of the depth of the second well / unitless 
h_barrier = 0


# In[3]:


def V(x, V0 = -gamma2): #construction of wells 
    if np.iterable(x):
        return np.array([V(xi, V0) for xi in x])
    elif x < 0:
        return 0
    elif x >= 0 and x < L1: #the first well
        return -gamma1
    elif x >= L1 and x < L1 + D: #the barrier
        return h_barrier
    elif x >= L1 + D  and x < L1 + L2 + D:
        return -gamma2
    elif x >= L1 + L2 + D:
        return 0

#test run 
xd = np.linspace(0-1, L1+L2+D+1, 11)
xd_plot = np.linspace(xd.min(), xd.max(), 100)
if gamma1 < gamma2:
    print(plt.ylim(-gamma2-1,0+2))
else:
    print(plt.ylim(-gamma1-1,0+2))
plt.plot(xd_plot, V(xd_plot) ,drawstyle='steps-mid', c='k', alpha=0.5)


# In[4]:


def fund(x, y, p): #the function of psi in finite well 
    E = p[0]
    return np.vstack((y[1], -(E-V(x))*y[0]))


# In[5]:


def bcd(ya, yb, p): #the boundary condition
    return np.array([ya[0], yb[0], ya[1] - 0.001])


# In[6]:


x1 = np.linspace(-0.73, L1+L2+D+0.73, 11) #iteration of the box


# In[7]:


y_d = np.zeros((2, x1.size))
y_d[0,4] = 1 #boundary guess for shooting


# In[8]:


def fk1(E, V0=gamma1, a=L1, n=1):
    k1 = np.sqrt(E)
    k2 = np.sqrt((V0-E))
    if n % 2:
        return k2 - k1 * np.tan(k1 * a / 2)
    else:
        return k2 + k1 / np.tan(k1 * a / 2)

def Eanalytic1(V0=gamma1, a=L1, pts=1000):
    """Finds the roots of the fk between 0 and V0 for odd and even n."""
    Ei = np.linspace(0.0, V0, pts)
    roots = []
    for n in [1, 2]:
        for i in range(pts - 1):
            soln = root_scalar(fk1, args=(V0, a, n), x0=Ei[i], x1=Ei[i + 1])
            if soln.converged and np.around(soln.root, 9) not in roots:
               roots.append(np.around(soln.root, 9))
    return np.sort(roots)

def fk2(E, V0=gamma2, a=L2, n=1):
    k1 = np.sqrt(E)
    k2 = np.sqrt((V0-E))
    if n % 2:
        return k2 - k1 * np.tan(k1 * a / 2)
    else:
        return k2 + k1 / np.tan(k1 * a / 2)

def Eanalytic2(V0=gamma2, a=L2, pts=1000):
    """Finds the roots of the fk between 0 and V0 for odd and even n."""
    Ei = np.linspace(0.0, V0, pts)
    roots = []
    for n in [1, 2]:
        for i in range(pts - 1):
            soln = root_scalar(fk2, args=(V0, a, n), x0=Ei[i], x1=Ei[i + 1])
            if soln.converged and np.around(soln.root, 9) not in roots:
               roots.append(np.around(soln.root, 9))
    return np.sort(roots)

elist1 = Eanalytic1() - gamma1
elist2 = Eanalytic2() - gamma2
print(elist1)
print(elist2)


# In[9]:


testlist = np.linspace(-gamma1,0,gamma1*4) #generate points 


# In[10]:


start_time = time.time() 
solnsd1 = [solve_bvp(fund, bcd, x1, y_d, p=[Ed]) for Ed in testlist]
print("--- %s seconds ---" % (time.time() - start_time))


# In[11]:


xd = np.linspace(0-1, L1+L2+D+1, 11) 
xd_plot = np.linspace(x1.min(), x1.max(), 100)

if gamma1 < gamma2: 
    print(plt.ylim(-gamma2-1, 0+2))
else:
    print(plt.ylim(-gamma1-1, 0+2))
plt.plot(xd_plot, V(xd_plot) , drawstyle='steps-mid', c='k', alpha=0.5)

for soln in solnsd1:
    y_plot = soln.sol(xd_plot)[0]
    l = plt.plot(xd_plot, 2 * y_plot / y_plot.max() + soln.p[0],linewidth = 0.6)
    plt.axhline(soln.p[0], ls='--', c=l[0].get_color(), linewidth = 0.6)
      
plt.xlabel(r'$x$')
plt.ylabel(r'$\psi(x)$')
plt.title("Energy Levels (D=0.3)")
plt.xlabel("z")
plt.ylabel("Energy")
plt.savefig('schro.png', dpi=1000, bbox_inches='tight')
plt.show()


# In[12]:


xvals1 = np.linspace(1, len(elist1),len(elist1))
xvals2 = np.linspace(1, len(elist2),len(elist2))
yfin1 = elist1 + np.linspace(gamma1, gamma1, len(elist1))
yfin2 = elist2 + np.linspace(gamma2, gamma2, len(elist2))
yinf1 = []
yinf2 = []

for n in range(1, len(elist1) + 1):
    yinf1.append((n**2) * yfin1[0])
for n in range(1, len(elist2) + 1):
    yinf2.append((n**2) * yfin2[0])
   
plt.plot(xvals1, yfin1, 's-', alpha=1, color = 'red', markersize=7, label='Finite Well_1')
plt.plot(xvals2, yfin2, 'o-', alpha=1, color = 'navy', markersize=7, label='Finite Well_2')
plt.plot(xvals1, yinf1, 's--', alpha=0.2, color = 'red', markersize=5, label = "Infinite Well_1")
plt.plot(xvals2, yinf2, 'o--', alpha=0.2, color = 'navy', markersize=5, label = "Infinite Well_2")
plt.xlabel("n", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.title('Infinite & Finite Wells Energy levels')
plt.legend(loc="best", frameon=True) #'loc' moves the legend around, frameon puts a box around the legend
plt.savefig("comparison", dpi=800, bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




