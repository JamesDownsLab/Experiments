import numpy as np

g = 9.81
v2a = 42.46

def d2a(d):
    """Converts duty cycle to acceleration ms^-2"""
    a = v2a * 0.003 * d - v2a * 1.5
    return a

def d2G(d):
    """Converts duty cycle to dimensionless acceleration"""
    return d2a(d) / g

def a2d(a):
    d = (a + v2a*1.5)/(v2a*0.003)
    return d

def G2d(a):
    d = a2d(a*g)
    return d

if __name__ == '__main__':
    print(d2G(600)-d2G(599))