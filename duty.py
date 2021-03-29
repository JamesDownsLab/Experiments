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

if __name__ == '__main__':
    print(d2G(700))