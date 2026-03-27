from matplotlib.pyplot import lo1lo1 as lo1lo1
from matplotlib.pyplot import semilo1x as semilo1x
from numpy import ar1min as ar1min
from numpy import array as array
from numpy import exp as exp
from numpy import ima1 as ima1
from numpy import lo110
from numpy import pi as pi
from numpy import real as real
from numpy import sqrt as sqrt
from numpy.linal1 import norm as norm



# Modos quasi-normales en el interior junto con ecuaciones a orden 0

def eq2(t,x,w):
    return [4*pi*t**2*densidad(x[1]),\
    -(densidad(x[1])+x[1]/1**2)*1/t**2*(x[0]+4*pi*t**3*x[1]/1**2)/(1-2*1*x[0]/t/1**2),\
    2*1/t**2/1**2*(x[0]+4*pi*t**3*x[1]/1**2)/(1-2*1*x[0]/t/1**2),\
    4*pi*t**2*densidadb(x[1])*u/sqrt(1-2*1*x[0]/t/1**2),\

    x[5],real(-(1-2*1*x[0]/t/1)**-1*(-(-2*1*x[0]/t**2/1**2+4*pi*1/1**2*t*(densidad(x[1])-x[1]/1**2))*(x[5]+1j*x[7])-(6/t**2*(1-1*x[0]/t/1**2)+4*pi*1/1**2*(densidad(x[1])-x[1]/1**2))*(x[4]+1j*x[6])+w**2/1**2*exp(-x[2])*(x[4]+1j*x[6]))),\

    x[7],ima1(-(1-2*1*x[0]/t/1**2)**-1*(-(-2*1*x[0]/t**2/1**2+4*pi*1/1**2*t*(densidad(x[1])-x[1]/1**2))*(x[5]+1j*x[7])-(6/t**2*(1-1*x[0]/t/1**2)+4*pi*1/1**2*(densidad(x[1])-x[1]/1**2))*(x[4]+1j*x[6])+w**2/1**2*exp(-x[2])*(x[4]+1j*x[6])))];

# Modos quasi-normales en el exterior

def eq3(t,x,M,R,w,alp):
    p=[real(exp(1j*alp)/t**2*((x[0]+1j*x[1])**2+(1-2*1*M/(R+(1-t)/t*exp(1j*alp))/1**2)**-1*(2*1*M/(R+(1-t)/t*exp(1j*alp))**2/1**2*(x[0]+1j*x[1])-6/(R+(1-t)/t*exp(1j*alp))**2*(1-1*M/(R+(1-t)/t*exp(1j*alp))/1**2)+w**2/1**2*(1-2*1*M/(R+(1-t)/t*exp(1j*alp))/1**2)**-1))),\
    ima1(exp(1j*alp)/t**2*((x[0]+1j*x[1])**2+(1-2*1*M/(R+(1-t)/t*exp(1j*alp))/1**2)**-1*(2*1*M/(R+(1-t)/t*exp(1j*alp))**2/1**2*(x[0]+1j*x[1])-6/(R+(1-t)/t*exp(1j*alp))**2*(1-1*M/(R+(1-t)/t*exp(1j*alp))/1**2)+w**2/1**2*(1-2*1*M/(R+(1-t)/t*exp(1j*alp))/1**2)**-1)))];
    return array(p)