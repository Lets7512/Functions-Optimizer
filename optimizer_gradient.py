import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import parser
from numpy import *
import warnings
warnings.filterwarnings("ignore")
from os import system






def parse_input(formula_input,x_in,y_in):
    formula = formula_input.replace("^","**") # formating input
    #formula = formula.replace("e","exp(1)") # formating input
    if formula_input.find("log") !=-1:
        formula = formula.replace("log","log10") # formating input
    formula = formula.replace("ln","log") # formating input
    code = parser.expr(formula).compile()
    x,y=x_in,y_in
    z = eval(code)
    return z

def cost(x,y):
    if type(x) != type([]):
        return parse_input(formula_cost,x,y)
        
    if type(x) == type([]):
        z=[]
        for i,j in zip(x,y):
            z.append(parse_input(formula_cost,i,j))
        return z

def g1(x,y):
    if type(x) != type([]):
        return parse_input(formula_g1,x,y)
        
    if type(x) == type([]):
        z=[]
        for i,j in zip(x,y):
            z.append(parse_input(formula_g1,i,j))
        return z
def g2(x,y):
    if type(x) != type([]):
        return parse_input(formula_g2,x,y)
        
    if type(x) == type([]):
        z=[]
        for i,j in zip(x,y):
            z.append(parse_input(formula_g2,i,j))
        return z


def constraints1(x,y):
    return (g1(x,y) < 0)
def constraints2(x,y):
    return (g2(x,y) < 0)

def d_cost(fun,x,y,var): # var = 0 for dx , var = 1 for dy
    h = 5*1e-10
    if var == 0:
        y=h
        return (fun(x+h,y)-fun(x,y))/(h)
    if var == 1:
        x=h
        return (fun(x,y+h)-fun(x,y))/(h)


def plot_3d_surface(cost,x,y,show=True,title="Cost Function"):
    fig = plt.figure(figsize=[10*1.25,6*1.25])
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(x, y)
    Z = cost(X,Y)
    ax.plot_surface(X, Y, Z,cmap="YlGnBu",alpha=0.5)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_title("3D Plotting of "+title)
    if show:
        plt.show()
    else:
        return ax

def plot_3d_points_on_surface(x,y,z,ax,show=True):
    ax.scatter(x, y, z,color="red",alpha=0.7,marker="x")
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_title("3D Plotting")
    if show:
        plt.show()


def plot_2d(z_grad,title="Functions vs Iterations",show=True):
    fig1 = plt.figure(figsize=[10,6])
    plt.plot(range(len(z_grad)),z_grad)
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title(title)
    plt.xlim([0,len(z_grad)])
    if show:
        plt.show()


#As for the optimizer we will apply the concept of gradient descent. a famous optimizing technique in machine learning.
def Gradient_descent(cost,xi,yi,x_range,y_range,alpha=0.001,iterations=1000): #alpha is the (learning rate)
    pre_step_size_z = 1
    precision = 1e-10
    current_x=xi
    current_y=yi
    grad_x_history=[]
    grad_y_history=[]
    grad_z_history=[]
    for i in range(iterations):
        previous_x = current_x
        previous_y = current_y
        previous_z = cost(previous_x,previous_y)
        current_x = current_x - alpha * d_cost(cost,previous_x,previous_y,0)
        current_y = current_y - alpha * d_cost(cost,previous_x,previous_y,1)
        current_z = cost(current_x,current_y)
        #print(current_x,current_y)
        if constraints1(current_x,current_y) and constraints2(current_x,current_y):
            grad_x_history.append(current_x)
            grad_y_history.append(current_y)
            grad_z_history.append(current_z)
            pre_step_size_x = abs(current_x - previous_x)
            pre_step_size_y = abs(current_y - previous_y)
            pre_step_size_z = abs(current_z - previous_z)
            #print(pre_step_size_z)
        if pre_step_size_z < precision:
            break
        if (x_range.min() > current_x or x_range.max() < current_x) or (y_range.min() > current_y or y_range.max() < current_y):
            print("Broke limit")
            break
    print("The minimum Z value is found to be ",round(current_z,2)," at (",round(current_x,2),",",round(current_y,2),")"," within the specified ranges, learning rate and iterations")
    return grad_x_history,grad_y_history,grad_z_history





def main():
    global formula_cost,formula_g1,formula_g2
    xr_min,xr_max = -7,7
    yr_min,yr_max = -7,7
    print("Enter Cost Function :",end=" ")
    formula_cost=input()
    print("")
    print("Enter g1 :",end=" ")
    formula_g1=input()
    print("")
    print("Enter g2 :",end=" ")
    formula_g2=input()
    print("")


    print("Enter X Range (  default = (-7,7)  leave empty for default values) with parenthesis and with comma(,) separated:",end=" ")
    x_r=input()
    if x_r !="":
        xr_min,xr_max=[float(i) for i in x_r.split(",")]
    print("")


    print("Enter Y Range (  default = (-7,7)  leave empty for default values) with parenthesis and with comma(,) separated:",end=" ")
    y_r=input()
    if y_r !="":
        yr_min,yr_max=[float(i) for i in y_r.split(",")]
    print("")

    print("Enter initial point (x,y) with parenthesis and with comma(,) separated:",end=" ")
    xi,yi=[float(i) for i in input().split(",")]
    print("")


    x = np.linspace(xr_min,xr_max, int((abs(xr_min)+abs(xr_max)) *100))
    y = np.linspace(yr_min,yr_max, int((abs(yr_min)+abs(yr_max)) *100))
    iteratn=2000

    x_grad,y_grad,z_grad=Gradient_descent(cost,xi,yi,x,y,iterations=iteratn)
    grad_z_g1=g1(x_grad,y_grad)
    grad_z_g2=g2(x_grad,y_grad)

    ax = plot_3d_surface(cost,x,y,show=False,title="Cost Function")
    plot_3d_points_on_surface(x_grad,y_grad,z_grad,ax=ax,show=False)
    plot_2d(z_grad,title="f${x,y}$ vs iterations",show=False)
    plot_2d(grad_z_g1,title="g1 vs iterations",show=False)
    plot_2d(grad_z_g2,title="g2 vs iterations",show=True)

if __name__ == "__main__":
    main()
    system('pause')

