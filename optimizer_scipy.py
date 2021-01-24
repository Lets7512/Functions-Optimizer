import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import parser
import re
from numpy import *
import warnings
warnings.filterwarnings("ignore")
from os import system

def parse_input(formula_input,x):
    formula = formula_input.replace("^","**") # formating input
    #formula = formula.replace("e","exp(1)") # formating input
    if formula_input.find("log") !=-1:
        formula = formula.replace("log","log10") # formating input
    formula = formula.replace("ln","log") # formating input
    formula = re.sub("(?!xp)(x)","x[0]",formula)
    formula = re.sub("y","x[1]",formula)
    code = parser.expr(formula).compile()
    z = eval(code)
    return z


def cost(x):
    if not calc_grad:
        grad_x_cost.append(x[0])
        grad_y_cost.append(x[1])
    return parse_input(formula_cost,x)

def g1(x):
    return parse_input(formula_g1,x)
def g2(x):
    return parse_input(formula_g2,x)


def plot_3d_surface(cost,x,y,show=True,title="Cost Function"):
    fig = plt.figure(figsize=[10*1.7,6*1.7])
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(x, y)
    Z_mesh=[]
    for x in zip(X,Y):
        z = cost(x)
        Z_mesh.append(z)
    Z_mesh=np.array(Z_mesh)
    ax.plot_surface(X, Y, Z_mesh,cmap="YlGnBu",alpha=0.5)
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
    #ax.set_title("3D Plotting")
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

def scipy_opt_cost(funct,initial_guess,x_range,y_range,constraints1,constraints2):
    cons = [{'type':'ineq', 'fun': constraints1},{'type':'ineq', 'fun': constraints2}]
    result = minimize(funct, initial_guess,method='BFGS',bounds=[x_range,y_range],constraints=cons)
    return result




def main():
    global formula_cost,formula_g1,formula_g2,grad_x_cost,grad_y_cost
    global calc_grad
    calc_grad=False
    grad_x_cost=[]
    grad_y_cost=[]
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


    print("Enter X Range (  default = (-7,7)  leave empty for default values) with comma(,) separated:",end=" ")
    x_r=input()
    if x_r !="":
        xr_min,xr_max=[float(i) for i in x_r.split(",")]
    print("")


    print("Enter Y Range (  default = (-7,7)  leave empty for default values) with comma(,) separated:",end=" ")
    y_r=input()
    if y_r !="":
        yr_min,yr_max=[float(i) for i in y_r.split(",")]
    print("")

    print("Enter initial point (x,y) with comma(,) separated:",end=" ")
    xi,yi=[float(i) for i in input().split(",")]
    print("")


    x = np.linspace(xr_min,xr_max, int((abs(xr_min)+abs(xr_max)) *100))
    y = np.linspace(yr_min,yr_max, int((abs(yr_min)+abs(yr_max)) *100))

    result=scipy_opt_cost(cost,[xi,yi],[xr_min,xr_max],[yr_min,yr_max],g1,g2)
    calc_grad = True
    iterations = result.nit
    mini=result.x
    print("The minimum Z value is found to be ",round(cost(mini),3)," at (",round(mini[0],3),",",round(mini[1],3),")"," within the specified ranges, learning rate and iterations")
    ax = plot_3d_surface(cost,x,y,show=False,title="Cost Function")


    grad_z_cost=[]
    grad_z_g1=[]
    grad_z_g2=[]
    for x in zip(grad_x_cost,grad_y_cost):
        grad_z_cost.append(cost(x))
        grad_z_g1.append(g1(x))
        grad_z_g2.append(g2(x))
    plot_3d_points_on_surface(grad_x_cost,grad_y_cost,grad_z_cost,ax=ax,show=False)
    plot_2d(grad_z_cost,title="$f{x,y}$ vs iterations",show=False)
    plot_2d(grad_z_g1,title="g1 vs iterations",show=False)
    plot_2d(grad_z_g2,title="g2 vs iterations",show=True)

if __name__ == "__main__":
    main()
    system('pause')