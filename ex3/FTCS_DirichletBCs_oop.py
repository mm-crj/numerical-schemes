'''
Forward method to solve 1D reaction-diffusion equation:
    u_t = D * u_xx + alpha * u
    
with Dirichlet boundary conditions u(x0,t) = 0, u(xL,t) = 0
and initial condition u(x,0) = 4*x - 4*x**2
'''


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class FTCSDirichlet:
    
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.x0 = 0
        self.xL = 1.0
        self.t0 = 0
        self.tF = 0.2
        self.D = 0.1 # Diffusion coefficient
        self.alpha = -3 # Reaction rate
        self.createGrid()
        self.params()
        self.solve()
        
    def createGrid(self):
        self.xspan = np.linspace(self.x0, self.xL, self.M)
        self.tspan = np.linspace(self.t0, self.tF, self.N)
        
    def params(self):
        # ----- Spatial discretization step -----
        dx = (self.xL - self.x0)/(self.M - 1)
        # ----- Time step -----
        dt = (self.tF - self.t0)/(self.N - 1)
        self.r = dt*self.D/dx**2
        self.s = dt*self.alpha
        
    def solve(self):
        # ----- Initializes matrix solution U -----
        self.U = np.zeros((self.M, self.N))
        # ----- Initial condition -----
        self.U[:,0] = 4*self.xspan - 4*self.xspan**2
        # ----- Dirichlet Boundary Conditions -----
        self.U[0,:] = 0.0
        self.U[-1,:] = 0.0
        
        # ----- Equation (15.8) in Lecture 15 -----
        for k in range(0, self.N-1):
            for i in range(1, self.M-1):
                self.U[i, k+1] = self.r*self.U[i-1, k] + (1-2*self.r+self.s)*self.U[i,k] + self.r*self.U[i+1,k] 
                
        self.T, self.X = np.meshgrid(self.tspan, self.xspan)
        
    def plot_(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        ax.plot_surface(self.X, self.T, self.U, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])

        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('U')
        ax.view_init(elev=33, azim=36)
        plt.tight_layout()
        plt.show()


def main():
    sim = FTCSDirichlet(40, 70)
    sim.plot_()
    
if __name__ == "__main__":
    main()
    
#M = 40 # GRID POINTS on space interval
#N = 70 # GRID POINTS on time interval