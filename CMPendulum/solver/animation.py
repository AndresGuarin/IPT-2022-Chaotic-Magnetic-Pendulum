"""
Created on Fri Jun  3 22:01:50 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_path(self,save={'on':False}):
    """
    @params:
        self: self object that contains the data of 
              CMPendulum.solver.pendulum.pendulum() object
        save: dictionary that contains the name of the video
              and enable or disble the saving process
              
              save={'on':True, 'name':'name.mp4'}
    """
    
    #local function for update each frame
    def update(j):
        
        # Clear the before plot
        ax.clear()
        
        # Plot the trajectories until the interval j
        ax.plot(self.X[:j],self.Y[:j],'-b',linewidth=0.5)
        ax.plot(self.X[j],self.Y[j],'o',markersize=5,color='b')
    
        # Plot the magnets
        for i in range(self.NMAGNETS):
            ax.plot(self.Mx[i],self.My[i],'or',markersize=10)
            ax.text(self.Mx[i]-0.02,self.My[i]-0.04,f'{np.round(self.mu_magn[i]*self.sign[i],2)}', fontsize=8)
        
        # Plot initial and final position
        ax.plot(self.X[0],self.Y[0],'Dg',label=f'Initial position ({np.round(self.X[0],3)},{np.round(self.Y[0],3)})', markersize=5)
        ax.plot(self.X[-1],self.Y[-1],'Dc',label='Final position', markersize=5)
        
        # Finding net magnetic Force
        FBx = self.FB[j][0]
        FBy = self.FB[j][1]
        FB_xy_mag = np.sqrt(FBx**2+FBy**2)
        
        # Plot net magnetic force vector
        ax.quiver(self.X[j],self.Y[j],0.25*FBx/FB_xy_mag,0.25*FBy/FB_xy_mag,color='red', label='Net Fb',angles='xy', scale_units='xy', scale=2)
        
        # Put on legend the Iteration and time of the frame
        ax.plot(0,0,'o',markersize=0, label=f'Iteration {j}')
        ax.plot(0,0,'o',markersize=0, label=f'Time {np.round(j*self.h,2)}s')
        
        # Put grids, legends, and limits
        plt.grid()
        plt.legend()
        plt.xlim(-self.lim,self.lim)
        plt.ylim(-self.lim,self.lim) 
    
    # Create figure and axis
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    
    # Put labels, title, grids, and legend
    plt.xlabel('X-axis [m]')
    plt.ylabel('Y-axis [m]')
    plt.title('Pendulum magnet trajectory')
    plt.grid()
    plt.legend()
    
    # Approximately Time per each plot
    f = 1/(0.25) 
    
    # Number of intervals that animation show per frame. Higher 'steps' lower the time of animation. steps < N
    steps=int(1/(self.h*f)) 
    
    # Number of intervals
    N = len(self.X)
    
    # Animate the movement
    self.ani=animation.FuncAnimation(fig,update,range(10,N,steps), repeat=False) 
    
    if save['on']:
        self.ani.save(save['name'],writer='ffmpeg')
        plt.show(self.ani)
    else: 
        # Show animation
        plt.show()