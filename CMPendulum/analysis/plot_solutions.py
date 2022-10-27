"""
Created on Thu Oct  6 16:04:02 2022

@author: AndresGuarin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  #Color maps

# In[1]-------------------------- Plot Trayectories----------------------------

# Path3D
def path3D(self):
    X = self.X
    Y = self.Y
    d = self.d
    l = self.l
    Z = - np.sqrt(l**2 - X**2 - Y**2) + (l+d)
    plt.figure(figsize=(5,4))
    ax = plt.axes(projection='3d')
    ax.plot3D(X,Y,Z,'gray')

# Plot trajectory
def path(self):    
    plt.plot(self.X,self.Y,'-b',linewidth=0.5)
    
    # Initial and final position
    plt.plot(self.X[0],self.Y[0],'Dg',label=f'Initial position ({np.round(self.X[0],3)},{np.round(self.Y[0],3)})', markersize=5)
    plt.plot(self.X[-1],self.Y[-1],'Dc',label='Final position', markersize=5)
    
    # Put legends, title, labales, and ticks
    plt.legend(fontsize=12)
    plt.title('Trajectory of the magnet',fontsize=15)
    plt.xlabel('x-axis [m]',fontsize=12)
    plt.ylabel('y-axis [m]',fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

# Phase portrait
def phase_portrait(self, fmt='-', color='red', linewidth=0.8, label='off'):
    # Get positions and velocity
    X=self.X
    Y=self.Y
    Vx=self.Vx
    Vy=self.Vy
    V = np.sqrt(Vx**2+Vy**2)*np.sign(Vy) #With sign

    # Create projection 3d
    plt.figure(figsize=(5,5))
    ax = plt.axes(projection='3d')

    # Plot phase portrait
    if label == 'off':
        plt.plot(X,Y,V,fmt,color=color,linewidth=linewidth)
    else:
        plt.plot(X,Y,V,fmt,color=color,linewidth=linewidth,label=label)
    # Put tilte and labels
    plt.title('Phase portrait 3D', fontsize=15)
    plt.xlabel('X [m]',fontsize=12)
    plt.ylabel('Y [m]',fontsize=12)
    ax.set_zlabel('Z [m]',fontsize=12)
    plt.legend()

# Evolution of positions X, Y
def time_series(self, X, fmt='-',color='blue',linewidth=0.8,alpha=1, label='off'):
    plt.figure(figsize=(5,5))
    
    # Create time array
    t=np.linspace(0,self.h*self.N,len(X))

    # Plot time series
    if label=='off':
        plt.plot(t,X,fmt,color,linewidth=linewidth,alpha=alpha)
    else:
        plt.plot(t,X,fmt,color,linewidth=linewidth,alpha=alpha, label=label)
        plt.legend()
        
    # Put the title and labels
    plt.title('Time series of X',fontsize=15)
    plt.xlabel('Time [s]',fontsize=12)
    plt.ylabel('X [m]',fontsize=12)
    plt.xticks(fontsize=9); plt.yticks(fontsize=9)

# In[2]------------------------- Plot table and vector space-------------------
# Plot border of table
def table(self):
    # Plot Table
    plt.figure(figsize=(6,6))
    a = self.Table_SIZE/2 +0.01
    plt.plot([-a,-a,a,a,-a], [-a,a,a,-a,-a], '-b', label='Table')
    
    # Put legends, title, labales, and limits
    plt.legend(loc='upper right')
    plt.title('Positions of magnets')
    plt.xlabel('x-axis [m]')
    plt.ylabel('y-axis [m]')
    plt.xlim(-self.lim,self.lim)
    plt.ylim(-self.lim,self.lim)

# Plot table and magnets
def alltable(self):
    Mx = self.Mx
    My = self.My
    mu = self.mu_magn
    table(self)
    
    # Plot magnets
    for i in range(len(Mx)):
        plt.plot(Mx[i],My[i],'or')
        plt.text(Mx[i]+0.005,My[i]+0.005,f'{i+1}')
        plt.text(Mx[i]-0.02,My[i]-0.02,'%.2f [$Am^2$]'%(mu[i]),fontsize=8)
    plt.plot(Mx[0], My[0], 'or', label='Magnets')
    
    # Put limits, labels, legend and title
    lim = self.Table_SIZE/2 + 0.04
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xlabel('X-axis [m]')
    plt.ylabel('Y-axis [m]')
    plt.legend(loc='upper right')
    plt.title('Table Magnets')

# Plot vector space of magnetic forces
def vector_space(self,res=20,a=0, net_forces=True, magnetic=False, tension=False):
    
    ## Put limits of plot
    if a==0:
        a=self.lim_vs # Predefined value
    elif a > self.l/2:
        a = self.l/2  # Ensure that the points are ploted in the zone inside the spheric pendulum
    
    ## Create data of vector tails
    t   = np.linspace(-a,a,res)
    NUM = len(t)
    X,Y = np.meshgrid(t,t)
    Z   = -np.sqrt(self.l**2 - X**2 - Y**2)+self.l+self.d
    
    ## Find net magnetic force vector space
    Fx = np.zeros((NUM,NUM))
    Fy = np.zeros((NUM,NUM))
    Fz = np.zeros((NUM,NUM))
    
    # Unit vector of the magnetic dipole m1 (from Q to P)
    M1x = X/self.l                 #X component 
    M1y = Y/self.l                 #Y component 
    M1z = (Z-self.l-self.d)/self.l #Z component 
    
    # Iterating over table magnets
    for i in range(self.NMAGNETS): 
        R = (self.Mx[i]-X)**2 + (self.My[i]-Y)**2 + Z**2  #Squared distance between magnet i and pendulum magnet
        m2 = self.mu_dir[i]                               #Table magnet constants
        
        #Unit vector from m2 (on table) to m1 (in pendulum)
        Ur_x = (X-self.Mx[i])/np.sqrt(R) #X component 
        Ur_y = (Y-self.My[i])/np.sqrt(R) #Y component 
        Ur_z = Z/np.sqrt(R)              #Z component 
        
        #Dot products
        M12 = M1x*m2[0] + M1y*m2[1] + M1z*m2[2]     #Dot products between m1 and m2
        M1r = M1x*Ur_x + M1y*Ur_y + M1z*Ur_z        #Dot products between m1 and ur
        M2r = m2[0]*Ur_x + m2[1]*Ur_y + m2[2]*Ur_z  #Dot products between m2 and ur
        
        #Sum contributions of all magnets
        Fx = Fx + self.S[i] / R**2 * ( (M12)*Ur_x + (M2r)*M1x + (M1r)*m2[0] - 5*(M1r)*(M2r)*Ur_x )
        Fy = Fy + self.S[i] / R**2 * ( (M12)*Ur_y + (M2r)*M1y + (M1r)*m2[1] - 5*(M1r)*(M2r)*Ur_y )
        Fz = Fz + self.S[i] / R**2 * ( (M12)*Ur_z + (M2r)*M1z + (M1r)*m2[0] - 5*(M1r)*(M2r)*Ur_z )
    
    F_xy_mag = np.sqrt(Fx**2 + Fy**2) # Magnitude of the projection of net magnetic force on plane xy
    
    ## Find tension vector space
    # Unit vector of tension (from P to Q) 
    UT_x = -X/self.l                #X component 
    UT_y = -Y/self.l                #Y component 
    UT_z = (self.l+self.d-Z)/self.l #Z component 

    FbUT = Fx*UT_x + Fy*UT_y + Fz*UT_z  #Dot product between net Fb and UT
    
    Tx = (self.m*9.8*(self.l+self.d-Z)/self.l - FbUT)*UT_x
    Ty = (self.m*9.8*(self.l+self.d-Z)/self.l - FbUT)*UT_y
    Tz = (self.m*9.8*(self.l+self.d-Z)/self.l - FbUT)*UT_z
    
    T_xy_mag = np.sqrt(Tx**2 + Ty**2)   #Magnitude of the projection of tension force on plane xy
    
    ## Find net force vector space (refusing friction force)
    Field_x = Fx+Tx
    Field_y = Fy+Ty
    Field_z = Fz+Tz
    
    Field_xy_mag = np.sqrt(Field_x**2 + Field_y**2) #Magnitude of the projection of force field on plane xy
        
    ## Plotting vectors
    # Field Forces
    if net_forces:
        plt.quiver(X,Y,Field_x/Field_xy_mag, Field_y/Field_xy_mag, width=0.0025, angles='xy', scale_units='width', scale=36, label='field forces')
    
    # Magnetic net forces
    if magnetic:
        plt.quiver(X,Y,Fx/F_xy_mag,Fy/F_xy_mag, width=0.0025, angles='xy', scale_units='width', scale=39, label='magnetic forces')                
    
    # Tension forces
    if tension:
        plt.quiver(X,Y,Tx/T_xy_mag, Ty/T_xy_mag, width=0.0025, angles='xy', scale_units='width', scale=37, label='tension forces')
 

# In[3]------------------------- Plot Energies---------------------------------
# Plot potential of the magnetic and gravitational force
def potential(self,res=20, a=0):
    
    # Put limits of plot
    if a==0:
        a = self.lim_vs #predifined value
    elif a > self.l:
        a = self.l/2 #Ensure that the points are ploted in the zone inside the spheric pendulum
    
    # Create data of the plot
    t   = np.linspace(-a,a,res)
    NUM = len(t)
    X,Y = np.meshgrid(t,t)
    Z   = -np.sqrt(self.l**2 - X**2 - Y**2)+self.l+self.d

    # Finding magnetic field B
    Bx = np.zeros((NUM,NUM))
    By = np.zeros((NUM,NUM))
    Bz = np.zeros((NUM,NUM))
    
    # Iterating over each table magnet
    for i in range(self.NMAGNETS):
        R = (self.Mx[i]-X)**2 + (self.My[i]-Y)**2 + Z**2   # Squared distance between magnet i and the pendulum magnet
    
        #Table magnet constants
        m2 = self.mu_dir[i]
        Cm = -(4*np.pi*10**-7 * self.mu_magn[i]) / (4*np.pi)
        
        #Unit vector from m2 (on table) to m1 (in pendulum)
        Ur_x = (X-self.Mx[i])/np.sqrt(R) #X component
        Ur_y = (Y-self.My[i])/np.sqrt(R) #Y component 
        Ur_z = Z/np.sqrt(R)              #Z component 
        
        #Dot products
        M2r = m2[0]*Ur_x + m2[1]*Ur_y + m2[2]*Ur_z  #Dot products between m2 and ur
        
        #Sum contributions of all magnets
        Bx = Bx + Cm/R**(3/2) * (m2[0] - 3*M2r*Ur_x)
        By = By + Cm/R**(3/2) * (m2[1] - 3*M2r*Ur_y)
        Bz = Bz + Cm/R**(3/2) * (m2[2] - 3*M2r*Ur_z)
    
    ## Finding magnetic potential
    
    # Unit vector from Q to P (direction of m1)
    Um1_x = X/self.l                 #X component  
    Um1_y = Y/self.l                 #Y component 
    Um1_z = (Z-self.l-self.d)/self.l #Z component 
    
    m1 = self.mu_P_magn                         # Pendulum magnet constants
    Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz) # Magnetic potential
    Ug = self.m*9.8*(Z-self.d)                  # Gravitational potential
    
    # Plot potential
    plt.figure(figsize=(5,4))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,Y,Ug+Um, cmap=cm.jet, edgecolor='none')

# Table of Potential, Kinetic, and Mechanical Energy
def table_energy_evol(self):
    # Pendulum constants
    m = self.m; d = self.d; l = self.l
    # Positions
    X = self.X; Y = self.Y; Z = - np.sqrt(l**2 - X**2 - Y**2) + (l+d)
    # Velocities
    Vx = self.Vx; Vy = self.Vy; Vz = (X*Vx+Y*Vy)/(l+d-Z)
    # Mgnetic dipoles constants and vector
    m1 = self.mu_P_magn
    Um1_x = X/l; Um1_y = Y/l; Um1_z = (Z-l-d)/l           
    NINTERVALS = len(self.B)   # Number of steps
    # Magnetic Fields
    Bx = np.array([self.B[i][0] for i in range(NINTERVALS)])
    By = np.array([self.B[i][1] for i in range(NINTERVALS)])
    Bz = np.array([self.B[i][2] for i in range(NINTERVALS)])
    # Potentials
    Ug = m*9.8*(Z-d)
    Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz)
    Um = Um - min(Um)           # Fit the surface of zero-potential
    U = Ug+Um
    # Kinetic Energy
    K = 0.5*m*(Vx**2+Vy**2+Vz**2)
    # Plot Energies
    plt.figure(figsize=(15,4))
    t = np.linspace(0, NINTERVALS*self.h, NINTERVALS)
    plt.title('Energies of the system', fontsize = 15)
    
    plt.subplot(131)
    plt.plot(t,U,'-',color='orange',label = 'Potential Energy',lw=1.2)
    plt.ylabel('Potential Energy [J]', fontsize = 12); plt.xlabel('Time [s]', fontsize = 12)
    plt.legend(fontsize=10); plt.xticks(size = 10);  plt.yticks(size = 10)
    
    plt.subplot(132)
    plt.plot(t,K,'-b',label = 'Kinetic Energy',lw=1.2)
    plt.ylabel('Kinetic Energy [J]', fontsize = 12); plt.xlabel('Time [s]', fontsize = 12)
    plt.legend(fontsize=10); plt.xticks(size = 10);  plt.yticks(size = 10)
    
    plt.subplot(133)
    plt.plot(t,U+K,'-r',label = 'Mechanical Energy',lw=1.0)
    plt.ylabel('Mechanical Energy [J]', fontsize = 12); plt.xlabel('Time [s]', fontsize = 12)
    plt.ylim(0,max(U)*11/10)
    plt.legend(fontsize=10); plt.xticks(size = 10);  plt.yticks(size = 10)
    
# Plot Potential and Kinetic Energy
def energy_evol_1(self, potential=True, Um_plot=False, Ug_plot=False, kinetic=True):
    # Pendulum constants
    m = self.m; d = self.d; l = self.l
    # Positions
    X = self.X; Y = self.Y; Z = - np.sqrt(l**2 - X**2 - Y**2) + (l+d)
    # Velocities
    Vx = self.Vx; Vy = self.Vy; Vz = (X*Vx+Y*Vy)/(l+d-Z)
    # Mgnetic dipoles constants and vector
    m1 = self.mu_P_magn
    Um1_x = X/l; Um1_y = Y/l; Um1_z = (Z-l-d)/l           
    NINTERVALS = len(self.B)   # Number of steps
    # Magnetic Fields
    Bx = np.array([self.B[i][0] for i in range(NINTERVALS)])
    By = np.array([self.B[i][1] for i in range(NINTERVALS)])
    Bz = np.array([self.B[i][2] for i in range(NINTERVALS)])
    # Potentials
    Ug = m*9.8*(Z-d)
    Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz)
    Um = Um - min(Um)           # Fit the surface of zero-potential
    U = Ug+Um
    # Kinetic Energy
    K = 0.5*m*(Vx**2+Vy**2+Vz**2)
    
    # Plot Energies
    plt.figure(figsize=(6,5))
    t = np.linspace(0, NINTERVALS*self.h, NINTERVALS)
    plt.title('Energies of the system', fontsize = 15)
    if potential:
        plt.plot(t,U,'-',color='orange',label = 'Potential Energy',lw=1.2)
    if Um_plot:
        plt.plot(t,Um,'--',color='blue',label = 'Magnetic',lw=1.2)
    if Ug_plot:
        plt.plot(t,Ug,'--',color='red',label = 'Gravitational',lw=1.2)
    if kinetic:
        plt.plot(t,K,'-b',label = 'Kinetic Energy',lw=1.2)
    plt.ylabel('Potential Energy [J]', fontsize = 12); plt.xlabel('Time [s]', fontsize = 12)
    plt.legend(fontsize=10); plt.xticks(size = 10);  plt.yticks(size = 10)
    


    
    