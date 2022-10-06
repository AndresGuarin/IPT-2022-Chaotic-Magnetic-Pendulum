"""
Created on Thu Oct  6 16:04:02 2022

@author: AndresGuarin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  #Color maps

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
def main_energy_evol(selfp):
    # Pendulum constants
    m = selfp.m; d = selfp.d; l = selfp.l
    # Positions
    X = selfp.X; Y = selfp.Y; Z = - np.sqrt(l**2 - X**2 - Y**2) + (l+d)
    # Velocities
    Vx = selfp.Vx; Vy = selfp.Vy; Vz = (X*Vx+Y*Vy)/(l+d-Z)
    # Mgnetic dipoles constants and vector
    m1 = selfp.mu_P_magn
    Um1_x = X/l; Um1_y = Y/l; Um1_z = (Z-l-d)/l           
    NINTERVALS = len(selfp.B)   # Number of steps
    # Magnetic Fields
    Bx = np.array([selfp.B[i][0] for i in range(NINTERVALS)])
    By = np.array([selfp.B[i][1] for i in range(NINTERVALS)])
    Bz = np.array([selfp.B[i][2] for i in range(NINTERVALS)])
    # Potentials
    Ug = m*9.8*(Z-d)
    Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz)
    Um = Um - min(Um)           # Fit the surface of zero-potential
    U = Ug+Um
    # Kinetic Energy
    K = 0.5*m*(Vx**2+Vy**2+Vz**2)
    # Plot Energies
    plt.figure(figsize=(15,4))
    t = np.linspace(0, NINTERVALS*selfp.h, NINTERVALS)
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
def energy_evol(selfp, potential=True, Um_plot=False, Ug_plot=False, kinetic=True):
    # Pendulum constants
    m = selfp.m; d = selfp.d; l = selfp.l
    # Positions
    X = selfp.X; Y = selfp.Y; Z = - np.sqrt(l**2 - X**2 - Y**2) + (l+d)
    # Velocities
    Vx = selfp.Vx; Vy = selfp.Vy; Vz = (X*Vx+Y*Vy)/(l+d-Z)
    # Mgnetic dipoles constants and vector
    m1 = selfp.mu_P_magn
    Um1_x = X/l; Um1_y = Y/l; Um1_z = (Z-l-d)/l           
    NINTERVALS = len(selfp.B)   # Number of steps
    # Magnetic Fields
    Bx = np.array([selfp.B[i][0] for i in range(NINTERVALS)])
    By = np.array([selfp.B[i][1] for i in range(NINTERVALS)])
    Bz = np.array([selfp.B[i][2] for i in range(NINTERVALS)])
    # Potentials
    Ug = m*9.8*(Z-d)
    Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz)
    Um = Um - min(Um)           # Fit the surface of zero-potential
    U = Ug+Um
    # Kinetic Energy
    K = 0.5*m*(Vx**2+Vy**2+Vz**2)
    
    # Plot Energies
    plt.figure(figsize=(6,5))
    t = np.linspace(0, NINTERVALS*selfp.h, NINTERVALS)
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
    
    
    
    