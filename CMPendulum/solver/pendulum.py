"""
Created on Fri Jun  3 22:01:50 2022

This module finds the path for a given initial conditions and physics parameters as
pendulum length, mass of the pendulum, arrangement of magnets, etc. There is also
plot functions to visualize the results and understand the solution. 

We implemented the RK4 method to solve the equations of motion. We used the newtonian 
formalism and consider the magnetic force as the expandend force between two perfect 
magnetic dipoles. The model implement the friction force and neglict the mass of the 
pendulum rod. The rod is consider as a unextensible, rigid and massless one.

You can use this code writing: 
    >>> from CMPendulum import pendulum as pend
    >>> p = pend.pendulum()
    >>> X, Y, Vx, Vy = p.find_path()   #Find trajectories
    >>> #Plot solution:
    >>> p.plot_alltable()              
    >>> p.plot_vector_space(res=30)
    >>> p.plot_path()
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, power

class pendulum:
    # In[0] ------------------------------------ 0. Initialize object -----------------------------------#
    def __init__(self, show=False):
        # Set predifined values
        self.set_pendulum()          # Set physical parameters
        self.set_physics()           # Set physics model and law of motion
        self.set_code_parameters()   # Set code parameters as number of points
        self.set_random()            # Set initial positions randomly
        if show:
            self.summary()
            
            
    # In[1] --------------------------------------1. Create object---------------------------------------#
    
    # Set Physical parameters
    def set_pendulum(self, l=0.54, d=0.03, R=0.01, m=0.04, mu_P_magn=2.0, mu_P_dir=np.array([0,0,-1]),
                     Table_SIZE = 0.3):
        # System dynamics parameters
        self.l = l        #Pendulum length
        self.d = d        #Vertical distance between pendulum and table
        self.R = R        #Friction constant   
        self.m = m        #Mass of the pendulum magnet
        self.mu_P_magn = mu_P_magn   #Magnitude of the magnetic dipole of the pendulum magnet
        self.mu_P_dir = mu_P_dir     #Direction of the magnetic dipole of pendulum magnet if it was in (x,y)=(0,0)
        self.Table_SIZE = Table_SIZE
    
        # Plot parameters
        self.lim    = self.Table_SIZE/2 + 0.04     #Limit of the plots
        self.lim_vs = (self.Table_SIZE + 0.05)/2   #Limit plot of vector space    
    
        # Other parameters
        self.lim = self.Table_SIZE/2 + 0.04         #Limit of the plots
        self.lim_vs = (self.Table_SIZE + 0.05)/2   #Limit plot of vector space
    
    # Set integration model and Law of motion (magnetic force between magnets)
    def set_physics(self,model='RK4',law='F1'):
        self.model = model  #  RK4, RKF
        self.law = law
        
    # Set Code parameters
    def set_code_parameters(self,h=0.01, N=2000):
        self.h = h        #Time lapse between each step
        self.N = N        #Total number of steps
        self.sec = 1/self.h  #Number of steps that waste one second
    
    
    # In[2] ---------------------------------2. Set initial conditions ----------------------------------#
    
    def set_random(self,Nmin=1,Nmax=8, mu=[1.07, 2.05, 1.95, 1.84, 1.64, 1.74, 1.43], 
                   u = [ np.array([0.66, -0.37, 0.65]), np.array([0.31, -0.35, 0.88]),
                         np.array([-0.6, -0.67, 0.44]), np.array([0,0,1]),
                         np.array([0,0,-1]) ]):
        """
        @params:
            Nmin: int, mínimo número de imanes.
            Nmax: int, máximo número de imanes.
            mu:   array-like, valores posibles de dipolo magnético.
            u: array-like, valores posibles de dirección de los imanes.
        
        @returns:
            none
        """
        # Table parameters
        self.NMAGNETS = np.random.randint(low=Nmin,high=Nmax)
                
        # Set initial positions
        lm = self.Table_SIZE
        self.x = np.random.rand()*lm -lm/2
        self.y = np.random.rand()*lm -lm/2
        self.coordinates = 'cartesians'
        
        # Set initial velocities
        self.vx = 0
        self.vy = 0
        
        # Set magnets positions
        self.Mx = []
        self.My = []
        self.Mz = []
        for i in range(self.NMAGNETS):
            self.Mx.append(np.random.rand()*self.Table_SIZE - self.Table_SIZE/2)
            self.My.append(np.random.rand()*self.Table_SIZE - self.Table_SIZE/2)
        self.Mx = np.array(self.Mx)
        self.My = np.array(self.My)
        self.Mz = np.zeros(self.NMAGNETS)
        M = np.array([self.Mx, self.My, self.Mz],dtype=float)
        self.M = M.T
        
        
        # Set magnet dipoles
        self.mu_magn = []   # Magnitud of the magnetic dipoles
        self.mu_dir = []    # Direction of the magnetic dipoles
        mu = np.abs(np.array(mu))
        mu_NUMS=len(mu)
        u_NUMS = len(u)
        for i in range(self.NMAGNETS): #set the magnetic dipoles to the magnets randomly
            n1 = np.random.randint(mu_NUMS)
            n2 = np.random.randint(u_NUMS)
            
            mu1 = mu[n1]
            u1 = u[n2]
            
            self.mu_magn.append(mu1)
            self.mu_dir.append(u1)
        
        # Finding force constant
        if self.law=='F1':
            self.S = 3*10**-7*self.mu_P_magn * np.array(self.mu_magn) 
            self.Cm = 10**-7*np.array(self.mu_magn)                    # Constants for magnetic field
    
    
    # Set Magnets parameters
    def set_magnets(self, Mx, My, Mz, mu, u):
        self.NMAGNETS = len(Mx)
        self.Mx = np.array(Mx)
        self.My = np.array(My)
        self.Mz = np.array(Mz)
        self.mu_magn = mu
        self.mu_dir = u
        M = np.array([self.Mx, self.My, self.Mz],dtype=float)
        self.M = M.T
        # Finding force constant
        if self.law=='F1':
            self.S = 3*10**-7*self.mu_P_magn * np.array(self.mu_magn)
            self.Cm = 10**-7*np.array(self.mu_magn)                    # Constants for magnetic field
        
    
    # Set initial conditions
    def set_initial_conditions(self, CI, coordinates):
        if coordinates == 'cartesians':       
            self.x = CI[0]
            self.y = CI[1]
            self.vx = CI[2]
            self.vy = CI[3]
            
        if coordinates == 'spherical':
            theta = CI[0]
            phi = CI[1]
            Dtheta = CI[2]
            Dphi = CI[3]
            self.x = self.l * np.cos(theta)*np.sin(phi)
            self.y = self.l * np.sin(theta)*np.sin(phi)
            self.z = self.l * np.cos(phi)
            self.vx = - Dtheta*self.y - (self.z*self.x*Dphi)/np.sqrt(self.x**2+self.y**2)
            self.vy = Dtheta*self.x - (self.z*self.y*Dphi)/np.sqrt(self.x**2+self.y**2)
            
            self.theta = theta
            self.phi = phi
            self.Dtheta = Dtheta
            self.Dphi = Dphi    
        self.coordinates = coordinates
        
    # In[3]---------------------------3. Functions for specific purposes -------------------------------#
    
    def summary(self):
        print('====================== Summary report ============================')
        print('Physical parameters: ')
        print('   l = ',self.l)
        print('   d = ',self.d)
        print('   R = ',self.R)
        print('   m = ',self.m)
        print('   mu_P_magn = ',self.mu_P_magn)
        print('   mu_P_dir  = ',self.mu_P_dir)
        print('   model = ',self.model)
        print('   law   = ',self.law)
        print('==================================================================')
        print('Code parameters:')
        print('   h = ',self.h)
        print('   N = ',self.N)
        print('==================================================================')
        print('Initial values')
        
        if self.coordinates == 'cartesians':
            print('   x  = ',self.x)
            print('   y  = ',self.y)
            print('   vx = ',self.vy)
            print('   vy = ',self.vx)
        
        if self.coordinates == 'spherical':
            print('   theta     = ',self.theta)
            print('   phi       = ',self.phi)
            print('   dot theta = ',self.Dtheta)
            print('   dot phi   = ',self.Dphi)
        
        print('   NMAGNETS = ',self.NMAGNETS)
        print('   mu_magn  = ',self.mu_magn)
        print('   mu_dir   = ',self.mu_dir)
        print('==================================================================')
        print('███╗   ███╗ █████╗  ██████╗ ███╗   ██╗███████╗████████╗██╗ ██████╗')
        print('████╗ ████║██╔══██╗██╔════╝ ████╗  ██║██╔════╝╚══██╔══╝██║██╔════╝')
        print('██╔████╔██║███████║██║  ███╗██╔██╗ ██║█████╗     ██║   ██║██║     ')
        print('██║╚██╔╝██║██╔══██║██║   ██║██║╚██╗██║██╔══╝     ██║   ██║██║     ')
        print('██║ ╚═╝ ██║██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   ██║╚██████╗')
        print('╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝')
        print()
        print('██████╗ ███████╗███╗   ██╗██████╗ ██╗   ██╗██╗     ██╗   ██╗███╗   ███╗')
        print('██╔══██╗██╔════╝████╗  ██║██╔══██╗██║   ██║██║     ██║   ██║████╗ ████║')
        print('██████╔╝█████╗  ██╔██╗ ██║██║  ██║██║   ██║██║     ██║   ██║██╔████╔██║')
        print('██╔═══╝ ██╔══╝  ██║╚██╗██║██║  ██║██║   ██║██║     ██║   ██║██║╚██╔╝██║')
        print('██║     ███████╗██║ ╚████║██████╔╝╚██████╔╝███████╗╚██████╔╝██║ ╚═╝ ██║')
        print('╚═╝     ╚══════╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝')
    
    def set_lim(self, lim):
        self.lim = lim
    
    def set_positions(self,CI, Mx, My):
        self.x = CI[0]
        self.y = CI[1]
        self.vx = CI[2]
        self.vy = CI[3]
        self.Mx = Mx
        self.My = My
        self.Mz = np.aray([0.,0.,0.])
        self.NMAGNETS = len(Mx)
        M = np.array([self.Mx, self.My, self.Mz],dtype=float)
        self.M = M.T
        
    def set_physical_parameters(self, m, l, d):
        self.m = m     #Mass of the pendulum
        self.l = l     #lenght of the pendulum
        self.d = d     #Distance between table and pendulum equilibrium position
    
    def set_magnetic_dipoles(self, mu_P, mu, u, u_P=np.array([0,0,-1])):
        self.mu_P_magn = abs(mu_P)           # Magnitude of the magnetic dipole
        self.mu_P_dir = u_P                  # Direction of the magnetic dipole of pendulum magnet if it was in (x,y)=(0,0)
        self.mu_magn = np.abs(np.array(mu))
        
        self.mu_dir = u                      # Direction magnetic dipoles of table magnets

        # Finding force constant
        if self.model=='F1':
            self.S = 3*10**-7*self.mu_P_magn * self.mu_magn
            self.Cm = 10**-7*np.array(self.mu_magn)                    # Constants for magnetic field

    # In[4]-------------------------------------4. Get functions----------------------------------------# 
    
    def get_self(self):
        return self
    
    def get_positions(self):
        return [self.X, self.Y, self.Vx, self.Vy]
    
    def get_end_positions(self):
        return [self.X[-1], self.Y[-1], self.Vx[-1], self.Vy[-1]]

    
    # In[5]----------------------------------5. Solver equation functions-------------------------------#
    
    # Find the next steps for each interval
    def find_path(self, Return='pos', show=False):
        
        # Plot summary
        if show:
            self.summary()
            print('Enter to find_path')
        
        self.X = [self.x]    # X positions
        self.Y = [self.y]    # Y positions
        self.Vx = [self.vx]  # X velocities
        self.Vy = [self.vy]  # Y velocities
        self.FB = []         # Arrays of the net magnetic forces
        self.B = []          # Arrays of the net magnetic fields
        
        s=self.sec     # Number of steps that waste one second.
        A = int(1.2*s) # Check point 1
        B = int(0.6*s) # Check point 2
        C = int(0.2*s) # Check point 3
        Rc = (0.01)**2 # Squared Critical distance to stop the calculations in meters^2
        
        #Finfing path
        for t in range(self.N):
            next_val = self.next_value(self.X[t],self.Y[t],self.Vx[t],self.Vy[t])
            
            self.FB.append(self.FF2)    # Magnitud magnetic Force B
            self.B.append(self.BB2)     # Magnitud magnetic Field B
            self.X.append(next_val[0])  # X position
            self.Y.append(next_val[1])  # Y position
            self.Vx.append(next_val[2]) # X velocity
            self.Vy.append(next_val[3]) # Y velocity
            
            if t>5*s:  #This conditional returns true when the time is higher than 5 seconds
                if ((self.X[t-A]-self.X[t])**2 + (self.Y[t-A]-self.Y[t])**2 < Rc):
                    if ((self.X[t-B]-self.X[t])**2 + (self.Y[t-B]-self.Y[t])**2 < Rc):
                        if ((self.X[t-C]-self.X[t])**2 + (self.Y[t-C]-self.Y[t])**2 < Rc):
                            break
            
        # Find the last net magnetic force
        aux = self.next_value(self.X[-1],self.Y[-1],self.Vx[-1],self.Vy[-1])
        self.FB.append(self.FF1)
        self.B.append(self.BB1)
        
        # Converting list to arrays
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Vx = np.array(self.Vx)
        self.Vy = np.array(self.Vy)
        
        if show:
            print('find_path has ended')
        
        if Return == 'final pos':
            return self.X[-1], self.Y[-1], self.Vx[-1], self.Vy[-1]
        
        elif Return == 'pos':
            return self.X, self.Y, self.Vx, self.Vy
        
        elif Return == 'none':
            pass
        
    # Returns the actual force given the initial conditions of the interval.
    def Fk(self,x,y,vx,vy):
        """
        Calculates the velocity and acelerations based on the initial conditions of the magnets 
        @params
        mx, my:
            arrays of doubles that contains the position of the magnets
        x,y,vx,vy:
            doubles of the position and velocity of the pendulum magnet

        @returns
        vx, vy ,ax, ay:
            doubles that contains the velocity and aceleration based on the given inputs
        """ 
        l = self.l
        d = self.d
        m = self.m
        R = self.R
        M = self.M
        S = self.S
        Cm = self.Cm
        u_m2 = self.mu_dir
        
        r_P = np.array([x,y,-np.sqrt(l**2-x**2-y**2)+ l+d]) # Position of pendulum magnet
        r_Q = np.array([0,0,l+d])                           # Position of the attached point of the rod
        rqp = r_P - r_Q                                     # Vector from Q to P (free point of rod)
        sign = -self.mu_P_dir[2]                            # Sign that indicates the orientation of the pendulum magnet
        m1 = sign * rqp/l                                   # Magnetic dipole vector
                
        # Finding net magnetic force
        Fb = np.array([0.,0.,0.])   # magnetic force
        B  = np.array([0.,0.,0.])    # magnetic field
        
        for i in range(self.NMAGNETS):
            r_mi = M[i]                       # Position of pendulum m_i
            r_vect = r_P - r_mi                 # Relative position between magnets
            r = np.linalg.norm(r_vect)          # Distance between magnets
            ur = r_vect / r                     # Unit vector of relative position (from m2 to m1)
            m2 = u_m2[i]                        # Magnetic dipole vector of the 2nd magnet
            
            Fb += S[i] * (dot(m1,m2)*ur + dot(m2,ur)*m1 +                             # Sum of the magnetic forces
                          dot(m1,ur)*m2 - 5*dot(m1,ur)*dot(m2,ur)*ur) / power(r,4)            
            B  += Cm[i] * (3*dot(m2,ur)*ur - m2) / power(r,3)                         # Sum of mgnetic field
        self.FF1 = Fb  # For save the data of the net magnetic force.
        self.BB1 = B
        
        # Finding tension
        uT = -rqp/l                            # Unit vector from P to Q
        z = r_P[2]                             # Z position of pendulum magnet
        T = (m*9.8*(l+d-z)/l - dot(Fb,uT))*uT  # Tension
        
        # Computes acelerations
        ax = (Fb[0] - R*vx + T[0])/m  # (Magnetic Force) - (Friction Force) + (Tension), all over mass pendulum
        ay = (Fb[1] - R*vy + T[1])/m
               
        return np.array([vx, vy, ax, ay])

    # Implemented RK4 equations to find the final positions of a given initial interval. Uses the force of the function Fk    
    def next_value(self, x,y,vx,vy):
        """
        Calculates the next positions and velocities after a time lapse of self.h
        Find the 4 k-values of the Runge-Kutta method (RK4). The equations used are:
        
        r´ = v
        v´ = Fk(r, v)
        r(t0) = r0, v(t0) = v0
        
        where ´ denotes time derivate, r=[x,y], v=[vx,vy]
        
        @params
        x,y,vx,vy:
            doubles of the position and velocity of the pendulum magnet

        @returns
        xf,yf,vxf,vyf
            doubles that contains the final positions and velocities of the pendulum magnet
        """
        
        #K1 and calculations
        k1 = self.h*self.Fk(x,y,vx,vy)
        x1 = x + k1[0]/2
        y1 = y + k1[1]/2
        vx1 = vx + k1[2]/2
        vy1 = vy + k1[3]/2
        
        #Save magnetic Force and magnetic field
        self.FF2 = self.FF1
        self.BB2 = self.BB1

        #K2 and calculations
        k2 = self.h*self.Fk(x1,y1,vx1,vy1)
        x2 = x + k2[0]/2
        y2 = y + k2[1]/2
        vx2 = vx + k2[2]/2
        vy2 = vy + k2[3]/2

        #K3 and calculations
        k3 = self.h*self.Fk(x2,y2,vx2,vy2)
        x3 = x + k3[0]
        y3 = y + k3[1]
        vx3 = vx + k3[2]
        vy3 = vy + k3[3]

        #K4 and calculating final positions
        k4 = self.h*self.Fk(x3,y3,vx3,vy3)
        xf = x + 1/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        yf = y + 1/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        vxf = vx + 1/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        vyf = vy + 1/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        return np.array([xf, yf, vxf, vyf])
    
    
    # In[6]-------------------------------------6. Plot functions---------------------------------------#
    
    # Plot trajectory
    def plot_path(self):    
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
    
    # Plot border of table
    def plot_table(self):
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
    def plot_alltable(self):
        Mx = self.Mx
        My = self.My
        mu = self.mu_magn
        self.plot_table()
        
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
    def plot_vector_space(self,res=20,a=0, net_forces=True, magnetic=False, tension=False):
        
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
        