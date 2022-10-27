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
from CMPendulum.analysis import plot_solutions as plot_

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

    def zero_magnetic_potential(self):
        d = self.d; l = self.l
        # Positions
        X = self.X; Y = self.Y; Z = - np.sqrt(l**2 - X**2 - Y**2) + (l+d)
        # Mgnetic dipoles constants and vector
        m1 = self.mu_P_magn
        Um1_x = X/l; Um1_y = Y/l; Um1_z = (Z-l-d)/l           
        NINTERVALS = len(self.B)   # Number of steps
        # Magnetic Fields
        Bx = np.array([self.B[i][0] for i in range(NINTERVALS)])
        By = np.array([self.B[i][1] for i in range(NINTERVALS)])
        Bz = np.array([self.B[i][2] for i in range(NINTERVALS)])
        # Potentials
        Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz)
        Um_min = min(Um)
        return Um_min
    
    def plot_path(self):
        plot_.path(self)

    # In[4]-------------------------------------4. Get functions----------------------------------------# 
    
    def get_self(self):
        return self
    
    def get_positions(self):
        return [self.X, self.Y, self.Vx, self.Vy]
    
    def get_end_positions(self):
        return [self.X[-1], self.Y[-1], self.Vx[-1], self.Vy[-1]]
    
    def get_kinetic_energy(self):
        m = self.m; d = self.d; l = self.l
        # Positions
        X = self.X; Y = self.Y; Z = - np.sqrt(l**2 - X**2 - Y**2) + (l+d)
        # Velocities
        Vx = self.Vx; Vy = self.Vy; Vz = (X*Vx+Y*Vy)/(l+d-Z)
        K = 0.5*m*(Vx**2+Vy**2+Vz**2)
        return K
    
    def get_initial_energy(self):
        m = self.m; d = self.d; l = self.l
        # Positions
        X0 = self.X[0]; Y0 = self.Y[0]; Z0 = - np.sqrt(l**2 - X0**2 - Y0**2) + (l+d)
        # Velocities
        Vx0 = self.Vx[0]; Vy0 = self.Vy[0]; Vz0 = (X0*Vx0+Y0*Vy0)/(l+d-Z0)
        m1 = self.mu_P_magn
        Um1_x = X0/l; Um1_y = Y0/l; Um1_z = (Z0-l-d)/l           
        # Magnetic Fields
        Bx = self.B[0][0] 
        By = self.B[0][1]
        Bz = self.B[0][2]
        # Potentials
        Ug = m*9.8*(Z0-d)
        Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz)
        Um_min = self.zero_magnetic_potential()
        Um = Um - Um_min           # Fit the surface of zero-potential
        # Kinetic Energy
        K = 0.5*m*(Vx0**2+Vy0**2+Vz0**2)
        E = Ug + Um + K
        return E
    
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