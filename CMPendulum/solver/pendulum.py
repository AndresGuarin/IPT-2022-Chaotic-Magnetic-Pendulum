"""
Created on Fri Jun  3 22:01:50 2022

This code calculates the positions and velocities of the magnetic pendulum using 
the positions of the arragment of magnets on the table and the initial conditions
 of this magnetic pendulum. The solution was obtained using the RK4 method
"""

from matplotlib import cm  #Color maps
import numpy as np
import matplotlib.pyplot as plt

class pendulum:
    def __init__(self):
        # System dynamics parameters
        self.R = 0.01          #friction constant   
        self.l = 0.54           #Pendulum length
        self.C = 9.8/self.l    #self-restoring force constant C=g/l
        self.d = 0.03       #vertical distance between pendulum and magnets
        self.h = 0.01       #time lapse between each step
        self.N = 2000       #total number of steps
        self.m = 0.036226       #Mass of the pendulum magnet
        self.mu_P_magn = 2       #Magnitude of the magnetic dipole of the pendulum magnet
        self.mu_P_dir = np.array([0,0,-1])  #Direction of the magnetic dipole of pendulum magnet if it was in (x,y)=(0,0)
        
        # Other parameters
        self.sec = 1/self.h #Number of steps that waste one second.
        self.Table_SIZE = 0.3
        self.NMAGNETS = np.random.randint(low=1,high=8)
        self.lim = self.Table_SIZE/2 + 0.04         #Limit of the plots
        self.lim_vs = (self.Table_SIZE + 0.05)/2   #Limit plot of vector space
        
        # Initial velocities
        self.vx = 0
        self.vy = 0
        
        # Set initial position randomly    
        lm = self.Table_SIZE
        self.x = np.random.rand()*lm -lm/2
        self.y = np.random.rand()*lm -lm/2
              
        # Create table
        self.Mx = []
        self.My = []
        for i in range(self.NMAGNETS):
            self.Mx.append(np.random.rand()*self.Table_SIZE - self.Table_SIZE/2)
            self.My.append(np.random.rand()*self.Table_SIZE - self.Table_SIZE/2)
        
        # Create dipoles
        self.mu = [-1.071875, -2.048, -1.9456, -1.8432, -1.6384, -1.7408, -1.4336]
        self.mu_magn = []   # Magnitud of the magnetic dipoles
        self.mu_dir = []    # Direction of the magnetic dipoles
        self.S = [] 
        self.mu_NUMS=len(self.mu)
        for i in range(self.NMAGNETS): #set the magnetic dipoles to the magnets randomly
            
            num = np.random.randint(self.mu_NUMS)
            mu = self.mu[num]
            
            self.S.append(np.abs(mu))
            self.mu_magn.append(np.abs(mu))
            self.mu_dir.append(np.array([0,0,-1])*np.sign(mu)) #The directions are in the z-axis
        
        self.mu_dir = np.array(self.mu_dir)
        self.sign = np.array([self.mu_dir[i][2]*-1 for i in range(self.NMAGNETS)])
        
        #Finding S values
        self.S = np.array(self.S)
        self.S = 3*(4*np.pi*10**-7 * self.mu_P_magn) / (4*np.pi * self.m) * self.S
    
    def set_lim(self, lim):
        self.lim = lim
    
    def set_positions(self,CI, Mx, My):
        self.x = CI[0]
        self.y = CI[1]
        self.vx = CI[2]
        self.vy = CI[3]
        
        self.Mx = Mx
        self.My = My
        self.NMAGNETS = len(Mx)
        
    def set_code_parameters(self, m, l, d):
        self.m = m     #Mass of the pendulum
        self.l = l     #lenght of the pendulum
        self.d = d     #Distance between table and pendulum equilibrium position
    
    def set_magnetic_dipoles(self, mu_P, mu):
        self.mu_P_magn = np.abs(mu_P)  #Magnitude of the magnetic dipole
        self.mu_P_dir = np.array([0,0,-1])*np.sign(mu_P) #Direction of the magnetic dipole of pendulum magnet if it was in (x,y)=(0,0)
        self.mu_magn = np.abs(np.array(mu))
        
        self.mu_dir = [np.array([0,0,-1])*np.sign(mu[i]) for i in range(self.NMAGNETS)] #Direction magnetic dipoles of table magnets
        self.mu_dir = np.array(self.mu_dir)
        self.mu_NUMS=len(self.mu_magn)
        
        #Finding S values
        self.S = np.copy(self.mu_magn)
        self.S = 3*(4*np.pi*10**-7 * self.mu_P_magn) / (4*np.pi) * self.S
    
        self.sign = [np.sign(mu[i]) for i in range(self.NMAGNETS)] #If sign is positive there is atraction; if negative, repulsion
    
    def get_self(self):
        return self
    
    def find_path(self, Return='pos'):
        # Finding the next steps for each interval
        self.X = [self.x]    #X positions
        self.Y = [self.y]    #Y positions
        self.Vx = [self.vx]  #X velocities
        self.Vy = [self.vy]  #Y velocities
        self.FB = []         #Arrays of the net magnetic forces
        
        s=self.sec #Number of steps that waste one second.
        A = int(1.2*s) #check point 1
        B = int(0.6*s) #check point 2
        C = int(0.2*s) #check point 3
        Rc = (0.01)**2 #squared Critical distance to stop the calculations. The magnitude is in meters^2
        
        #Finfing path
        for t in range(self.N):
            next_val=self.next_value(self.X[t],self.Y[t],self.Vx[t],self.Vy[t])
            
            self.FB.append(self.FF2)
            self.X.append(next_val[0])
            self.Y.append(next_val[1])
            self.Vx.append(next_val[2])
            self.Vy.append(next_val[3])
            
            if t>5*s:  #This conditional returns true when the time is higher than 5 seconds
                if ((self.X[t-A]-self.X[t])**2 + (self.Y[t-A]-self.Y[t])**2 < Rc):
                    if ((self.X[t-B]-self.X[t])**2 + (self.Y[t-B]-self.Y[t])**2 < Rc):
                        if ((self.X[t-C]-self.X[t])**2 + (self.Y[t-C]-self.Y[t])**2 < Rc):
                            break
            
        # Find the last net magnetic force
        aux = self.next_value(self.X[-1],self.Y[-1],self.Vx[-1],self.Vy[-1])
        self.FB.append(self.FF1)
        
        # Converting list to arrays
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Vx = np.array(self.Vy)
        self.Vy = np.array(self.Vx)
        
        if Return == 'final pos':
            return self.X[-1], self.Y[-1], self.Vx[-1], self.Vy[-1]
        elif Return == 'pos':
            return self.X, self.Y, self.Vx, self.Vy

    def plot_path(self):
        # Plot trajectory
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
        
    def plot_table(self):
        
        # Plot magnets
        plt.figure(figsize=(6,6))
        for i in range(self.NMAGNETS):
            plt.plot(self.Mx[i],self.My[i],'or')
        plt.plot(self.Mx[0],self.My[0],'or',label='Magnets')
        
        # Plot Table
        self.ap = self.Table_SIZE/2 +0.01
        plt.plot([-self.ap,-self.ap,self.ap,self.ap,-self.ap],[-self.ap,self.ap,self.ap,-self.ap,-self.ap],'-b',label='Table')
        
        # Put legends, title, labales, and limits
        plt.legend(loc='upper right')
        plt.title('Positions of magnets')
        plt.xlabel('x-axis [m]')
        plt.ylabel('y-axis [m]')
        plt.xlim(-self.lim,self.lim)
        plt.ylim(-self.lim,self.lim)

    def plot_alltable(self):
        Mx = self.Mx
        My = self.My
        mu = self.mu_magn * self.sign
        self.plot_table()
        
        # Plot magnets
        for i in range(len(Mx)):
            plt.plot(Mx[i],My[i],'or')
            plt.text(Mx[i]+0.005,My[i]+0.005,f'{i+1}')
            plt.text(Mx[i]-0.02,My[i]-0.02,'%.2f [$Am^2$]'%(mu[i]),fontsize=8)
        
        # Put limits, labels, legend and title
        lim = self.Table_SIZE/2 + 0.04
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)
        plt.xlabel('X-axis [m]')
        plt.ylabel('Y-axis [m]')
        plt.legend(loc='upper right')
        plt.title('Table Magnets')

    def plot_vector_space(self,res=20,a=0):
        
        # Put limits of plot
        if a==0:
            a=self.lim_vs #Predefined value
        elif a > self.l:
            a = self.l/2 #Ensure that the points are ploted in the zone inside the spheric pendulum
        
        # Create data of vector tails
        t = np.linspace(-a,a,res)
        NUM = len(t)
        X, Y = np.meshgrid(t,t)
        Z = -np.sqrt(self.l**2 - X**2 - Y**2)+self.l+self.d
        
        # Finding net magnetic force
        Fx = np.zeros((NUM,NUM))
        Fy = np.zeros((NUM,NUM))
        Fz = np.zeros((NUM,NUM))
        
        # Unit vector of the magnetic dipole m1 (from Q to P)
        M1x = X/self.l                 #X component 
        M1y = Y/self.l                 #Y component 
        M1z = (Z-self.l-self.d)/self.l #Z component 
        
        # Iterating over table magnets
        for i in range(self.NMAGNETS):
            
            #Squared distance between magnet i and pendulum magnet
            R = (self.Mx[i]-X)**2 + (self.My[i]-Y)**2 + Z**2  
            
            #Table magnet constants
            m2 = self.mu_dir[i]
            
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
        
        # Magnitude of the projection of net magnetic force on plane xy
        F_xy_mag = np.sqrt(Fx**2 + Fy**2)
        
        # Finding tension Force
        
        # Unit vector of tension (from P to Q) 
        UT_x = -X/self.l                #X component 
        UT_y = -Y/self.l                #Y component 
        UT_z = (self.l+self.d-Z)/self.l #Z component 
        
        # Dot product
        FbUT = Fx*UT_x + Fy*UT_y + Fz*UT_z  #Dot product between net Fb and UT
        
        # Finding Tension
        Tx = (self.m*9.8*(self.l+self.d-Z)/self.l - FbUT)*UT_x
        Ty = (self.m*9.8*(self.l+self.d-Z)/self.l - FbUT)*UT_y
        Tz = (self.m*9.8*(self.l+self.d-Z)/self.l - FbUT)*UT_z
        
        # Magnitude of the projection of tension force on plane xy
        T_xy_mag = np.sqrt(Tx**2 + Ty**2)
        
        # Find force field (refusing friction force)
        Field_x = Fx+Tx
        Field_y = Fy+Ty
        Field_z = Fz+Tz
        
        # Magnitude of the projection of force field on plane xy
        Field_xy_mag = np.sqrt(Field_x**2 + Field_y**2)
        
        # Plotting vectors
        
        # Field Forces
        plt.quiver(X,Y,Field_x/Field_xy_mag, Field_y/Field_xy_mag, width=0.0025, angles='xy', scale_units='width', scale=36)
        
        #Magnetic net forces
        #plt.quiver(X,Y,Fx/F_xy_mag,Fy/F_xy_mag, linewidths=4, width=0.0025, angles='xy', scale_units='width', scale=39)                
        
        #Tensions
        #plt.quiver(X,Y,Tx/T_xy_mag, Ty/T_xy_mag, linewidths=4, width=0.0025, angles='xy', scale_units='width', scale=37)

    
    def plot_potential(self,res=20, a=0):
        
        # Put limits of plot
        if a==0:
            a = self.lim_vs #predifined value
        elif a > self.l:
            a = self.l/2 #Ensure that the points are ploted in the zone inside the spheric pendulum
        
        # Create data of the plot
        t = np.linspace(-a,a,res)
        NUM = len(t)
        X, Y = np.meshgrid(t,t)
        Z = -np.sqrt(self.l**2 - X**2 - Y**2)+self.l+self.d

        # Finding magnetic field B
        Bx = np.zeros((NUM,NUM))
        By = np.zeros((NUM,NUM))
        Bz = np.zeros((NUM,NUM))
        
        # Iterating over each table magnet
        for i in range(self.NMAGNETS):
            
            #Squared distance between magnet i and the pendulum magnet
            R = (self.Mx[i]-X)**2 + (self.My[i]-Y)**2 + Z**2  
            
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
        
        # Finding magnetic potential
        
        # Unit vector from Q to P (direction of m1)
        Um1_x = X/self.l                 #X component  
        Um1_y = Y/self.l                 #Y component 
        Um1_z = (Z-self.l-self.d)/self.l #Z component 
        
        # Pendulum magnet constants
        m1 = self.mu_P_magn
        
        # Magnetic potential
        Um = -m1 * (Um1_x*Bx + Um1_y*By + Um1_z*Bz)
        
        # Finding gravitational potential
        Ug = self.m*9.8*(Z-self.d)
        
        # Plot potential
        plt.figure(figsize=(5,4))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X,Y,Ug+Um, cmap=cm.jet, edgecolor='none')
            
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
        
        # z component of pendulum magnet
        z = -np.sqrt(self.l**2-x**2-y**2)+self.l+self.d       
        
        # Magnetic dipole vector m1 of pendulum magnet.     
        
        # Vector from Q to P
        rqp = np.array([x,y,z-self.l-self.d])
        
        # Sign that indicates the orientation of the pendulum magnet
        sign = -self.mu_P_dir[2]
        
        # Magnetic dipole
        m1 = sign * rqp/self.l
        
        # Finidng the sum of the magnetic forces
        Fb = np.array([0,0,0])
        for i in range(self.NMAGNETS):
            #Squared distance between pendulum and table magnet
            r = (self.Mx[i]-x)**2 + (self.My[i]-y)**2 + z**2
            
            #Magnetic dipole vector of the table magnet
            m2 = self.mu_dir[i]
            
            #Unit vector of relative position (from m2 to m1)
            ur = np.array( [x-self.Mx[i], y-self.My[i], z] )/np.sqrt(r) 
            
            #Sum of the forces
            Fb = Fb + self.S[i] * ( (m1@m2)*ur + (m2@ur)*m1 + (m1@ur)*m2 - 5*(m1@ur)*(m2@ur)*ur ) / r**2
        
        # Save the data of the net magnetic force.
        self.FF1 = Fb
        
        # Finding tension
        
        # Unit vector from P to Q
        uT = -rqp/self.l 
        
        # Tension
        T = (self.m*9.8*(self.l+self.d-z)/self.l - Fb@uT)*uT
        
        # Acelerations
        ax = (Fb[0] - self.R*vx + T[0])/self.m
        ay = (Fb[1] - self.R*vy + T[1])/self.m
               
        return np.array([vx, vy, ax, ay])

    
    def next_value(self, x,y,vx,vy):
        """
        Calculates the next positions and velocities after a time lapse of self.h
        Find the 4 k-values of the Runge-Kutta method (RK4)
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
        
        #Save magnetic Force
        self.FF2 = self.FF1

        #K2 and calculations
        k2 = self.h*self.Fk(x1,y1,vx1,vy1)
        x2 = x + k2[0]/2
        y2 = y + k2[1]/2
        vx2 = vx + k2[2]/2
        vy2 = vy + k2[3]/2

        #K3 and calculations
        k3 = self.h*self.Fk(x2,y2,vx2,vy2)
        x2 = x + k2[0]
        y2 = y + k2[1]
        vx2 = vx + k2[2]
        vy2 = vy + k2[3]

        #K4 and calculating final positions
        k4 = self.h*self.Fk(x2,y2,vx2,vy2)
        xf = x + 1/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        yf = y + 1/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        vxf = vx + 1/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        vyf = vy + 1/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        return np.array([xf, yf, vxf, vyf])