import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pytz

import CMPendulum.solver.pendulum as pend

class basins:
    def __init__(self):
        pass
    def basins(self,res, lims, params, timezone='Colombia', save_n_times=10, start_fil=0):
        """
        @params:
            res: float
            Resolution of the final image, that has a 'res' number of cols and fils
            
            lims: array_like 
            Structure: [[xi,xf], [yi,yf]], where [xi,xf] is x-limits and [yi,yf] 
            is the y-limits. Contains the limits for the grid of initial conditions. 
            
            params: dictionary
            Structure: {'Mx':Mx, 'My':My, 'm':m, 'l':l, 'd':d, 'mu_P':mu_P, 'mu':mu}
            where:
                - Mx: array_like. Has the x-positions of magnets.
                - My: array_like. Has the y-positions of magnets.
                - m: float. It's the mass value of pendulum.
                - l: float. It's the pendulum length
                - d: float. Vertical distance between rod (at reats) and magnets
                - mu_P: positive float. Magnitude of the magnetic dipole of the pendulum magnet.
                - mu: array_like. Has the magnetic dipoles values for the table magnets.
                    Positive values for atraction, negative for repulsion.
            *All the values are in SI units (meters, kilograms, seconds)
            
            timezone: string
            Values: 'Colombia', 'Denmark'
            It's your country name. It is for printing the time with the correct hour.

            save_n_times: integer
            Is the number of times that you want to save the progress of basins.

            start_fil: integer
            Is the index of the row in which yo want to star the cycle of the final positions.
            This is useful if you have a save point before, and you want to continue in the part
            that you left.

        @returns:
            
        @examples:
            Let's do a basic example with 3 magnets:
                >>> import CMPendulum.analysis.basins as basin
                >>> bs = basin.basins()
                >>> bs.basins(res=20, lims=[[-0.2,0.2],[-0.2,0.2]], params={'Mx':[-0.09, 0, 0.09],
                            'My':[-0.05,1,0.05], 'm':0.036, 'l':0.54, 'd':0.03, 'mu_P':1.84, 
                            'mu':[-0.65,-0.65,-0.65]}, timezone='Colombia')
                >>> bs.plot_im()
                plot of the image is shown
                
        """
        ## Save in 'self' the entry parameters
        self.N=res
        self.lims=lims
        self.params=params
        self.timezone=timezone
        
        ## Set initial parameters and conditions
        N = res
        tx=np.linspace(lims[0][0],lims[0][1],N)
        ty=np.linspace(lims[1][0],lims[1][1],N)
        mx, my = np.meshgrid(tx,ty)
        Mat = np.zeros((N,N,2)) #Matrix that will contains the final positions per each run
        
        ## Set parameters
        Mx = params['Mx'] #Positions table magnets
        My = params['My']  
        mu_P = params['mu_P'] #magnetic dipole magnitude (pendulum)
        mu = params['mu'] #magnetic dipoles in atraction form (table)
        m = params['m'] #mass
        l = params['l'] #length of pendulum rod
        d = params['d'] #vertical distance between rod (at reast) and magnets
        zone={'Colombia':'ETC/GMT-5','Denmark':'ETC/GMT+2'} #for datetime
        
        ##Find the data of final positions
        #Crate pendulum
        p=pend.pendulum()
        aux=-1
        inicio = time.time()
        # Run over each x initial condition
        for i in range(start_fil,len(mx)):
            
            # Print the advances of calculations and save the progress
            val = i/(N-1)*save_n_times
            if int(val) != aux:
                tz = pytz.timezone(zone[timezone])
                print('[','#'*int(val)+'-'*int(save_n_times-int(val)),']',
                      f'{int(val)}/{save_n_times}',datetime.now(tz).strftime("%H:%M:%S"))
                aux=int(val)
                np.savetxt('Mat_advances_x.csv', Mat[:,:,0], delimiter=',')
                np.savetxt('Mat_advances_y.csv', Mat[:,:,1], delimiter=',')
            
            # Run over each y initial condition
            for j in range(0,len(my)):
                
                # Define intial values
                CI = [mx[i][j],my[i][j],0,0]  #[x,y,vx,vy] initial values.
                
                # Set initial conditions
                p.set_positions(CI,Mx,My)
                p.set_magnetic_dipoles(mu_P,mu)
                
                # Find path
                Xf, Yf, Vxf, Vyf = p.find_path(Return='final pos')
                
                # Get Final positions
                Mat[i][j][0] = Xf
                Mat[i][j][1] = Yf
        
        # Save final data
        np.savetxt('Mat_x.csv', Mat[:,:,0], delimiter=',')
        np.savetxt('Mat_y.csv', Mat[:,:,1], delimiter=',')
        
        # Show total time
        fin = time.time()
        print('Total time %.2f s'%(fin-inicio))
        
        self.Mat = Mat
    
    def plot_im(self, Ax=0, Ay=0, colors=0):
        """
        @params:
            Ax: array-like
            Bla bla

            Ay: array-like
        @returns:

        @examples:

        """
        # Set predefined values
        if Ax==0 or Ay==0 or colors==0:
            Ax = self.params['Mx']
            Ay = self.params['My'] 
            n = len(Ax)          #número de atractores
            p1 = 0.7; p2 = 0.3   #p1 multiplica el color gris, p2 el color azul. Modificar para cambiar la tonalidad de los pixeles a su gusto.
            colors = [[p1/(i+1), p1/(i+1),p1/(i+1)+p2] for i in range(n)]  #colors = [[R,G,B],..] cada lista es un cógio de colores RGB.

        # Create matrix that will contains the colours data
        N = self.N          #tamaño Mat
        im = np.zeros((N,N,3))
        
        # Colour each pixel (i,j)
        for i in range(N):
            for j in range(N):
                
                # Find the distances to each magnet (or atractor)
                d=[]
                for k in range(len(Ax)):
                    d.append(np.sqrt((self.Mat[i,j,0] - Ax[k])**2+(self.Mat[i,j,1]-Ay[k])**2) )
                
                # Set the color based on the minimum distance respecting to each magnet
                index = d.index(min(d))
                im[N-1-i,j] = colors[index]
        
        # Show figure
        plt.figure(figsize=(6,6))
        plt.imshow(im)
