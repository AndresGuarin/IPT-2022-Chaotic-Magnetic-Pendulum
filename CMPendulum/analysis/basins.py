import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pytz

import CMPendulum.solver.pendulum as pend

def basins(res, lims, params, timezone):
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
    @returns:
        
    @examples:
        Let's do a basic example with 3 magnets:
            >>> import CMPendulum.analysis.basins as bs
            >>> bs.basins(res=20, lims=[[-0.2,0.2],[-0.2,0.2]], params={'Mx':[-0.09, 0, 0.09],
                         'My':[-0.05,1,0.05], 'm':0.036, 'l':0.54, 'd':0.03, 'mu_P':1.84, 
                         'mu':[-0.65,-0.65,-0.65]}, timezone='Colombia')
            >>> bs.plot_im()
            plot of the image is shown
            
    """
    ## Set initial parameters and conditions
    N=res
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
    zone={'Colombia':'ETC/GMT-7','Denmark':'ETC/GMT+2'} #for datetime
    
    ##Find the data of final positions
    #Crate pendulum
    #pend=
    aux=-1
    inicio = time.time()
    # Run over each x initial condition
    for i in range(0,len(mx)):
        
        # Print the advances of calculations and save the progress
        val = i/(N-1)*10
        if int(val) != aux:
            tz = pytz.timezone(zone[timezone])
            print('[','#'*int(val), ']',f'{int(val)}/15',datetime.now(tz).strftime("%H:%M:%S"))
            aux=int(val)
            np.savetxt('Mat_advances_x.csv', Mat[:,:,0], delimiter=',')
            np.savetxt('Mat_advances_y.csv', Mat[:,:,1], delimiter=',')
        
        # Run over each y initial condition
        for j in range(0,len(my)):
            
            # Create pendulum
            p = pend.pendulum()
            CI = [mx[i][j],my[i][j],0,0]  #[x,y,vx,vy] initial values.
            
            # Set initial conditions
            p.set_positions(CI,Mx,My)
            p.set_magnetic_dipoles(mu_P,mu)
            
            # Find path
            X, Y, Vx, Vy = p.find_path()
            
            # Get Final positions
            Mat[i][j][0] = X[-1]
            Mat[i][j][1] = Y[-1]
    
    # Save final data
    np.savetxt('Mat_x.csv', Mat[:,:,0], delimiter=',')
    np.savetxt('Mat_y.csv', Mat[:,:,1], delimiter=',')
    
    # Show total time
    fin = time.time()
    print('Total time %.2f s'%(fin-inicio))
    
    ## Plot image of basin of atraction
    im = np.zeros((N,N,3))
    
    # Colour each pixel (i,j)
    for i in range(N):
        for j in range (N):
            
            # Find the distances to each magnet (or atractor)
            d=[]
            for k in range(len(Mx)):
                d.append( np.sqrt((Mat[i][j][0] - Mx[k])**2+(Mat[i][j][1]-My[k])**2) )
            
            # Set the color based on the minimum distance respecting to each magnet
            index = d.index(min(d))
            im[N-1-i,j, index] = 1
    
    # Show figure
    plt.figure()
    plt.imshow(im)
    plt.show()

def plot_im():
    return 0