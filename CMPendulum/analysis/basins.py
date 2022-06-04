import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

## Set initial parameters and conditions

# Resolution of tha basin of atraction
N=20

# Get data of initial conditions
t=np.linspace(-0.2,0.2,N)
mx, my = np.meshgrid(t,t)
Mat = np.zeros((N,N,2))

# Positions of magnets and magnetic dipoles
thett = np.deg2rad(30)
Mx = [0.1*np.cos(-thett), 0.1*np.cos(3*thett), 0.1*np.cos(7*thett)]         #Positions table magnets
My = [0.1*np.sin(-thett), 0.1*np.sin(3*thett), 0.1*np.sin(7*thett)]  
m = 0.036226                   #mass
mu_P = 1.84                    #magnetic dipole magnitude (pendulum)
mu = [1.94, 1.94, 1.94]  #magnetic dipoles in atraction form (table)
zone={'Colombia':'ETC/GMT-7','Denmark':'ETC/GMT+2'}

##Find the data of final positions
aux=-1
inicio = time.time()

# Run over each x initial condition
for i in range(0,len(mx)):
    
    # Print the advances of calculations and save the progress
    val = i/(N-1)*10
    if int(val) != aux:
        tz = pytz.timezone(zone[time_zone])
        print('[','#'*int(val), ']',f'{int(val)}/15',datetime.now(tz).strftime("%H:%M:%S"))
        aux=int(val)
        np.savetxt('Mat_adv_x.csv', Mat[:,:,0], delimiter=',')
        np.savetxt('Mat_adv_y.csv', Mat[:,:,1], delimiter=',')
    
    # Run over each y initial condition
    for j in range(0,len(my)):
        
        # Create pendulum
        pend = pendulum()
        CI = [mx[i][j],my[i][j],0,0]  #[x,y,vx,vy] initial values.
        
        # Set initial conditions
        pend.set_positions(CI,Mx,My)
        pend.set_magnetic_dipoles(mu_P,mu)
        
        # Find path
        X, Y, Vx, Vy = pend.find_path()
        
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
