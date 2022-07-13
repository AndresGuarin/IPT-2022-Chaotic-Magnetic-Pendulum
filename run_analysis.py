"""
@authors:   Juan Guarin, Brayan Barajas, Angela Barajas
            Jose Garavito, Gabriela Sánchez, Juan Figueroa
Created on Fri Jun  3 22:05:50 2022

This code use the multipurpose module of CMPendulum to solve and plot the trajectories
of a magnetic pendulum of N magnets. This code was used for a solution of one problem
in the International Physicists' Tournament (ed. 2022). More info at https://iptnet.info/
"""
import numpy as np
import matplotlib.pyplot as plt
import CMPendulum.analysis.basins as basin
import CMPendulum.analysis.path as pt

# In[] Basins -----------------------------------------------------------------------
bs = basin.basins()

#Positions table magnets
angles = np.deg2rad(np.array([-30,90,210]))
Mx = 0.1*np.cos(angles)        
My = 0.1*np.sin(angles)

#Finding final positions
bs.basins(res=10, lims=[[-0.2,0.2],[-0.2,0.2]], params={'Mx':Mx, 'My':My,
         'm':0.036, 'l':0.54, 'd':0.03, 'mu_P':1.84, 'mu':[1.94,1.94,1.94]},
         timezone='Colombia')

#Plot basin taking the magnet positions as the atractor positions 
bs.plot_im()
plt.savefig('Images/basin.png',dpi=300)
plt.show()


# In[] Path -------------------------------------------------------------------------

