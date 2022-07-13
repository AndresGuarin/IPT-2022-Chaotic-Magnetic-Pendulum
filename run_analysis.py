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
import CMPendulum.solver.pendulum as pend

# In[] Basins -----------------------------------------------------------------------
bs = basin.basins()

#Positions table magnets
angles = np.deg2rad(np.array([-30,90,210]))
Mx = 0.1*np.cos(angles)        
My = 0.1*np.sin(angles)

#Finding final positions
#bs.basins(res=10, lims=[[-0.2,0.2],[-0.2,0.2]], params={'Mx':Mx, 'My':My,
#         'm':0.036, 'l':0.54, 'd':0.03, 'mu_P':1.84, 'mu':[1.94,1.94,1.94]},
#         timezone='Colombia')

#Plot basin taking the magnet positions as the atractor positions 
#bs.plot_im()
#plt.savefig('Images/basin.png',dpi=300)
#plt.show()


# In[] Path -------------------------------------------------------------------------
CI=[0.15,0.12,0,0]; CI2 = [0.151,0.119,0,0]
mu_P=1.84; mu=[1.94,1.94,1.94]

# Path 1
p = pend.pendulum()
p.set_positions(CI,Mx,My); p.set_magnetic_dipoles(mu_P, mu)
X,Y,Vx,Vy = p.find_path()
self = p.get_self()

pt.phase_portrait(self)
pt.time_series(self)

# Path 2
p2 = pend.pendulum()
p2.set_positions(CI2,Mx,My); p2.set_magnetic_dipoles(mu_P, mu)
X2,Y2,Vx2,Vy2 = p2.find_path()
self2 = p2.get_self()

ly = pt.lyapunov()
ly.plot_differences([self, self2], kind='log')
plt.savefig('Images/differences.png',dpi=300)
plt.show()

ly.plot_fit([0,2.76],plot_fit='lin')
plt.savefig('Images/lypunov_fit.png',dpi=300)
plt.show()