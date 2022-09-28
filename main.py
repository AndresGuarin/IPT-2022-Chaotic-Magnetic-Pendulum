"""
@authors:   Juan Guarin, Brayan Barajas, Angela Barajas
            Jose Garavito, Gabriela Sánchez, Juan Figueroa
Created on Fri Jun  3 22:05:50 2022

This code use the multipurpose module of CMPendulum to solve and plot the trajectories
of a magnetic pendulum of N magnets. This code was used for a solution of one problem
in the International Physicists' Tournament (ed. 2022). More info at https://iptnet.info/
"""

import matplotlib.pyplot as plt
from CMPendulum import pendulum as pend, animation as ani, basins as bs

#Put on the console for animation: %matplotlib auto

#Create pendulum
p = pend.pendulum()

#Find trajectories
X, Y, Vx, Vy = p.find_path(show=True)
selfp = p.get_self()

#Plot trajectories
p.plot_alltable()              
p.plot_vector_space(res=30)
p.plot_path()
plt.legend(loc='upper right')
plt.savefig('Images/path.png',dpi=200); plt.show()

#Animate movement
ani.animate_path(selfp)
plt.show()

#Plot potential, table,...
p.plot_potential(res=50)
plt.savefig('Images/potential.png',dpi=200); plt.show()

p.plot_alltable()
plt.legend(loc='upper right')
plt.savefig('Images/alltable.png',dpi=200); plt.show()

print('Done')