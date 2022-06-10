"""
Created on Fri Jun  3 22:05:50 2022

"""

import CMPendulum.solver.pendulum as pend
import CMPendulum.solver.animation as ani
import matplotlib.pyplot as plt

import CMPendulum.analysis.basins as bs

#Put on the console for animation %matplotlib auto

#Create pendulum
p = pend.pendulum()

#Find trajectories
X, Y, Vx, Vy = p.find_path()
selfp = p.get_self()

#Plot trajectories
p.plot_table()              
p.plot_vector_space(res=30)
p.plot_path()
plt.legend(loc='upper right')
#plt.savefig('path.png',dpi=200)

#Animate movement
ani.animate_path(selfp)

#Plot potential, table,...
p.plot_potential(res=50)
#plt.savefig('potential.png',dpi=200)
p.plot_alltable()
plt.legend(loc='upper right')
#plt.savefig('alltable.png',dpi=200)

print('Done')