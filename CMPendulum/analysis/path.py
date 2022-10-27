import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class lyapunov:
    def __init__(self, ):
        pass

    def plot_differences(self, selfp, kind='log'):
        """
        @params:
            selfp: self object
            Is the self object of the pendulum class that contains all the parameters and data
            of the trajectorie of the body.

            kind: string
            Values: 'log', 'normal'. Determines if the plot is logaritmic or not.

        @returns:
            none

        @example:
            Let's do an example with initial conditions randomly:
                >>> import CMPendulum.solver.pendulum as pend
                >>> import CMPendulum.analysis.path as pt
                >>> p = pend.pendulum(); p.find_path(Return='none')
                >>> self = p.get_self()
                >>> p2 = pend.pendulum()
                Ahora pongamos las mismas condiciones iniciales y número de imanes:
                >>> CI=np.array([self.x,self.y,self.vx,self.vy])+np.array([0.01,0.01,0,0])
                >>> mu_P = -self.mu_P_magn*np.sign(self.mu_P_dir[2])
                >>> mu = [-self.mu_magn[i]*np.sing(self.mu_dir[i][2]) for i in range(len(self.mu_magn))]
                >>> p2.set_positions(CI,self.Mx,self.My); p2.set_magnetic_dipoles(mu_P, mu)
                >>> p2.find_path(Return='none')
                >>> self2 = p2.get_self()
                >>> ly = pt.lyapunov()
                >>> ly.plot_differences([self, self2], kind='log', plot_fit='exp')
                Show plot of the difference of paths.
        """

        # Get data of paths
        X1, Y1 = selfp[0].X, selfp[0].Y 
        X2, Y2 = selfp[1].X, selfp[1].Y

        # Distance between paths
        nn = min([len(X1),len(X2)])
        self.D=np.sqrt((X1[:nn]-X2[:nn])**2 + ((Y1[:nn]-Y2[:nn])**2))

        # Time array
        self.t=np.linspace(0,selfp[0].h * len(self.D) ,len(self.D))

        plt.figure(figsize=(5,5))
        # Plot difference vs time
        if kind=='log':
            plt.plot(self.t,self.D, label='Distance')
        elif kind=='normal':
            plt.plot(self.t, np.log(self.D/self.D[0]), '-b',label='Log distance')

        # Put labels, title, legend, ticks and grid
        plt.ylabel(r'$log\left(\frac{D(t)}{D(0)}\right)$', fontsize = 12)
        plt.xlabel('Time [s]', fontsize = 12)
        plt.title('Changes in Paths due Initial Conditions', fontsize = 15)
        plt.legend(fontsize=10, ncol=1)
        plt.xticks(size = 10)
        plt.yticks(size = 10)
        plt.grid(True)

        self.X1 = X1; self.X2 = X2
        self.Y1 = Y1; self.Y2 = Y2
        self.h = selfp[0].h

    def plot_fit(self, time_range, plot_fit='exp'):
        """
        @params:
            time_range: array-like
            Structure [ti,tf] that are the limits of the time interval in which
            we fit the data.

            plot_tipe: string
            Values: 'lin', 'exp'. Decides if you want to plot the fit as a linear fit
            (in the log domain) or as the exponensial function (in the normal domain).
        """

        #Check start and end
        ti = time_range[0]
        tf = time_range[1]
        start = int(ti/self.h)
        end = int(tf/self.h)

        def linear(x, m, b):
            return x*m + b

        t_set = self.t[start:end+1]
        D_set = self.D[start:end+1]
        params, cov = curve_fit(linear, t_set, np.log(D_set/D_set[0]))
        var = np.sqrt(np.diag(cov))

        # Plot difference vs time
        plt.figure(figsize=(5,5))
    
        if plot_fit == 'lin':
            plt.plot(t_set, np.log(D_set/D_set[0]),color='blue', label='Log distance')
            plt.plot(t_set, linear(t_set, params[0], params[1]), color='orange', label=f'Linear Fit: log(D)={np.round(params[0],2)}t+{np.round(params[1],2)}')

            # Put labels, title, legend, ticks and grid
            plt.ylabel('Log(D(t) / D0) [dimensionless]', fontsize = 12)
            plt.xlabel('Time [s]', fontsize = 12)
            plt.title('Chaotic behaviour, changing 1mm initial positions', fontsize = 15)
            plt.legend(fontsize=10, ncol=1)
            plt.xticks(size = 10)
            plt.yticks(size = 10)

            xtext = 5/7*(t_set[-1]-t_set[0]) + t_set[0]
            log_set = np.log(D_set/D_set[0])
            ytext = 2/6*(max(log_set)-min(log_set)) + min(log_set)
            plt.text(xtext, ytext, 'lambda = %.3f'%(params[0]), fontsize=10)
            
            plt.grid(True)
            print('Lyapunov Exponent----------------')
            print('lambda = %.3f'%(params[0]))
            
        elif plot_fit=='exp':
            t_Ly = np.linspace(self.t[start], self.t[end-1300],200)
            y_Ly = self.D[start]*np.e**(params[0]*(t_Ly)+params[1])
            plt.plot(t_Ly, y_Ly, '-',label=r'Fit $D(t) = %.5f \ e^{%.2f t + %.2f}$'%(np.round(self.D[start],5),np.round(params[0],2),np.round(params[1],2))
                    ,linewidth=1.5, color='red', alpha=0.6)
            plt.plot(t,D,'-b',label='Distance', linewidth=1.5, alpha=0.6)

            # Put labels, title, legend, ticks and grid
            plt.ylabel('D(t) [m]', fontsize = 12)
            plt.xlabel('Time [s]', fontsize = 12)
            plt.title('Chaotic behaviour, changing 1mm initial positions', fontsize = 15)
            plt.legend(fontsize=10, ncol=1, loc='upper left')
            plt.xticks(size = 10)
            plt.yticks(size = 10)
            plt.grid(True)

            xtext = 5/7*(self.t[-1]-self.t[0]) + self.t[0]
            ytext = 2/6*(max(self.D)-min(self.D)) + min(self.D)
            plt.text(xtext, ytext, 'lambda = %.3f'%(params[0]), fontsize=10)