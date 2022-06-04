# In[] 
##Phase portrait 3D (x,y,v)

for i in range(2):
    # Get positions and velocity
    X=selfs[i].X
    Y=selfs[i].Y
    Vx=selfs[i].Vx
    Vy=selfs[i].Vy
    V = np.sqrt(Vx**2+Vy**2)*np.sign(Vy) #With sign

    # Create projection 3d
    plt.figure()
    ax = plt.axes(projection='3d')

    # Plot phae portrait
    plt.plot(X,Y,V,'-r',linewidth=0.5)

    # Put tilte and labels
    plt.title('Phase portrait 3D', fontsize=15)
    plt.xlabel('X [m]',fontsize=12)
    plt.ylabel('Y [m]',fontsize=12)
    ax.set_zlabel('Z [m]',fontsize=12)
    plt.savefig(f'Phase portrait_{i}.png',dpi=300)
    plt.show()

    # Time series
    plt.figure()

    # Create time array
    t=np.linspace(0,selfs[i].h*selfs[i].N,len(X))

    # Plot time series
    plt.plot(t,X,'-b',linewidth=0.5)

    # Put the title and labels
    plt.title('Time series of X',fontsize=15)
    plt.xlabel('Time [s]',fontsize=12)
    plt.ylabel('X [m]',fontsize=12)
    plt.savefig(f'Time series of X {i}.png',dpi=300)
    plt.show()
    

## Superposition
plt.figure()
ax = plt.axes(projection='3d')
colors=['blue','red']
#colors=['blue','red','green','orange']*3
for i in range(0,2):
    # Get positions and velocity
    X=selfs[i].X
    Y=selfs[i].Y
    Vx=selfs[i].Vx
    Vy=selfs[i].Vy
    V = np.sqrt(Vx**2+Vy**2)*np.sign(Vy) #With sign
    
    # Plot phae portrait
    plt.plot(X,Y,V,color=colors[i],linewidth=0.8,label=f'Path {i+1}',alpha=0.9)

# Put tilte and labels
plt.title('Phase portrait 3D', fontsize=15)
plt.xlabel(r'X [m]',fontsize=12)
plt.ylabel(r'Y [m]',fontsize=12)
plt.legend()
ax.set_zlabel(r'V sign(Vx) [m/s]',fontsize=12)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(9)
plt.show()
plt.savefig(f'Hex_pos_phase_portrait_{i}{i+1}.png',dpi=300)

# In[]

## Differents between paths
pair=3 #From 0 to 8

X1, Y1 = selfs[pair*2].X, selfs[pair*2].Y 
X2, Y2 = selfs[pair*2+1].X, selfs[pair*2+1].Y

# Distance between paths
nn = min([len(X1),len(X2)])
D=np.sqrt((X1[:nn]-X2[:nn])**2 + ((Y1[:nn]-Y2[:nn])**2))

# Time array
t=np.linspace(0,selfs[0].h * selfs[0].N ,len(D))

plt.figure()

# Plot difference vs time
#plt.plot(t,D, label='Distance')
plt.plot(t, np.log(D/D[0]), '-b',label='Log distance')

# Put labels, title, legend, ticks and grid
plt.ylabel(r'Log(D(t)/D(0)) [dimensionless]', fontsize = 12)
plt.xlabel('Time [s]', fontsize = 12)
plt.title('Chaotic behaviour, changing 1mm the initial positions', fontsize = 15)
plt.legend(fontsize=10, ncol=1)
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.grid(True)

# Save image
plt.savefig(f'r={datr[pair]} mu={datmu[pair]}/Apos_diference_trajectories_{pair*2+1}{pair*2+2}.png',dpi=300)

# Show image
plt.show()

##Check start and end
ti = 0
tf = 0.7263
start = int(ti/selfs[pair*2].h)
end = int(tf/selfs[pair*2].h)
print(start, end)

starts = [108999, 0, 0, 0, 0, 0, 0, 0]
ends =   [561249, 7117, 7262, 0, 0, 0, 0, 0]

start = starts[pair]
end = ends[pair]

def linear(x, m, b):
    return x*m + b

t_set = t[start:end+1]
D_set = D[start:end+1]
params, cov = curve_fit(linear, t_set, np.log(D_set/D_set[0]))
var = np.sqrt(np.diag(cov))

# Plot difference vs time
plt.figure()
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

plt.text(xtext, ytext, f'lambda = {np.round(params[0],3)}', fontsize=10)
plt.grid(True)
plt.savefig(f'r={datr[pair]} mu={datmu[pair]}/Apos_Lyapunov_{2*pair+1}{2*pair+2}.png',dpi=300)
plt.show()
Lambdas[pair] = params[0]

plt.figure()
t_Ly = np.linspace(t[start], t[end-1300],200)
y_Ly = D[start]*np.e**(params[0]*(t_Ly)+params[1])
plt.plot(t_Ly, y_Ly, '-',label=r'Fit $D(t) = %.5f \ e^{%.2f t + %.2f}$'%(np.round(D[start],5),np.round(params[0],2),np.round(params[1],2))
         ,linewidth=1.5, color='red', alpha=0.6)

#start2 = 121
#end2 = 166
#t_Ly2 = np.linspace(t[start2], t[end2+2],200)
#y_Ly2 = D[start2]*np.e**(params2[0]*(t_Ly2)+params2[1])
#plt.plot(t_Ly2, y_Ly2, '-',label=f'Fit 2 D(t) = {np.round(D[start2],4)}e^({np.round(params2[0],2)} t - {-np.round(params2[1],2)})',linewidth=1.5, color='orange', alpha=0.6)
plt.plot(t,D,'-b',label='Distance', linewidth=1.5, alpha=0.6)

plt.ylabel('D(t) [m]', fontsize = 12)
plt.xlabel('Time [s]', fontsize = 12)
plt.ylim(-0.0055, 0.42)
plt.title('Chaotic behaviour, changing 1mm initial positions', fontsize = 15)
plt.legend(fontsize=10, ncol=1, loc='upper left')
plt.xticks(size = 10)
plt.yticks(size = 10)
plt.savefig(f'r={datr[pair]} mu={datmu[pair]}/Apos_Exponential_divergence_{2*pair+1}{2*pair+2}.png',dpi=300)
plt.show()