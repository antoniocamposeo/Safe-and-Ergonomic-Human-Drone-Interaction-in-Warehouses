import numpy as np
import matplotlib.pyplot as plt
from iLQR_APF import *
from Quadrotor import *
import APF_collision_avoidanca
from APF_collision_avoidanca import *
from sklearn.metrics import mean_squared_error

quadrotor = Quadrotor(0.2, 0.1, 5, 9.81, 0.01)
# time step s
quadrotor.dt = 0.01
# mass Kg
quadrotor.m = 1
# acceleration m/s^2
quadrotor.g = 9.81
# radius
quadrotor.r = 0.2
# inertia
quadrotor.I = 0.0133

N = 501
x_0 = np.array([620, 1.5, 0, 0, 0, 0])

x_bar = np.array([620, 5, 0, 0, 0, 0])

u_0 = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

u_bar = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

Q = np.identity(6)

R = np.identity(2)

# 1° DECOLLO X-Y -- BASE DI RICARICA A ZONA DI HOVERING

x_t_1, u_opt = LQR_2EQ(N, x_0, x_bar, u_0, u_bar, quadrotor, R, Q)
x1 = x_t_1.copy()

t = np.arange(N) * 0.01
plt.figure(2)

# plt.plot(t, x_t_1[0, 0:N], label="x",color='k',markersize=10)
plt.plot(t, x_t_1[1, 0:N], label="y", color='k', markersize=10)
plt.plot(t[1], x_t_1[1, 0], 'ro', markersize=10)
plt.plot(t[N - 1], x_t_1[1, N - 1], 'ro', color='g', markersize=10)
# plt.plot(t[1], x_t_1[0,0], 'ro', markersize=10)
# plt.plot(t[N-1], x_t_1[0,N-1], 'ro', color='g', markersize=10)
plt.xlabel('Time')
plt.ylabel('Position y')
plt.grid()
plt.legend(loc="upper left")
plt.title("1° DECOLLO")
plt.show()
plt.plot(x_t_1[0, 0:N - 1], x_t_1[1, 0:N - 1])



mse_x = mean_squared_error([620, 620], [x_t_1[0, 0], x_t_1[0, N - 1]])
mse_y = mean_squared_error([1.5, 5], [x_t_1[1, 1], x_t_1[1, N - 1]])
print(mse_x, mse_y)


# 1° PLANNING TRAJECTORY X-Z --ZONA DI HOVERING A POSIZIONE DI PRELIEVO
# Generate some points
nrows = 3000;
ncols = 6500;


def create_obtacle(nrows, ncols):
    obstacle = np.zeros((nrows, ncols))
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))

    # Generate some obstacle

    obstacle[0:20, :] = True;
    # obstacle[]
    obstacle[nrows - 40:, :] = True;
    obstacle[:, 0:40] = True;
    obstacle[:, ncols - 40:] = True;

    obstacle[10:1300, 650:700] = True;
    obstacle[550:600, 0:700] = True;
    obstacle[1100:1150, 0:700] = True;
    obstacle[1650:1700, 0:300] = True;
    obstacle[2200:2250, 0:700] = True;
    obstacle[2200:3000, 0:700] = True;
    obstacle[2200:3000, 6000:6500] = True;
    obstacle[10:500, 6000:6500] = True;
    # X MAX 3000
    # Y MAX 5000

    obstacle[300:400, 1100:1900] = True;
    obstacle[450:550, 1100:1900] = True;

    obstacle[700:800, 1100:1900] = True;
    obstacle[850:950, 1100:1900] = True;

    obstacle[1300:1400, 1100:1900] = True;
    obstacle[1450:1550, 1100:1900] = True;

    obstacle[1850:1950, 1100:1900] = True;
    obstacle[2000:2100, 1100:1900] = True;

    obstacle[2250:2350, 1100:1900] = True;
    obstacle[2400:2500, 1100:1900] = True;

    obstacle[300:400, 2300:3100] = True;
    obstacle[450:550, 2300:3100] = True;

    obstacle[700:800, 2300:3100] = True;
    obstacle[850:950, 2300:3100] = True;

    obstacle[1300:1400, 2300:3100] = True;
    obstacle[1450:1550, 2300:3100] = True;

    obstacle[1850:1950, 2300:3100] = True;
    obstacle[2000:2100, 2300:3100] = True;

    obstacle[2250:2350, 2300:3100] = True;
    obstacle[2400:2500, 2300:3100] = True;

    obstacle[300:400, 3400:4200] = True;
    obstacle[450:550, 3400:4200] = True;

    obstacle[700:800, 3400:4200] = True;
    obstacle[850:950, 3400:4200] = True;

    obstacle[1300:1400, 3400:4200] = True;
    obstacle[1450:1550, 3400:4200] = True;

    obstacle[1850:1950, 3400:4200] = True;
    obstacle[2000:2100, 3400:4200] = True;

    obstacle[2250:2350, 3400:4200] = True;
    obstacle[2400:2500, 3400:4200] = True;

    obstacle[300:400, 4600:5400] = True;
    obstacle[450:550, 4600:5400] = True;

    obstacle[700:800, 4600:5400] = True;
    obstacle[850:950, 4600:5400] = True;

    obstacle[1300:1400, 4600:5400] = True;
    obstacle[1450:1550, 4600:5400] = True;

    obstacle[1850:1950, 4600:5400] = True;
    obstacle[2000:2100, 4600:5400] = True;

    obstacle[2250:2350, 4600:5400] = True;
    obstacle[2400:2500, 4600:5400] = True;

    return obstacle, x, y


obstacle, x, y = create_obtacle(nrows, ncols)

plt.imshow(obstacle, 'Blues_r', origin='lower')




# Display repulsive potential
plt.imshow(APF_collision_avoidanca.repulsive(obstacle), 'gray', origin='lower')
plt.title('Repulsive Potential')

# Compute attractive force
start = [620, 1500]

goal = [3150, 850]

# Display attractive potential
plt.imshow(APF_collision_avoidanca.attractice(x, y, goal), 'gray', origin='lower')
plt.title('Attractive Potential')
# Display 2D configuration space
plt.imshow(1 - obstacle, 'Blues_r', origin='lower')
plt.plot(start[0], start[1], 'ro', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
plt.xlabel('x')
plt.ylabel('y')

plt.title('Configuration Space')
# Combine terms
f = APF_collision_avoidanca.attractice(x, y, goal) + APF_collision_avoidanca.repulsive(obstacle)
plt.imshow(f, 'Blues_r', origin='lower')
plt.title('Total Potential')

# Plan route
route1 = APF_collision_avoidanca.GradientBasedPlanner(f, start, goal, 10000)

# Compute gradients for visualization
[gx, gy] = np.gradient(-f)
plt.figure(figsize=(12, 8))
plt.imshow(gy, 'Blues_r', origin='lower')
plt.title('Gx=df/dx - gradient')
plt.figure(figsize=(12, 8))
plt.imshow(gx, 'Blues_r', origin='lower')
plt.title('Gy=df/dy - gradient')

# Velocities plot
skip = 50
xidx = np.arange(0, ncols, skip)
yidx = np.arange(0, nrows, skip)

APF_collision_avoidanca.gradient_plot(x, y, gy, gx, skip=50)
plt.plot(route1[:, 0], route1[:, 1], color='yellow', linewidth=3)
plt.plot(start[0], start[1], 'ro', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
plt.imshow(f, 'Blues', origin='lower')
# plt.imshow(obstacle, 'Blues_r', origin='lower')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure(1)
plt.plot(route1[:, 0], route1[:, 1], linewidth=3, color='k')
plt.plot(start[0], start[1], 'ro', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()



x_new, y_new = APF_collision_avoidanca.Split_BacktrackStates(route1)
# new_route = APF_collision_avoidanca.Interpolation(x,y,2)
x = [int(x) for x in x_new]
y = [int(y) for y in y_new]
# print(x)
new_route = np.array(list(zip(x, y)))
# print(new_route.T)
print(len(new_route))


N = len(new_route) * 200

print(N)
x_0 = np.array([start[0], start[1], 0, 0, 0, 0])
u_0 = np.array([(quadrotor.m * quadrotor.g) / 2, (quadrotor.m * quadrotor.g) / 2])
x_bar = np.zeros([6, N])

for i in range(1, len(new_route)):
    x_bar[0, i * 200 - 200:200 * i] = new_route[i][0]
    x_bar[1, i * 200 - 200:200 * i] = new_route[i][1]
# print()
# print(len(route))
# print(x_bar.shape[1])

# x_bar[0,N-100:N] = route[len(new_route)-1][0]
# x_bar[1,N-100:N] = route[len(new_route)-1][1]
u_bar = np.zeros([2, N - 1])
u_bar[0, :] = np.repeat(((quadrotor.m * quadrotor.g) / 2), N - 1)
u_bar[1, :] = np.repeat(((quadrotor.m * quadrotor.g) / 2), N - 1)

# Q e R per iLQR
# matrice Q 6x6 -- State Cost matrix
Q = np.identity(6) * 1
Q[0, 0] = 5
Q[1, 1] = 5
Q[2, 2] = 10
Q[3, 3] = 10
Q[4, 4] = 10
Q[5, 5] = 10
# matrice R 2x2 -- Input cost matrix
R = np.identity(2) * 1
R[0, 0] = 2000
R[1, 1] = 2000

# x_t, u_opt = iLQR(N, x_0, x_bar, u_0, u_bar, R, Q, quadrotor)
x_t, u_opt = LQR(N, x_0, x_bar, u_0, u_bar, quadrotor, R, Q)
x2 = x_t.copy()
print(x_t)

t = np.arange(N - 1) * 0.01

plt.figure(2)
plt.plot(t, x_t[0, 0:N - 1], label="x")
plt.plot(t, x_t[1, 0:N - 1], label="y")
plt.plot(t[0:N - 200], x_bar[0, 0:N - 200], label="trajectory")
plt.plot(t[0:N - 200], x_bar[1, 0:N - 200], label="trajectory")
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend(loc="upper left")
plt.title("iLQR trajectory ")
plt.show()

plt.figure(2)
plt.plot(x_t[0, 0:N - 1], x_t[1, 0:N - 1], label="x-y")
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend(loc="upper right")
plt.title("X-Y iLQR trajectory")
plt.show()


mse_x = mean_squared_error(x_bar[0, :N - 200], x_t[0, :N - 200])
mse_y = mean_squared_error(x_bar[1, :N - 200], x_t[1, :N - 200])
print(mse_x)
print(mse_y)
print(x_bar[0, :N - 800])

# %%

x_1 = [(int(x_t[0][i]), int(x_t[1][i])) for i in range(0, x_t.shape[1], 2)]

x_2 = np.array(x_1)

skip = 50

xidx = np.arange(0, ncols, skip)
yidx = np.arange(0, nrows, skip)

# APF_collision_avoidanca.gradient_plot(x,y, gy,gx, skip=50)
# plt.plot(start[0], start[1], 'ro', markersize=10)
# plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
# plt.plot(x_2[:,0], x_2[:,1], linewidth=1)
plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

APF_collision_avoidanca.gradient_plot(x, y, gy, gx, skip=50)

plt.plot(x_2[:14300, 0], x_2[:14300, 1], 'y', linewidth=3)
plt.plot(start[0], start[1], 'ro', color='g', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='b', markersize=10)

# plt.plot(start[0], start[1],'ro',color='c',markersize=10)
# plt.plot(goal[0], goal[1], 'ro', color='b', markersize=10)
plt.imshow(f, 'Blues', origin='lower')
# plt.imshow(obstacle, 'Blues_r', origin='lower')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.plot(start[0], start[1], 'ro', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
plt.plot(route1[:N - 500, 0], route1[:N - 500, 1], linewidth=3)
plt.plot(x_2[:N - 500, 0], x_2[:N - 500, 1], linewidth=3)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

# %%

# 1* ATTERRAGGIO DA HOVERING A ZONA DI PRELIEVO
print(x_t[0, 27799])
N_1 = 2000

x_0 = np.array([x_t[0, 27799], 5, 0, 0, 0, 0])

x_bar = np.array([3150, 2, 0, 0, 0, 0])

u_0 = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

u_bar = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

Q = np.identity(6) * 250
Q[0, 0] = 20
Q[1, 1] = 20
Q[2, 2] = 300
Q[3, 3] = 300
Q[4, 4] = 300
Q[5, 5] = 300

R = np.identity(2) * 100

x_t_3, u_opt = LQR_2EQ(N_1, x_0, x_bar, u_0, u_bar, quadrotor, R, Q)
x3 = x_t_3.copy()
print(x_t_3[0, N_1 - 1])
print(x_t_3[1, N_1 - 1])

# %%

t = np.arange(N_1) * 0.01
print(t[N_1 - 1])
# plt.figure(2)
# plt.plot(t, x_t_3[0, 0:N_1], label="x")
plt.plot(t, x_t_3[1, 0:N_1], label="y")
# plt.plot(t, x_t_3[3, 0:N_1], label="VX")
# plt.plot(t, x_t_3[4, 0:N_1], label="Vy")
plt.plot(t[1], x_t_3[1, 0], 'ro', markersize=10)
plt.plot(t[N_1 - 1], x_t_3[1, N_1 - 1], 'ro', color='g', markersize=10)

plt.xlabel('Time')
plt.ylabel('Position x-y')
plt.legend(loc="upper left")
plt.title("1° ATTERRRAGGIO")
plt.show()
plt.plot(t, x_t_3[0, 0:N_1], label="x")
plt.plot(t[1], x_t_3[0, 0], 'ro', markersize=10)
plt.plot(t[N_1 - 1], x_t_3[0, N_1 - 1], 'ro', color='g', markersize=10)

plt.xlabel('Time')
plt.ylabel('Position x-y')
plt.legend(loc="upper left")
plt.title("1° ATTERRRAGGIO")
plt.show()

plt.plot(x_t_3[0, 0:N_1], x_t_3[1, 0:N_1], label="x")

# %%

mse_x = mean_squared_error([x_t[0, 27799], 3150], [x_t_3[0, 0], x_t_3[0, N_1 - 1]])
mse_y = mean_squared_error([5, 2], [x_t_3[1, 1], x_t_3[1, N_1 - 1]])
print(mse_y)
print(mse_x)

# %%

# PLOT COMPLETO ASSE X-Y
x_t[1, :] = 5
N_1 = x_t_1.shape[1]
N_2 = x_t.shape[1]
N_3 = x_t_3.shape[1]
print(x_t_3)
print(N_1, N_2, N_3)
N_tot = N_1 + N_2 + N_3
print(N_tot)
x_T = np.empty([6, N_tot])
print(x_T.shape[1])

x_T[:, :501] = x_t_1
x_T[:, 501:(N_1 + N_2)] = x_t
x_T[:, (N_1 + N_2):(N_1 + N_2 + N_3)] = x_t_3

t = np.arange(x_T.shape[1]) * 0.01
print(t)
plt.plot(t, x_T[1, :], 'black')

# %%

# 2° DECOLLO ---- XY MASSA COSIDERATA 0.300KG

quadrotor = Quadrotor(0.2, 0.1, 5, 9.81, 0.01)
# time step s
quadrotor.dt = 0.01
# mass Kg
quadrotor.m = 1.3
# acceleration m/s^2
quadrotor.g = 9.81
# radius
quadrotor.r = 0.2
# inertia
quadrotor.I = 0.0173

N = 501
x_0 = np.array([3150, 2, 0, 0, 0, 0])

x_bar = np.array([3150, 5, 0, 0, 0, 0])

u_0 = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

u_bar = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

Q = np.identity(6)

R = np.identity(2)

# %%

# 2° DECOLLO X-Y -- BASE DI RICARICA A ZONA DI HOVERING

x_t_1_1, u_opt = LQR_2EQ(N, x_0, x_bar, u_0, u_bar, quadrotor, R, Q)

x4 = x_t_1_1.copy()

# %%

t = np.arange(N) * 0.01
print(t[N - 1])
# plt.figure(2)
# #plt.plot(t, x_t[0, 0:N], label="x")
# plt.plot(t, x_t[1, 0:N], label="y")
# # plt.plot(t[0], x_bar[0], 'ro', markersize=5)
# # plt.plot(t[N-1], x_bar[1], 'ro', color='green', markersize=10)
# plt.xlabel('Time')
# plt.ylabel('Position x-y')
# plt.legend(loc="upper left")
# plt.title("1° DECOLLO")
# # plt.show()
# print(x_t[1,N-1])

fig, axs = plt.subplots(2)

fig.suptitle('2° DECOLLO')
axs[0].plot(t, x_t_1_1[1, 0:N], '+', label="y", color='b')
axs[0].set_title('Axis y')
axs[1].plot(t, x_t_1_1[0, 0:N], '+', label="x", color='b')
axs[0].set_title('Axis x')
axs[0].grid()
axs[1].grid()
for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# %%

mse_x = mean_squared_error([3150, 3150], [x4[0, 0], x4[0, N - 1]])
mse_y = mean_squared_error([2, 5], [x4[1, 1], x4[1, N - 1]])
print(mse_y)
print(mse_x)

# %%

# 2° PLANNING TRAJECTORY X-Z --ZONA DI HOVERING A POSIZIONE DI PRELIEVO - MASSA 0.300KG

nrows = 3000;
ncols = 6500;


def create_obtacle(nrows, ncols):
    obstacle = np.zeros((nrows, ncols))
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))

    # Generate some obstacle

    obstacle[0:20, :] = True;
    # obstacle[]
    obstacle[nrows - 40:, :] = True;
    obstacle[:, 0:40] = True;
    obstacle[:, ncols - 40:] = True;

    obstacle[10:1300, 650:700] = True;
    obstacle[550:600, 0:700] = True;
    obstacle[1100:1150, 0:700] = True;
    obstacle[1650:1700, 0:300] = True;
    obstacle[2200:2250, 0:700] = True;
    obstacle[2200:3000, 0:700] = True;
    obstacle[2200:3000, 6000:6500] = True;
    obstacle[10:500, 6000:6500] = True;
    # X MAX 3000
    # Y MAX 5000

    obstacle[300:400, 1100:1900] = True;
    obstacle[450:550, 1100:1900] = True;

    obstacle[700:800, 1100:1900] = True;
    obstacle[850:950, 1100:1900] = True;

    obstacle[1300:1400, 1100:1900] = True;
    obstacle[1450:1550, 1100:1900] = True;

    obstacle[1850:1950, 1100:1900] = True;
    obstacle[2000:2100, 1100:1900] = True;

    obstacle[2250:2350, 1100:1900] = True;
    obstacle[2400:2500, 1100:1900] = True;

    obstacle[300:400, 2300:3100] = True;
    obstacle[450:550, 2300:3100] = True;

    obstacle[700:800, 2300:3100] = True;
    obstacle[850:950, 2300:3100] = True;

    obstacle[1300:1400, 2300:3100] = True;
    obstacle[1450:1550, 2300:3100] = True;

    obstacle[1850:1950, 2300:3100] = True;
    obstacle[2000:2100, 2300:3100] = True;

    obstacle[2250:2350, 2300:3100] = True;
    obstacle[2400:2500, 2300:3100] = True;

    obstacle[300:400, 3400:4200] = True;
    obstacle[450:550, 3400:4200] = True;

    obstacle[700:800, 3400:4200] = True;
    obstacle[850:950, 3400:4200] = True;

    obstacle[1300:1400, 3400:4200] = True;
    obstacle[1450:1550, 3400:4200] = True;

    obstacle[1850:1950, 3400:4200] = True;
    obstacle[2000:2100, 3400:4200] = True;

    obstacle[2250:2350, 3400:4200] = True;
    obstacle[2400:2500, 3400:4200] = True;

    obstacle[300:400, 4600:5400] = True;
    obstacle[450:550, 4600:5400] = True;

    obstacle[700:800, 4600:5400] = True;
    obstacle[850:950, 4600:5400] = True;

    obstacle[1300:1400, 4600:5400] = True;
    obstacle[1450:1550, 4600:5400] = True;

    obstacle[1850:1950, 4600:5400] = True;
    obstacle[2000:2100, 4600:5400] = True;

    obstacle[2250:2350, 4600:5400] = True;
    obstacle[2400:2500, 4600:5400] = True;

    return obstacle, x, y


obstacle, x, y = create_obtacle(nrows, ncols)

plt.figure(figsize=(20, 8))
plt.imshow(obstacle, 'copper_r', origin='lower')
# plt.colorbar()


# %%


# Display repulsive potential
plt.imshow(APF_collision_avoidanca.repulsive(obstacle), 'gray', origin='lower')
plt.title('Repulsive Potential')

# Compute attractive force
start = [3150, 900]

goal = [6000, 1000]

# Display attractive potential
plt.imshow(APF_collision_avoidanca.attractice(x, y, goal), 'gray', origin='lower')
plt.title('Attractive Potential')
# Display 2D configuration space
plt.imshow(1 - obstacle, 'Blues_r', origin='lower')
plt.plot(start[0], start[1], 'ro', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
plt.xlabel('x')
plt.ylabel('y')

plt.title('Configuration Space')
# Combine terms
f = APF_collision_avoidanca.attractice(x, y, goal) + APF_collision_avoidanca.repulsive(obstacle)
plt.imshow(f, 'Blues_r', origin='lower')
plt.title('Total Potential')

# Plan route
route2 = APF_collision_avoidanca.GradientBasedPlanner(f, start, goal, 10000)

# Compute gradients for visualization
[gx, gy] = np.gradient(-f)
plt.figure(figsize=(12, 8))
plt.imshow(gy, 'Blues_r', origin='lower')
plt.title('Gx=df/dx - gradient')
plt.figure(figsize=(12, 8))
plt.imshow(gx, 'Blues_r', origin='lower')
plt.title('Gy=df/dy - gradient')

# Velocities plot
skip = 10
xidx = np.arange(0, ncols, skip)
yidx = np.arange(0, nrows, skip)

APF_collision_avoidanca.gradient_plot(x, y, gy, gx, skip=50)
plt.plot(route2[:, 0], route2[:, 1], color='yellow', linewidth=3, alpha=1)
plt.plot(route1[:, 0], route1[:, 1], color='royalblue', linewidth=3)
plt.plot(620, 1500, 'ro', color='g', markersize=10)
plt.plot(start[0], start[1], 'ro', color='b', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='r', markersize=10)
plt.imshow(f, 'Blues_r', origin='lower')
# plt.imshow(obstacle, 'Blues_r', origin='lower')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Gradient', 'First Trajectory Route ', 'Second Trajectory Route', 'Charging Base', 'Picking Bay',
            'Palletizing Area'], loc="upper left")
plt.show()

plt.figure(1)
plt.plot(route2[:, 0], route2[:, 1], linewidth=3, color='k')
plt.plot(start[0], start[1], 'ro', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()



x_new, y_new = APF_collision_avoidanca.Split_BacktrackStates(route2)
# new_route = APF_collision_avoidanca.Interpolation(x,y,2)
x = [int(x) for x in x_new]
y = [int(y) for y in y_new]
print(x)
new_route = np.array(list(zip(x, y)))
# print(new_route.T)
# print(len(new_route))

# %%


N = len(new_route) * 200

print(N)
x_0 = np.array([3150, 900, 0, 0, 0, 0])
u_0 = np.array([(quadrotor.m * quadrotor.g) / 2, (quadrotor.m * quadrotor.g) / 2])
x_bar = np.zeros([6, N])

for i in range(1, len(new_route)):
    x_bar[0, i * 200 - 200:200 * i] = new_route[i][0]
    x_bar[1, i * 200 - 200:200 * i] = new_route[i][1]
# print()
# print(len(route))
# print(x_bar.shape[1])

# x_bar[0,N-100:N] = route[len(new_route)-1][0]
# x_bar[1,N-100:N] = route[len(new_route)-1][1]
u_bar = np.zeros([2, N - 1])
u_bar[0, :] = np.repeat(((quadrotor.m * quadrotor.g) / 2), N - 1)
u_bar[1, :] = np.repeat(((quadrotor.m * quadrotor.g) / 2), N - 1)

# Q e R per iLQR
# matrice Q 6x6 -- State Cost matrix
Q = np.identity(6) * 1
Q[0, 0] = 100
Q[1, 1] = 100
Q[2, 2] = 200
Q[3, 3] = 200
Q[4, 4] = 200
Q[5, 5] = 200
# matrice R 2x2 -- Input cost matrix
R = np.identity(2) * 1
R[0, 0] = 2000
R[1, 1] = 2000

# x_t, u_opt = iLQR(N, x_0, x_bar, u_0, u_bar, R, Q, quadrotor)
x_t_1_2, u_opt = LQR(N, x_0, x_bar, u_0, u_bar, quadrotor, R, Q)
x5 = x_t_1_2.copy()
print(x_t_1_2)

t = np.arange(N - 1) * 0.01

plt.figure(2)
plt.plot(t, x_t_1_2[0, 0:N - 1], label="x")
plt.plot(t, x_t_1_2[1, 0:N - 1], label="y")
plt.plot(t[0:N - 200], x_bar[0, 0:N - 200], label="trajectory")
plt.plot(t[0:N - 200], x_bar[1, 0:N - 200], label="trajectory")
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend(loc="upper left")
plt.title("iLQR trajectory ")
plt.show()

plt.figure(2)
plt.plot(x_t_1_2[0, 0:N - 200], x_t_1_2[1, 0:N - 200], label="x-y")
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend(t, loc="upper right")
plt.title("X-Y iLQR trajectory")
plt.show()

# %%

mse_x = mean_squared_error(x_bar[0, :N - 500], x5[0, :N - 500])
mse_y = mean_squared_error(x_bar[1, :N - 500], x5[1, :N - 500])
print(mse_x)
print(mse_y)

# %%

x_1 = [(int(x_t_1_2[0][i]), int(x_t_1_2[1][i])) for i in range(0, x_t_1_2.shape[1], 2)]

x_2_1 = np.array(x_1)

skip = 50

xidx = np.arange(0, ncols, skip)
yidx = np.arange(0, nrows, skip)

APF_collision_avoidanca.gradient_plot(x, y, gy, gx, skip=50)
# plt.plot(start[0], start[1], 'ro', markersize=10)
# plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
# plt.plot(x_2[:,0], x_2[:,1], linewidth=1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

APF_collision_avoidanca.gradient_plot(x, y, gy, gx, skip=50)

plt.plot(x_2_1[:, 0], x_2_1[:, 1], 'b', linewidth=3)
plt.plot(start[0], start[1], 'ro', markersize=10)
plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)

# plt.imshow(obstacle, 'Blues_r', origin='lower')
# plt.imshow(f, 'Blues_r', origin='lower')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure(figsize=(20, 8))
# APF_collision_avoidanca.gradient_plot(x,y, gy,gx, skip=50)
plt.imshow(obstacle, 'Blues_r', origin='lower')
plt.plot(x_2[:14300, 0], x_2[:14300, 1], color='mediumorchid', linewidth=3, label='First Free Flight without Load')
plt.plot(x_2_1[:, 0], x_2_1[:, 1], color='gold', linewidth=3, label='Second Free Flight with Load')
plt.plot(620, 1500, 'ro', color='r', markersize=10, label='Charging Base')
plt.plot(start[0], start[1], 'ro', color='b', markersize=10, label='Picking Bay')
plt.plot(goal[0], goal[1], 'ro', color='g', markersize=10, label='Palletizing Area')
# plt.imshow(obstacle, 'Blues_r', origin='lower')
# plt.imshow(f, 'PuBu', origin='lower')
plt.imshow(f, 'Blues', origin='lower', alpha=0.9)
# plt.colorbar()
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.legend(loc="upper left")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# plt.plot(start[0], start[1], 'ro', markersize=10)
# plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10)
# plt.plot(route[:,0], route[:,1], linewidth=3)
# plt.plot(x_2[:,0], x_2[:,1], linewidth=3)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid()
# plt.show()

# %%

plt.figure(figsize=(20, 8))
# APF_collision_avoidanca.gradient_plot(x,y, gy,gx, skip=50)
plt.imshow(obstacle, 'Blues_r', origin='lower')
# plt.plot(x_2[:14300,0], x_2[:14300,1],color='mediumorchid',linewidth=3,label='First Free Flight without Load' )
plt.plot(x_2_1[:, 0], x_2_1[:, 1], color='gold', linewidth=3, label='Second Free Flight with Load')
plt.plot(620, 1500, 'ro', color='r', markersize=10, label='Charging Base')
plt.plot(start[0], start[1], 'ro', color='b', markersize=10, label='Picking Bay')
plt.plot(goal[0], goal[1], 'ro', color='g', markersize=10, label='Palletizing Area')

plt.plot(3400, 950, 'ro', color='gold', linewidth=3)

plt.plot(x_2_1[1500, 0], x_2_1[1500, 1], 'ro', color='green', linewidth=3)

# plt.imshow(obstacle, 'Blues_r', origin='lower')
# plt.imshow(f, 'PuBu', origin='lower')
plt.imshow(f, 'Blues', origin='lower', alpha=0.9)
# plt.colorbar()
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.legend(loc="upper left")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

## 3400 950 punto dello scaffale


# %%

import math

p = [3500, 950]
q = [x_2_1[1500, 0], x_2_1[1500, 1]]

# Calculate Euclidean distance
print(math.dist(p, q))

# %%

# 2* ATTERRAGGIO DA HOVERING A ZONA DI PRELIEVO
print(x_t_1_2.shape)
N_1 = 2000

x_0 = np.array([6000, 5, 0, 0, 0, 0])

x_bar = np.array([6000, 2, 0, 0, 0, 0])

u_0 = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

u_bar = np.array([quadrotor.m * quadrotor.g / 2, quadrotor.m * quadrotor.g / 2])

Q = np.identity(6) * 250
Q[0, 0] = 200
Q[1, 1] = 200
Q[2, 2] = 10
Q[3, 3] = 10
Q[4, 4] = 10
Q[5, 5] = 10

R = np.identity(2) * 3000

x_t_4, u_opt = LQR_2EQ(N_1, x_0, x_bar, u_0, u_bar, quadrotor, R, Q)
print(x_t_4[0, N_1 - 1])
print(x_t_4[1, N_1 - 1])
x6 = x_t_4.copy()
t = np.arange(N_1) * 0.01
print(t[N_1 - 1])
# plt.figure(2)
# plt.plot(t, x_t_3[0, 0:N_1], label="x")
plt.plot(t, x_t_4[1, 0:N_1], label="y")
# plt.plot(t, x_t_3[3, 0:N_1], label="VX")
# plt.plot(t, x_t_3[4, 0:N_1], label="Vy")
plt.plot(t[1], x_t_4[1, 0], 'ro', markersize=10)
plt.plot(t[N_1 - 1], x_t_4[1, N_1 - 1], 'ro', color='g', markersize=10)

plt.xlabel('Time')
plt.ylabel('Position x-y')
plt.legend(loc="upper left")
plt.title("1° ATTERRRAGGIO")
plt.show()
plt.plot(t, x_t_4[0, 0:N_1], label="x")
plt.plot(t[1], x_t_4[0, 0], 'ro', markersize=10)
plt.plot(t[N_1 - 1], x_t_4[0, N_1 - 1], 'ro', color='g', markersize=10)

plt.xlabel('Time')
plt.ylabel('Position x-y')
plt.legend(loc="upper left")
plt.title("1° ATTERRRAGGIO")
plt.show()

plt.plot(x_t_3[0, 0:N_1], x_t_3[1, 0:N_1], label="x")

# %%

mse_x = mean_squared_error([6000, 6000], [x_t_4[0, 0], x_t_4[0, N_1 - 1]])
mse_y = mean_squared_error([5, 2], [x_t_4[1, 0], x_t_4[1, N_1 - 1]])
print(mse_x)
print(mse_y)
print(x_t_4[1, 0])
print(x_t_4[1, N_1 - 1])
RMSE = math.sqrt(MSE)

# %%

######
# primo decollo x-z
# xs = 0 a N_1 pari ai valori di x di  x_t_1[0,:N_1]
# ys = 0 da 0 a N_1
# zs = 0 a N_1 pari ai valori di y di x_t_1[1,:N_1]

xT = np.zeros([3, 62002])

# x_t_1 1° decollo
# x_t  1° tracking
# x_t_3 1° atterraggio
N_1 = x1.shape[1]
N_2 = x2.shape[1]
N_3 = x3.shape[1]
print(N_1 + N_2 + N_3)
print(x_t_1[1, N_1 - 1])
xT[0, 0:N_1] = x1[0, :]
xT[1, 0:N_1] = 1500
xT[2, 0:N_1] = x1[1, :]

######
# primo hovering
# xs_1 = da N_1 a N_2 valori pari a x_t[0,N_2]
# ys_1 = da N_1 a N_2 valori pari a x_t[1,N_2]
# zs_1 = 5 da N_1 a N_1+N_2
######
print(x1[0, N_1 - 1])
print(x1[1, N_1 - 1])
print(x2[0, 1])
print(x2[1, 1])

xT[0, N_1:(N_1 + N_2)] = x2[0, :]
xT[1, N_1:(N_1 + N_2)] = x2[1, :]
xT[2, N_1:(N_1 + N_2)] = 5

# primo atterraggio ù
# xs_2 = da N_2+N_1 a N_2 +N_1+N_3 pari ai valori di x di  x_t_3[0,:N_3]
# ys_2 = 0 da 0 a N_1
# zs_2 = 0 a N_1 pari ai valori di y di x_t_3[1,:N_3]
xT[0, (N_1 + N_2):(N_1 + N_2 + N_3)] = x3[0, :]
xT[1, (N_1 + N_2):(N_1 + N_2 + N_3)] = np.linspace(764, 850, N_3)
xT[2, (N_1 + N_2):(N_1 + N_2 + N_3)] = x3[1, :]

# x_t_1_1 2° decollo
# x_t_1_2  2° tracking
# x_t_4 2° atterraggio
N_4 = x4.shape[1]
N_5 = x5.shape[1]
N_6 = x6.shape[1]

print(N_1 + N_2 + N_3 + N_4 + N_5 + N_6)

xT[0, (N_1 + N_2 + N_3):(N_1 + N_2 + N_3 + N_4)] = x4[0, :]
xT[1, (N_1 + N_2 + N_3):(N_1 + N_2 + N_3 + N_4)] = 850
xT[2, (N_1 + N_2 + N_3):(N_1 + N_2 + N_3 + N_4)] = x4[1, :]

######
# primo hovering
# xs_1 = da N_1 a N_2 valori pari a x_t[0,N_2]
# ys_1 = da N_1 a N_2 valori pari a x_t[1,N_2]
# zs_1 = 5 da N_1 a N_1+N_2
######

xT[0, (N_1 + N_2 + N_3 + N_4):(N_1 + N_2 + N_3 + N_4 + N_5)] = x5[0, :]
xT[1, (N_1 + N_2 + N_3 + N_4):(N_1 + N_2 + N_3 + N_4 + N_5)] = x5[1, :]
xT[2, (N_1 + N_2 + N_3 + N_4):(N_1 + N_2 + N_3 + N_4 + N_5)] = 5

print(x5[0, N_5 - 1])
print(x5[1, N_5 - 1])
print(x6[0, 1])
print(x6[1, 1])

# primo atterraggio
# xs_2 = da N_2+N_1 a N_2 +N_1+N_3 pari ai valori di x di  x_t_3[0,:N_3]
# ys_2 = 0 da 0 a N_1
# zs_2 = 0 a N_1 pari ai valori di y di x_t_3[1,:N_3]
xT[0, (N_1 + N_2 + N_3 + N_4 + N_5):(N_1 + N_2 + N_3 + N_4 + N_5 + N_6)] = x6[0, :]
xT[1, (N_1 + N_2 + N_3 + N_4 + N_5):(N_1 + N_2 + N_3 + N_4 + N_5 + N_6)] = np.linspace(968, 1000, N_6)
xT[2, (N_1 + N_2 + N_3 + N_4 + N_5):(N_1 + N_2 + N_3 + N_4 + N_5 + N_6)] = x6[1, :]

fig = plt.figure()

xs = xT[0, :(N_1 + N_2 + N_3 + N_4 + N_5 + N_6)]
ys = xT[1, :(N_1 + N_2 + N_3 + N_4 + N_5 + N_6)]
zs = xT[2, :(N_1 + N_2 + N_3 + N_4 + N_5 + N_6)]
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot3D(xs, ys, zs, color='b', markersize=3)
ax.plot(xT[0, 0], xT[1, 0], xT[2, 0], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7, alpha=0.6)
ax.plot(xT[0, N_1], xT[1, N_1], xT[2, N_1], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7,
        alpha=0.6)
ax.plot(xT[0, (N_1 + N_2)], xT[1, (N_1 + N_2)], xT[2, (N_1 + N_2)], markerfacecolor='k', markeredgecolor='k',
        marker='x', markersize=7, alpha=0.6)
ax.plot(xT[0, (N_1 + N_2 + N_3)], xT[1, (N_1 + N_2 + N_3)], xT[2, (N_1 + N_2 + N_3)], markerfacecolor='k',
        markeredgecolor='k', marker='x', markersize=7, alpha=0.6)
ax.plot(xT[0, (N_1 + N_2 + N_3 + N_4)], xT[1, (N_1 + N_2 + N_3 + N_4)], xT[2, (N_1 + N_2 + N_3 + N_4)],
        markerfacecolor='k', markeredgecolor='k', marker='x', markersize=7, alpha=0.6)
ax.plot(xT[0, (N_1 + N_2 + N_3 + N_4 + N_5)], xT[1, (N_1 + N_2 + N_3 + N_4 + N_5)],
        xT[2, (N_1 + N_2 + N_3 + N_4 + N_5)], markerfacecolor='g', markeredgecolor='g', marker='o', markersize=7,
        alpha=0.6)
ax.plot(xT[0, (N_1 + N_2 + N_3 + N_4 + N_5 + N_6) - 1], xT[1, (N_1 + N_2 + N_3 + N_4 + N_5 + N_6) - 1],
        xT[2, (N_1 + N_2 + N_3 + N_4 + N_5 + N_6) - 1], markerfacecolor='g', markeredgecolor='g', marker='o',
        markersize=7, alpha=0.6)
fig.subplots_adjust(top=2.1, bottom=-.2)
# ax.scatter(xs,ys,zs)

ax.set_title('Quadrotor Traiectory')
ax.view_init(20, 50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# %%
