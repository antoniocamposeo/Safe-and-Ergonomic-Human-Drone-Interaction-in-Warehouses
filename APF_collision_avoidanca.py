import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage.morphology import distance_transform_edt as bwdist


def repulsive(obstacle):
    #d = bwdist(obstacle==0) #distance from obstacle
    # Rescale and transform distances
#     d2 = (d/40.) + 1
#     d0 = 2 #eta_0
#     nu = 450
    d = bwdist(obstacle==0);

    # Rescale and transform distances

    d2 = (d/20.) + 1;

    d0 = 2;
    nu = 200;

#     repulsive = nu*((1./d2 - 1/d0)**2);

#     repulsive [d2 > d0] = 0;

    repulsive = nu*((1./d2 - 1/d0)**2)

    repulsive[d2 > d0] = 0
    return repulsive

def attractice(x,y,goal):
    
    #xi = 1 / 300.  # k/2
    xi = 1 / 1000.
    attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 ); #Parabolic potential
    return attractive


def GradientBasedPlanner (f, start_coords, end_coords, max_its):
    # GradientBasedPlanner : This function plans a path through a 2D
    # environment from a start to a destination based on the gradient of the
    # function f which is passed in as a 2D array. The two arguments
    # start_coords and end_coords denote the coordinates of the start and end
    # positions respectively in the array while max_its indicates an upper
    # bound on the number of iterations that the system can use before giving
    # up.
    # The output, route, is an array with 2 columns and n rows where the rows
    # correspond to the coordinates of the robot as it moves along the route.
    # The first column corresponds to the x coordinate and the second to the y coordinate

    [gy, gx] = np.gradient(-f);

    route = np.vstack( [np.array(start_coords), np.array(start_coords)] )
    for i in range(max_its):
        current_point = route[-1,:];
#         print(sum( abs(current_point-end_coords) ))
        if sum( abs(current_point-end_coords) ) < 5.0:
            print('Reached the goal !');
            break
        ix = int(round( current_point[1],3));
        iy = int(round( current_point[0],3 ));
        vx = gx[ix, iy]
        vy = gy[ix, iy]
        dt = 1 / np.linalg.norm([vx, vy]);
        next_point = current_point + dt*np.array( [vx, vy] );
        route = np.vstack( [route, next_point] );
    route = route[1:,:]

    return route


def gradient_plot(x,y, gx,gy, skip=10):
    plt.figure(figsize=(20,8))
    Q = plt.quiver(x[::skip, ::skip], y[::skip, ::skip], gx[::skip, ::skip], gy[::skip, ::skip],
                   pivot='mid', units='inches',alpha=0.6)
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure',alpha=0.5)
    #plt.scatter(x[::skip, ::skip], y[::skip, ::skip], color='g', s=2)


def create_obtacle(nrows,ncols):
    obstacle = np.zeros((nrows, ncols))
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))

    # Generate some obstacle
    obstacle[0:, 0:10] = True;
    obstacle[989:, 1:] = True;
    obstacle[0:, 990:] = True;
    obstacle[0:10, 10:] = True;

    obstacle[100:300, 50:80] = True;
    obstacle[500:800, 50:80] = True;
    obstacle[100:300, 140:170] = True;
    obstacle[500:800, 140:170] = True;
    obstacle[100:300, 230:260] = True;
    obstacle[500:800, 230:260] = True;

    obstacle[100:300, 360:390] = True;
    obstacle[500:800, 360:390] = True;
    obstacle[100:300, 450:480] = True;
    obstacle[500:800, 450:480] = True;

    obstacle[100:300, 540:570] = True;
    obstacle[500:800, 540:570] = True;

    obstacle[100:300, 670:700] = True;
    obstacle[500:800, 670:700] = True;
    obstacle[100:300, 760:790] = True;
    obstacle[500:800, 760:790] = True;
    obstacle[100:300, 850:880] = True;
    obstacle[500:800, 850:880] = True;
    obstacle[100:300, 940:970] = True;
    obstacle[500:800, 940:970] = True;

    obstacle[870:900, 50:260] = True;
    obstacle[870:900, 360:570] = True;
    obstacle[870:900, 670:970] = True;

    return obstacle,x,y


def Split_BacktrackStates(backtrackStates):
    x_2 = []
    for i in range(2, len(backtrackStates), 20):
        k = int(backtrackStates[i][0])
        k1 = int(backtrackStates[i][1])
        x_2.append((k, k1))
    A = np.array(x_2)
    x__1 = A[:, 0]
    x__2 = A[:, 1]
    return x__1, x__2

def Interpolation(x_1,z_1,grado):
    x = np.array(x_1)
    z = np.array(z_1)
    param = np.linspace(0, 1, x.size)
    spl = make_interp_spline(param, np.c_[x, z], k=grado)  # (1)
    xnew, y_smooth = spl(np.linspace(0, 1, x.size * 19)).T  # (2)
    plt.plot(xnew, y_smooth, 'g')
    # plt.show()
    # plt.scatter(x, z)
    plt.title(f"INTERPOLAZIONE GRADO{grado}")
    plt.show()

    x_new = [int(x) for x in xnew]
    y_new = [int(y) for y in y_smooth]
    path_new = list(zip(x_new, y_new))

    return path_new
