import numpy as np

class Quadrotor:

    def __init__(self, m, I, r, g, dt):
        self.m = m
        self.I = I
        self.r = r
        self.g = g
        self.dt = dt

    def next_state(self, x_state, u):
        x = x_state[0]
        y = x_state[1]
        theta = x_state[2]
        x_dot = x_state[3]  # v_x
        y_dot = x_state[4]  # v_y
        theta_dot = x_state[5]  # omega

        # integrazione del modello
        # x_dot = (A*x_error + B*x_error)
        # x_next = actual_x + dt*(A*(x_error)+B*x_error)

        x_next = (x + self.dt * x_dot)
        y_next = (y + self.dt * y_dot)
        theta_next = (theta + self.dt * theta_dot)
        x_dot_next = x_dot + self.dt * (-np.sin(theta) * (u[0] + u[1]) / self.m)
        y_dot_next = y_dot + self.dt * (-self.g + (np.cos(theta) * (u[0] + u[1]) / self.m))
        theta_dot_next = theta_dot + self.dt * ((u[0] - u[1]) / self.I) * self.r

        x_state = np.array([x_next, y_next, theta_next, x_dot_next, y_dot_next, theta_dot_next])

        return x_state

    # function:getA
    # return: matrice A linearizzata in x e u
    def getA(x, u, m, dt):
        """Matrice A linearizzata"""
        A = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, dt * (-np.cos(x[2]) * (u[0] + u[1]) / m), 1, 0, 0],
                      [0, 0, dt * (-np.sin(x[2]) * (u[0] + u[1]) / m), 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        return A

    # function:getB
    # return: matrice B linearizzata in x e u
    def getB(x, u, m, r, I, dt):
        """Matrice B linearizzata """
        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [dt * (-np.sin(x[2]) / m), dt * (-np.sin(x[2]) / m)],
                      [dt * (np.cos(x[2]) / m), dt * (np.cos(x[2]) / m)],
                      [dt * (r / I), -dt * (r / I)]])
        return B

