import numpy as np
from Quadrotor import *


def LQR_2EQ(N, x_0, x_bar, u_0, u_bar, quadrotor, R, Q):
    K_1 = np.empty([6, N - 1])
    K_2 = np.empty([6, N - 1])
    # get A and B
    A = Quadrotor.getA(x_bar, u_bar, quadrotor.m, quadrotor.dt)
    B = Quadrotor.getB(x_bar, u_bar, quadrotor.m, quadrotor.r, quadrotor.I, quadrotor.dt)
    P = Q.copy()

    for i in range(0, N - 1):
        K = -np.dot((np.dot((np.dot((np.linalg.pinv(np.dot((np.dot(B.T, P)), B) + R)), B.T)), P)), A)
        P = Q + np.dot((np.dot(A.T, P)), A) + np.dot((np.dot((np.dot(A.T, P)), B)), K)
        K_12 = K.T.copy()
        K_1[:, N - i - 2] = (K_12[:, 0]).copy()
        K_2[:, N - i - 2] = (K_12[:, 1]).copy()

    empty_mat = np.empty([6, N])
    x_1 = np.empty([6, N])
    x_1[:, 0] = x_0
    u_opt = np.zeros([2, N])
    u_opt[:, 0] = u_0.copy()

    # integrazione al primo step
    empty_mat[:, 0] = Quadrotor.next_state(quadrotor, x_1[:, 0], u_opt[:, 0])
    x_1[:, 1] = empty_mat[:, 0]
    K_tot = np.concatenate((K_1[:, 0], K_2[:, 0]), axis=0)
    K_tot = K_tot.reshape((2, 6))
    u_opt[:, 1] = np.dot(K_tot, (x_1[:, 1] - x_bar)) + u_bar  # u ottimale

    for i in range(1, N - 1):
        empty_mat[:, i] = Quadrotor.next_state(quadrotor, x_1[:, i], u_opt[:, i])
        x_1[:, i + 1] = empty_mat[:, i]
        K_tot = np.concatenate((K_1[:, i], K_2[:, i]), axis=0)
        K_tot = K_tot.reshape((2, 6))
        u_opt[:, i + 1] = np.dot(K_tot, (x_1[:, i + 1] - x_bar)) + u_bar

    return x_1, u_opt


def LQR(N, x_0, x_bar, u_0, u_bar, quadrotor, R, Q):
    # Q = np.identity(6) * 1
    # Q[0, 0] = 1000
    # Q[1, 1] = 100
    # Q[2, 2] = 10
    # R= np.identity(2) *1000

    P = Q.copy()
    K_1 = np.empty([6, N - 1])
    K_2 = np.empty([6, N - 1])
    k_t_1 = np.empty([N - 1])
    k_t_2 = np.empty([N - 1])
    q = -np.dot(Q, x_bar[:, N - 1])
    p = q.copy()

    # linearizzazione ad ogni step
    # calcolando i valori di K e k ottimali ad ogni step
    for s in range(0, N - 1):
        if s == 0:
            A = Quadrotor.getA(x_0, u_0, quadrotor.m, quadrotor.dt)
            B = Quadrotor.getB(x_0, u_0, quadrotor.m, quadrotor.r, quadrotor.I, quadrotor.dt)
        A = Quadrotor.getA(x_bar[:, s], u_bar[:, s], quadrotor.m, quadrotor.dt)
        B = Quadrotor.getB(x_bar[:, s], u_bar[:, s], quadrotor.m, quadrotor.r, quadrotor.I, quadrotor.dt)
        K = -np.dot((np.dot((np.dot((np.linalg.inv(np.dot((np.dot(B.T, P)), B) + R)), B.T)), P)), A)
        P = Q + np.dot((np.dot(A.T, P)), A) + np.dot((np.dot((np.dot(A.T, P)), B)), K)
        r = np.dot(R, u_bar[:, N - s - 2])
        k_t = - np.dot(np.linalg.inv(R - np.dot(B.T, np.dot(P, B))), (np.dot(B.T, p) + r))
        p = q + np.dot(A.T, p) + np.dot((np.dot((np.dot(A.T, P)), B)), k_t)

        K_12 = K.T.copy()
        K_1[:, N - s - 2] = (K_12[:, 0]).copy()
        K_2[:, N - s - 2] = (K_12[:, 1]).copy()
        k_t_1[N - s - 2] = (k_t[0]).copy()
        k_t_2[N - s - 2] = (k_t[1]).copy()

    x_t = np.empty([6, N])
    x_t[:, 0] = x_0
    u_opt = np.zeros([2, N - 1])

    for i in range(0, N - 1):
        K_tot = np.concatenate((K_1[:, i], K_2[:, i]), axis=0)
        K_tot = K_tot.reshape((2, 6))
        k_tot = np.array([k_t_1[i], k_t_2[i]])
        k_tot = k_tot.reshape((1, 2))
        u_opt[:, i] = np.dot(K_tot, (x_t[:, i] - x_bar[:, i])) + u_bar[:, i] + k_tot
        x_t[:, i + 1] = Quadrotor.next_state(quadrotor, x_t[:, i], u_opt[:, i])

    return x_t, u_opt


def iLQR(N, x_0, x_bar, u_0, u_bar, R, Q, quadrotor):
    A = Quadrotor.getA(x_0, u_0, quadrotor.m, quadrotor.dt)
    B = Quadrotor.getB(x_0, u_0, quadrotor.m, quadrotor.r, quadrotor.I, quadrotor.dt)
    P = Q.copy()
    K_1 = np.empty([6, N - 1])
    K_2 = np.empty([6, N - 1])
    k_t_1 = np.empty([N - 1])
    k_t_2 = np.empty([N - 1])
    q = -np.dot(Q, x_bar[:, N - 1])
    p = q.copy()

    # ricavo qundi u_guess che sarebbe il controllo ottimale
    # x_guess un approssimazione dei valori di x_bar

    for s in range(0, N - 1):
        K = -np.dot((np.dot((np.dot((np.linalg.inv(np.dot((np.dot(B.T, P)), B) + R)), B.T)), P)), A)
        r = -np.dot(R, u_bar[:, N - s - 2])
        k = - np.dot(np.linalg.inv(R - np.dot(B.T, np.dot(P, B))), (np.dot(B.T, p) + r))
        p = q + np.dot(A.T, p) + np.dot((np.dot((np.dot(A.T, P)), B)), k)
        P = Q + np.dot((np.dot(A.T, P)), A) + np.dot((np.dot((np.dot(A.T, P)), B)), K)
        q = -np.dot(Q, x_bar[:, N - s - 2])

        K_12 = K.T.copy()
        K_1[:, N - s - 2] = (K_12[:, 0]).copy()
        K_2[:, N - s - 2] = (K_12[:, 1]).copy()
        k_t_1[N - s - 2] = (k[0]).copy()
        k_t_2[N - s - 2] = (k[1]).copy()

    x_g = np.empty([6, N])
    x_g[:, 0] = x_0
    u_g = np.zeros([2, N - 1])
    for i in range(0, N - 1):
        K_tot = np.concatenate((K_1[:, i], K_2[:, i]), axis=0)
        K_tot = K_tot.reshape((2, 6))
        k_tot = np.array([k_t_1[i], k_t_2[i]])
        k_tot = k_tot.reshape((1, 2))
        u_g[:, i] = np.dot(K_tot, (x_g[:, i] - x_bar[:, i])) + u_bar[:, i] + k_tot
        x_g[:, i + 1] = Quadrotor.next_state(quadrotor, x_g[:, i], u_g[:, i])

    N_iter = 50

    cost_total = np.empty([1, N_iter])
    alpha = 1
    epsilon = 10000

    for t in range(0, N_iter):
        Q = np.identity(6)*0.1
        # Q[0, 0] = 20
        # Q[1, 1] = 20
        # Q[2, 2] = 20
        # Q[3, 3] = 20
        # Q[4, 4] = 20
        # Q[5, 5] = 20
        # matrice R 2x2 -- Input cost matrix
        R = np.identity(2) * 2000
        # R[0, 0] = 600
        # R[1, 1] = 600
        P = Q.copy()

        K_1 = np.empty([6, N - 1])
        K_2 = np.empty([6, N - 1])
        k_t_1 = np.empty([N - 1])
        k_t_2 = np.empty([N - 1])
        q = -np.dot(Q, x_bar[:, N - 1])
        p = q.copy()

        for s in range(0, N - 1):
            A = Quadrotor.getA(x_g[:, s], u_g[:, s], quadrotor.m, quadrotor.dt)
            B = Quadrotor.getB(x_g[:, s], u_g[:, s], quadrotor.m, quadrotor.r, quadrotor.I, quadrotor.dt)
            K = -np.dot((np.dot((np.dot((np.linalg.inv(np.dot((np.dot(B.T, P)), B) + R)), B.T)), P)), A)
            r = -np.dot(R, u_bar[:, N - s - 2])
            k = - np.dot(np.linalg.inv(R - np.dot(B.T, np.dot(P, B))), (np.dot(B.T, p) + r))
            p = q + np.dot(A.T, p) + np.dot((np.dot((np.dot(A.T, P)), B)), k)
            P = Q + np.dot((np.dot(A.T, P)), A) + np.dot((np.dot((np.dot(A.T, P)), B)), K)
            q = -np.dot(Q, x_bar[:, N - s - 2])
            K_12 = K.T.copy()
            K_1[:, N - s - 2] = (K_12[:, 0]).copy()
            K_2[:, N - s - 2] = (K_12[:, 1]).copy()
            k_t_1[N - s - 2] = (k[0]).copy()
            k_t_2[N - s - 2] = (k[1]).copy()

        x_f = np.empty([6, N])
        x_f[:, 0] = x_0
        u_opt = np.zeros([2, N - 1])

        for i in range(0, N - 1):
            K_tot = np.concatenate((K_1[:, i], K_2[:, i]), axis=0)
            K_tot = K_tot.reshape((2, 6))
            k_tot = np.array([k_t_1[i], k_t_2[i]])
            k_tot = k_tot.reshape((1, 2))
            u_opt[:, i] = np.dot(K_tot, (x_f[:, i] - x_bar[:, i])) + u_bar[:, i] + alpha * k_tot
            x_f[:, i + 1] = Quadrotor.next_state(quadrotor, x_f[:, i], u_opt[:, i])

        x_t = x_f - x_bar

        u_t = u_opt - u_bar
        cost = 0
        for j in range(0, N - 1):
            cost = cost + np.dot((np.dot(x_t[:, j].T, Q)), x_t[:, j]) + np.dot((np.dot(u_t[:, j].T, R)), u_t[:, j])

        cost = cost + np.dot((np.dot(x_t[:, N - 1].T, Q)), x_t[:, N - 1])

        cost_total[0, t] = cost

        if (abs(cost_total[0, (t - 1)]) - cost_total[0, t]) < epsilon:
            alpha = alpha * 0.9
        u_g = u_opt.copy()
        x_t = x_f.copy()

    return x_t, u_opt
