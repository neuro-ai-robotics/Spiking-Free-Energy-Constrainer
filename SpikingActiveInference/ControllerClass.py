import numpy as np
import control as ctrl
import nengo
rng = np.random.default_rng()

class LQR:
    def __init__(self, system):
        self.A = system.A_lin
        self.B = system.B_lin
        self.C = system.C

        u_k = system.u_k
        y_k = system.y_k
        x_k = system.x_k

        Vn_cov=0.0001
        Vd_cov=0.0001

        #Make Q and R matrices
        self.Q = 100*np.identity(x_k)
        self.Q[int(x_k/2):, int(x_k/2):] = 10 #1*np.identity(int(x_k/2))

        self.R = 0.01*np.identity(u_k)

        self.Kc, _, _ = ctrl.lqr(self.A, self.B, self.Q, self.R)

        self.Vn = Vn_cov*np.identity(y_k)    # observation covariance
        self.Vd = Vd_cov*np.identity(x_k)    # system covariance

        #Kalman filter gain matrix calculation
        Kf_t, _, _ = ctrl.lqr(self.A.T, self.C.T, self.Vd, self.Vn)
        self.Kf = Kf_t.T

        self.y = np.zeros((len(self.C),1))
        self.mu = system.x0_lin
        self.u = -self.Kc @ self.mu

    def update(self, y, target, dt = 0.001):
        if y is float:
            self.y = np.array([y])
        else: 
            self.y = y

        mu_dot = (self.A @ self.mu + self.B @ self.u) + self.Kf @ (self.y - self.C @ self.mu)

        self.mu = self.mu + mu_dot*dt

        self.u = self.Kc @ (target - self.mu)

        return self.mu, self.u, 0
    
class ActInf:
    def __init__(self, system):
        x0 = system.x0_lin

        self.u_k = system.u_k
        
        self.x_k = system.x_k
        self.z_k = system.z_k
        self.p_k = int(self.x_k/2)

        self.y_k = system.y_k + self.z_k

        if system.system == '2D_masses':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 5
            c = 10

            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims)

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

        elif system.system == 'SMD':
            dims = 1
            self.mu_k = 4

            k_ideal = 5
            c_ideal = 5

            A_ideal = np.array([[       0,       0,        0,        0],
                                [       0,       0,        0,        0], 
                                [       0,       0,        0,        1], 
                                [ k_ideal, c_ideal, -k_ideal, -c_ideal]])
            
            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A
            
        elif system.system == 'coupledSMD':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 10
            c = 15

            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims) 

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

        self.B = np.zeros((size, self.u_k))
        self.B[self.z_k:, :] = system.B_lin

        self.C = np.eye(size)

        self.Targ = np.zeros([self.mu_k, self.mu_k])
        self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
        self.Targ[:2*dims, :2*dims] = np.eye(2*dims)

        self.Py = 10
        self.Pu = 1

        self.u = np.zeros(self.u_k)

        self.mu = np.zeros(self.mu_k)

    def update(self, y, target, dt):
        y_plus = np.concatenate([target, y])

        mu_dot = self.Py*(y_plus - self.mu) + self.Pu*(self.Targ@self.mu - self.mu)
        
        self.mu = self.mu + mu_dot*dt

        self.u = self.B.T@self.A_dif@self.mu

        return self.mu, self.u, 0
    
class SCN:
    def __init__(self, system):
        self.system = system
        x0 = system.x0_lin

        self.u_k = system.u_k
        
        self.x_k = system.x_k
        self.z_k = system.z_k
        self.p_k = int(self.x_k/2)

        self.y_k = system.y_k + self.z_k

        if system.system == '2D_masses':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 10
            c = 5

            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims)

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

            self.N = 200
            D = np.zeros([size, self.N])

            for i in range(self.mu_k):
                D[i, 4*i] = 1
                D[i, 4*i+1] = -1
                D[i, 4*i+2] = 1
                D[i, 4*i+3] = -1
            for i in range(4*self.mu_k, self.N):
                D[:, i] = rng.normal(0, 1, self.mu_k)
            shuffled_indices = np.random.permutation(self.N)
            self.D = D[:, shuffled_indices]/(0.2*self.N)

            self.Targ = np.zeros([self.mu_k, self.mu_k])
            self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
            self.Targ[:2*dims, :2*dims] = np.eye(2*dims)

        elif system.system == '2D_masses_different':
            N = 3
            self.mu_k = 4 + self.x_k
            self.z_k = 4
            self.y_k = system.y_k + self.z_k
            # Make a matrix A_ideal with spring dynamics connecting each mass to the "target mass" (the zeroth mass)
            self.A_ideal = self.make_different_dynamics(N, k_goal=5, k_form=5, c=5)
            size = len(self.A_ideal)

            A_form = self.make_different_dynamics(N, k_goal=0, k_form=5, c=5)
            x_form = np.zeros(4*(N+1))
            x_form[:4] = 0
            for i in range(N):
                x_form[4+2*i] = 3*np.cos(2*np.pi*i/N)
                x_form[4+2*i+1] = 3*np.sin(2*np.pi*i/N)
                x_form[4+2*N+2*i] = 0
                x_form[4+2*N+2*i+1] = 0
            
            form_term = A_form @ x_form
            self.form_term = form_term[10:]

            self.A = np.zeros((size, size))

            self.A[4:, 4:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

            self.N = 200
            D = np.zeros([size, self.N])
            for i in range(self.mu_k):
                D[i, 4*i] = 1
                D[i, 4*i+1] = -1
                D[i, 4*i+2] = 1
                D[i, 4*i+3] = -1
            for i in range(4*self.mu_k, self.N):
                D[:, i] = rng.normal(0, 1, self.mu_k)
            shuffled_indices = np.random.permutation(self.N)
            self.D = D[:, shuffled_indices]/(0.1*self.N)

            # Make Targ matrix to compare all positions and velocities to the target mass (the zeroth mass)
            self.Targ = np.zeros([self.mu_k, self.mu_k])
            for i in range(1,4):
                self.Targ[2*i:2*i+2, :2] = np.eye(2)
                self.Targ[2*i+2*N:2*i+2*N+2, 2:4] = np.eye(2)
            self.Targ[:4, :4] = np.eye(4)

        elif system.system == 'SMD':
            dims = 1
            self.mu_k = 4

            k_ideal = 10
            c_ideal = 5

            A_ideal = np.array([[       0,       0,        0,        0],
                                [       0,       0,        0,        0], 
                                [       0,       0,        0,        1], 
                                [ k_ideal, c_ideal, -k_ideal, -c_ideal]])
            
            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

            self.N = 32
            D = np.zeros([size, self.N])
            for i in range(self.mu_k):
                D[i, 4*i] = 1
                D[i, 4*i+1] = -1
                D[i, 4*i+2] = 1
                D[i, 4*i+3] = -1
            for i in range(4*self.mu_k, self.N):
                D[:, i] = rng.normal(0, 1, self.mu_k)
            shuffled_indices = np.random.permutation(self.N)
            self.D = D[:, shuffled_indices]/(1*self.N)

            self.Targ = np.zeros([self.mu_k, self.mu_k])
            self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
            self.Targ[:2*dims, :2*dims] = np.eye(2*dims)

        elif system.system == 'coupledSMD':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 10
            c = 5
            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims) 

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

            self.N = size*10
            D = np.zeros([size, self.N])
            for i in range(5):
                D[:, size*(2*i):size*(2*i+1)] = np.eye(size)
                D[:, size*(2*i+1):size*(2*i+2)] = -np.eye(size)
            shuffled_indices = np.random.permutation(self.N)
            self.D = D[:, shuffled_indices]/(0.1*self.N)

            self.Targ = np.zeros([self.mu_k, self.mu_k])
            self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
            self.Targ[:2*dims, :2*dims] = np.eye(2*dims)

        self.B = np.zeros((size, self.u_k))
        self.B[self.z_k:, :] = system.B_lin

        self.u = np.zeros(self.u_k)

        size = len(self.A_ideal)
        self.C = np.eye(self.y_k)

        #Make H matrix
        self.eps_size = 2*self.y_k
        self.H = np.zeros((self.eps_size, self.y_k))    
        self.H[:self.y_k, :] = self.C
        self.H[:self.y_k, :] = np.eye(self.y_k) #- self.Targ

        self.S = np.zeros((self.eps_size, size))
        self.S[self.y_k:, :] = self.A_ideal # + self.Targ

        self.P = np.eye(self.eps_size)
        self.P[:self.y_k, :self.y_k] = 10*np.eye(self.y_k)

        input_matrix = np.zeros((self.eps_size, self.y_k))
        input_matrix[:self.y_k, :] = np.eye(self.y_k)

        self.O_input = self.D.T@self.H.T@self.P@input_matrix
        self.O_slow = self.D.T@self.H.T@self.P@self.S@self.D
        self.O_fast = self.D.T@self.H.T@self.P@self.H@self.D
        self.O_fast_use = self.O_fast

        self.Thr = np.diag(self.O_fast)/2
        self.Thr *= 2

        self.alive = np.ones(self.N)

        self.v = np.zeros(self.N)
        self.r = np.zeros(self.N)
        self.s = np.zeros(self.N)

        self.r_adapt = np.zeros(self.N)
        self.lambda_adapt = 0.1
        self.y_prev = np.zeros(self.y_k)

        self.mu = self.D@self.r

        self.sig2 = 0.0001 # noise variance

        self.perturb = False

        self.s_list = []
        for i in range(1):
            self.s_list.append(np.zeros(self.N))

    def make_different_dynamics(self, n, k_goal=1, k_form=0.0, c=0):
        """
        Create the dynamics matrix A for n 2D masses and one equilibrium mass.
        The equilibrium mass (first) has no dynamics; its position is set externally.
        Each mass is attracted to the equilibrium mass and to other masses by springs.
        State order: [xe, ye, vxe, vye, x1, y1, ..., xn, yn, vx1, vy1, ..., vxn, vyn]
        Returns: A (4*(n+1) x 4*(n+1)) numpy array
        """
        N = n + 1  # total masses (including equilibrium)
        size = 4 * N
        A = np.zeros((size, size))

        # For each real mass (i = 1..n)
        for i in range(1, N):
            # Indices for position and velocity in the state vector
            idx_x = 4 + 2*(i-1)      # x_i
            idx_y = 4 + 2*(i-1) + 1  # y_i
            idx_vx = 4 + 2*n + 2*(i-1)      # vx_i
            idx_vy = 4 + 2*n + 2*(i-1) + 1  # vy_i

            # Position derivatives: dx/dt = vx, dy/dt = vy
            A[idx_x, idx_vx] = 1
            A[idx_y, idx_vy] = 1

            # Velocity derivatives: dv/dt = spring forces + damping
            # Damping
            A[idx_vx, idx_vx] = -c
            A[idx_vy, idx_vy] = -c

            # Attraction to equilibrium mass (mass 0)
            A[idx_vx, 0] = k_goal      # xe
            A[idx_vy, 1] = k_goal      # ye
            A[idx_vx, idx_x] = -k_goal # x_i
            A[idx_vy, idx_y] = -k_goal # y_i

            # Formation springs: attraction to other real masses
            for j in range(1, N):
                if i == j:
                    continue
                idx_xj = 4 + 2*(j-1)
                idx_yj = 4 + 2*(j-1) + 1
                A[idx_vx, idx_xj] += k_form
                A[idx_vy, idx_yj] += k_form
                A[idx_vx, idx_x]  += -k_form
                A[idx_vy, idx_y]  += -k_form

            # Equilibrium mass (i=0) has no dynamics (rows remain zero)

        return A

    def setup(self, y0):
        self.y_prev = y0
        self.r = np.linalg.pinv(self.D)@self.y_prev
        self.mu = self.D@self.r

    def kill(self):
        self.alive = np.ones(self.N)
        for i in range(int(self.N/4)):
            self.alive[4*i] = 0

    def set_voltage_noise(self, noise):
        self.sig2 = noise

    def update(self, y_inp, target, dt):

        s_delay = self.s_list.pop(0) if len(self.s_list) > 0 else np.zeros(self.N)

        #Compute Update Terms
        if self.system.system == '2D_masses_different':
            y = np.zeros(len(y_inp)+4)
            y[:4] = target
            y[4:] = y_inp
            a = y - self.y_prev
            self.y_prev = y

        else:
            y = np.concatenate([target, y_inp])
            a = y - self.y_prev
            self.y_prev = y

        if self.perturb == True:
            factors = np.random.uniform(0.9, 1.1, [self.N, self.N])
            self.O_fast_use = self.O_fast* factors

        #Update network variables
        self.v = self.v + (self.O_input@a 
                           + dt*self.O_slow@self.r 
                           - self.O_fast_use@s_delay 
                           + self.sig2*rng.normal(0, 1, self.N)*np.sqrt(dt))

        self.v = self.v*self.alive

        self.r = self.r + s_delay

        self.s = np.zeros(len(self.s))
        Thr = self.Thr + self.r_adapt
        above = np.where(self.v > Thr)[0]
        if len(above):
            self.s = np.zeros(len(self.s))
            #self.s[rng.choice(above)] = 1
            self.s[above] = 1

        self.r_adapt = (1 - self.lambda_adapt)*self.r_adapt + self.s

        self.s_list.append(self.s)

        #Update mu and u
        self.mu = self.D@self.r

        self.u = self.B.T@self.A_dif@self.mu
        #self.u = self.B.T@self.mu
        if self.system.system == '2D_masses_different':
            self.u = self.u + self.form_term
            
        return self.mu, self.u, s_delay

class ActInf_SCN:
    def __init__(self, system):
        x0 = system.x0_lin

        self.u_k = system.u_k
        
        self.x_k = system.x_k
        self.z_k = system.z_k
        self.p_k = int(self.x_k/2)

        self.y_k = system.y_k + self.z_k

        if system.system == '2D_masses':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 10
            c = 5

            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims)

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

        elif system.system == 'SMD':
            dims = 1
            self.mu_k = 4

            k_ideal = 5
            c_ideal = 5

            A_ideal = np.array([[       0,       0,        0,        0],
                                [       0,       0,        0,        0], 
                                [       0,       0,        0,        1], 
                                [ k_ideal, c_ideal, -k_ideal, -c_ideal]])
            
            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A
            
        elif system.system == 'coupledSMD':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 10
            c = 15

            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims) 

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

        self.B = np.zeros((size, self.u_k))
        self.B[self.z_k:, :] = system.B_lin

        self.C = np.eye(size)

        self.Targ = np.zeros([self.mu_k, self.mu_k])
        self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
        self.Targ[:2*dims, :2*dims] = np.eye(2*dims)

        self.Py = 10
        self.Pu = 10

        self.u = np.zeros(self.u_k)

        self.mu = np.zeros(self.mu_k)

        self.N = 200
        self.tau = 0.1

        self.v = np.zeros(self.N)
        self.r = np.zeros(self.N)
        self.s = np.zeros(self.N)

        self.r_adapt = np.zeros(self.N)
        self.lambda_adapt = 0.1

        self.D = np.zeros((self.mu_k, self.N))
        for i in range(self.mu_k):
            self.D[i, 2*i] = 1
            self.D[i, 2*i+1] = -1
            #self.D[i, 4*i+2] = 1
            #self.D[i, 4*i+3] = -1
        for i in range(2*self.mu_k, self.N):
            self.D[:, i] = rng.normal(0, 1, self.mu_k)

        self.D = self.D/(0.1*self.N)

        self.O_mu = self.Pu*self.Targ - self.Pu*np.eye(self.mu_k) - self.Py*np.eye(self.mu_k)

        self.Omega_input = self.Py*self.D.T

        self.Omega_slow = self.D.T@(self.O_mu + self.tau*np.eye(self.mu_k))@self.D

        self.Omega_fast = -self.D.T@self.D

        self.Thr = np.diag(self.Omega_fast)/2
        #self.Thr *= 2

    def update(self, y, target, dt):
        y_plus = np.concatenate([target, y])

        self.v = (1-dt*self.tau)*self.v + dt*self.Omega_input@y_plus + dt*self.Omega_slow@self.r + self.Omega_fast@self.s

        Thr = self.Thr + self.r_adapt
        self.s = np.zeros(len(self.s))
        above = np.where(self.v > Thr)[0]
        if len(above):
            self.s[above] = 1

        self.r = (1-dt*self.tau)*self.r + self.s

        self.r_adapt = (1 - self.lambda_adapt)*self.r_adapt + self.s
        
        self.mu = self.D@self.r

        self.u = self.B.T@self.A_dif@self.mu

        return self.mu, self.u, self.s
    
class ActInf_Nengo:
    def __init__(self, system):
        self.x0 = system.x0_lin

        self.u_k = system.u_k
        
        self.x_k = system.x_k
        self.z_k = system.z_k
        self.p_k = int(self.x_k/2)

        self.y_k = system.y_k + self.z_k

        if system.system == '2D_masses':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 10 
            c = 15

            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims)

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

            self.B = np.zeros((size, self.u_k))
            self.B[self.z_k:, :] = system.B_lin

            self.C = np.eye(size)

            self.Targ = np.zeros([self.mu_k, self.mu_k])
            self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
            self.Targ[:2*dims, :2*dims] = np.eye(2*dims)

        elif system.system == 'SMD':
            dims = 1
            self.mu_k = 4

            k_ideal = 10
            c_ideal = 5

            A_ideal = np.array([[       0,       0,        0,        0],
                                [       0,       0,        0,        0], 
                                [       0,       0,        0,        1], 
                                [ k_ideal, c_ideal, -k_ideal, -c_ideal]])
            
            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

            self.B = np.zeros((size, self.u_k))
            self.B[self.z_k:, :] = system.B_lin

            self.C = np.eye(size)

            self.Targ = np.zeros([self.mu_k, self.mu_k])
            self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
            self.Targ[:2*dims, :2*dims] = np.eye(2*dims)
            
        elif system.system == 'coupledSMD':
            dims = int(self.x_k/2)
            self.mu_k = 4*dims

            k = 10
            c = 15

            A_ideal = np.zeros((self.mu_k, self.mu_k))

            A_ideal[2*dims:3*dims, 3*dims:4*dims] = np.eye(dims)

            A_ideal[3*dims:4*dims, :dims] = k*np.eye(dims)
            A_ideal[3*dims:4*dims, dims:2*dims] = c*np.eye(dims)
            A_ideal[3*dims:4*dims, 2*dims:3*dims] = -k*np.eye(dims)
            A_ideal[3*dims:4*dims, 3*dims:4*dims] = -c*np.eye(dims) 

            self.A_ideal = A_ideal
            size = len(self.A_ideal)

            self.A = np.zeros((size, size))

            self.A[2*dims:, 2*dims:] = system.A_lin
            self.A_dif = self.A_ideal - self.A

            self.B = np.zeros((size, self.u_k))
            self.B[self.z_k:, :] = system.B_lin

            self.C = np.eye(size)

            self.Targ = np.zeros([self.mu_k, self.mu_k])
            self.Targ[2*dims:, :2*dims] = np.eye(2*dims)
            self.Targ[:2*dims, :2*dims] = np.eye(2*dims)

        self.Py = 10
        self.Pu = 1

        self.Dyn = self.Pu*self.Targ -self.Py*np.eye(self.mu_k) - self.Pu*np.eye(self.mu_k)

        self.Y_in = self.Py*np.eye(self.mu_k)

        self.U = self.B.T@self.A_dif

        self.u = np.zeros(self.u_k)

        self.mu = np.zeros(self.mu_k)

        self.N = 200

        self.spike_data = np.zeros(self.N)

        self.setup_nengo()

    def setup_nengo(self):
        self.obs_y = [self.x0]  # Use mutable list to simulate reference
        self.obs_tar = [np.zeros(int(self.mu_k/2))]  # Use mutable list to simulate reference
        self.spike_data = np.zeros(self.N)

        def obs_func(t):
            return self.obs_y[0]

        def tar_func(t):
            return self.obs_tar[0]
        
        def capture_spikes(t, x):
            self.spike_data = x

        def capture_mu(t, x):
            self.mu = x

        self.model = nengo.Network(label="Controlled Oscillator")
        with self.model:
            network = nengo.Ensemble(self.N, dimensions=self.mu_k, radius=5, neuron_type=nengo.LIF(), max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.3, 0.9))

            nengo.Connection(network, network, transform=self.Dyn, synapse=1)

            y_node = nengo.Node(output=obs_func, size_out=int(self.mu_k/2))
            t_node = nengo.Node(output=tar_func, size_out=int(self.mu_k/2))

            nengo.Connection(t_node, network[:int(self.mu_k/2)], transform=self.Py*np.eye(int(self.mu_k/2)), synapse=0.001)
            nengo.Connection(y_node, network[int(self.mu_k/2):], transform=self.Py*np.eye(int(self.mu_k/2)), synapse=0.001)

            spike_node = nengo.Node(capture_spikes, size_in=self.N)
            mu_node = nengo.Node(capture_mu, size_in=self.mu_k)

            nengo.Connection(network.neurons, spike_node, synapse=None)
            nengo.Connection(network, mu_node, synapse=0.1)

            # Output probe
            #self.network_probe = nengo.Probe(network, synapse=1)

        self.sim = nengo.Simulator(self.model, dt=0.001)
        #self.data = self.sim.data
    
    def update(self, y, target, dt):
        self.obs_y[0] = y
        self.obs_tar[0] = target

        self.sim.step()

        spikes = self.spike_data
        mu_nengo = self.mu
        u_nengo = self.U@mu_nengo

        return mu_nengo, u_nengo, spikes