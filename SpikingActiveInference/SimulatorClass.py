#Imports
import numpy as np
rng = np.random.default_rng()
import sys
from PlantClass import Plant
from ControllerClass import LQR, SCN, ActInf, ActInf_SCN, ActInf_Nengo
import time
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28
FPS = 1000

class Simulator_Basic:
    def __init__(self, T, system_type, controller_type = 'none'):

        self.T = T
        self.dt = 1/FPS
        self.Nt = int(T/self.dt)
        self.time = np.arange(0, self.T, self.dt)

        noise_obs  = 1/ 10000
        noise_ctrl = 1/ 10000

        if system_type == '2D_masses':
            N = 3
            self.system = Plant('2D_masses', N)
            self.system.set_noise(noise_ctrl, noise_obs)

            target = np.zeros([self.Nt, len(self.system.x0)])
            for i in range(N):
                target[:10000, 2*i]   = np.cos(2*i*np.pi/N)
                target[:10000, 2*i+1] = np.sin(2*i*np.pi/N)

                target[10000:20000, 2*i]   = 1+ np.cos(2*i*np.pi/N)
                target[10000:20000, 2*i+1] = 0.5+ np.sin(2*i*np.pi/N)

                target[20000:30000, 2*i]   = -0.5 + 0.5*np.cos(2*i*np.pi/N)
                target[20000:30000, 2*i+1] = 1 + 0.5*np.sin(2*i*np.pi/N)

            ### Convolve the target signal with a gaussian kernel to make the target trajectories smoother
            kernel_size = 1000
            kernel = np.exp(-0.5 * (np.linspace(-2, 2, kernel_size) / 0.5) ** 2)    
            self.target = np.zeros([self.Nt, len(self.system.x0)])
            for i in range(len(self.system.x0)):
                self.target[:, i] = np.convolve(target[:, i], kernel, mode='same') / np.sum(kernel)

            y0 = np.concatenate([self.target[0], self.system.x0])

        elif system_type == '2D_masses_different':
            N = 3
            self.system = Plant('2D_masses', N)
            self.system.system = '2D_masses_different'
            self.system.set_noise(noise_ctrl, noise_obs)

            self.system.x0[0] = 1
            self.system.x0[1] = 2
            self.system.x0[2] = 3
            self.system.x0[3] = 4
            self.system.x0[4] = 5
            self.system.x0[5] = 6

            self.target = np.zeros((self.Nt, 4))
            w = 3
            phi = np.pi
            self.target[:, 0] = np.sin(np.linspace(phi, w*np.pi + phi, self.Nt))
            self.target[:, 1] = np.cos(np.linspace(phi, w*np.pi + phi, self.Nt))
            self.target[:, 2] = w*np.pi*np.cos(np.linspace(phi, w*np.pi + phi, self.Nt))
            self.target[:, 3] = -w*np.pi*np.sin(np.linspace(phi, w*np.pi + phi, self.Nt))

            y0 = np.concatenate([self.target[0], self.system.x0])

        elif system_type == 'SMD':
            self.system = Plant('SMD')
            self.system.set_noise(noise_ctrl, noise_obs)

            self.target = np.zeros([self.Nt, len(self.system.x0)])
            self.target[:10000, 0] = 2
            self.target[10000:20000, 0] = 5
            self.target[20000:30000, 0] = 2

            kernel_size = 1000
            kernel = np.exp(-0.5 * (np.linspace(-2, 2, kernel_size) / 0.5) ** 2)    
            self.target[:, 0] = np.convolve(self.target[:, 0], kernel, mode='same') / np.sum(kernel)
            self.target[:, 1] = np.convolve(self.target[:, 1], kernel, mode='same') / np.sum(kernel)

            y0 = np.concatenate([self.target[0], self.system.x0])

        elif system_type == 'coupledSMD':
            self.system = Plant('coupledSMD')
            self.system.set_noise(noise_ctrl, noise_obs)

            dims = int(len(self.system.x0)/2)
            self.target = np.zeros([self.Nt, len(self.system.x0)])
            self.target[:10000, :dims-1] = self.system.x0[:dims-1] + 0.5
            self.target[10000:20000, :dims-1] = self.system.x0[:dims-1] - 1
            self.target[20000:30000, :dims-1] = self.system.x0[:dims-1]
            self.target[:, dims-1] = self.system.x0[dims-1]

            kernel_size = 1000
            kernel = np.exp(-0.5 * (np.linspace(-2, 2, kernel_size) / 0.5) ** 2)    
            for i in range(dims):
                self.target[:, i] = np.convolve(self.target[:, i], kernel, mode='same') / np.sum(kernel)
                self.target[:, i + dims] = np.convolve(self.target[:, i + dims], kernel, mode='same') / np.sum(kernel)

            y0 = np.concatenate([self.target[0], self.system.x0])

        else:
            print("Invalid system")
            sys.exit()

        # Controller type
        if controller_type == 'LQR':
            self.controller = LQR(self.system)
            self.run = self.run_controller

        elif controller_type == 'ActInf':
            self.controller = ActInf(self.system)
            self.run = self.run_controller

        elif controller_type == 'SCN':
            self.controller = SCN(self.system)
            self.controller.setup(y0)
            self.run = self.run_controller

        elif controller_type == 'ActInf_SCN':
            self.controller = ActInf_SCN(self.system)
            self.run = self.run_controller

        elif controller_type == 'ActInf_Nengo':
            self.controller = ActInf_Nengo(self.system)
            self.run = self.run_controller
        
        self.controller_type = controller_type
        self.state = self.system.x0

    def run_controller(self):
        """Run the simulation with the controller for plotting purposes only.
        No displays are shown."""
        
        counter = 0

        u = self.controller.u

        x_list = [self.state]
        mu_list = []
        y_list = [self.system.g(self.state)]
        u_list = [u]
        F_list = []
        eps_list = []
        Times = []

        if self.controller_type == 'SCN' or self.controller_type == 'ActInf_SCN' or self.controller_type == 'ActInf_Nengo':
            spikes_vec = np.zeros([self.controller.N, self.Nt])

        while True:
            new_state, y = self.system.step(self.state, u, self.dt)

            start_time = time.perf_counter()
            mu, u, spikes = self.controller.update(y, self.target[counter], self.dt)
            elapsed_time = time.perf_counter() - start_time
            Times.append(elapsed_time)

            x_list.append(new_state)
            mu_list.append(mu)
            y_list.append(y)

            eps_y = y - self.system.C @ mu[int(self.controller.z_k):]
            eps_mu = self.controller.Targ@mu - mu
            eps = np.concatenate([eps_y, eps_mu])
            eps_list.append(eps)
            F = eps.T @ eps
            F_list.append(F)

            u_list.append(u)

            if self.controller_type == 'SCN' or self.controller_type == 'ActInf_SCN' or self.controller_type == 'ActInf_Nengo':
                spikes_vec[:, counter] = spikes

            self.state = new_state
            
            counter += 1

            if counter == self.Nt:
                break

        output_list = (np.array(x_list), np.array(u_list), np.array(mu_list), np.array(y_list), self.target, np.array(eps_list), np.array(F_list), self.time, np.array(spikes_vec), np.array(Times))
        return output_list


class Simulator_Compare_Internal:
    def __init__(self, T):

        self.T = T
        self.dt = 1/FPS
        self.Nt = int(T/self.dt)
        self.time = np.arange(0, self.T, self.dt)

        self.noise_0 = 1/10000

        self.noise_obs  = 1/100
        self.noise_ctrl = 1/100


        self.setup_2D_masses()
        self.system.set_noise(self.noise_0, self.noise_0)

    def setup_2D_masses(self):
        N = 3
        self.system = Plant('2D_masses', N)

        self.target = np.zeros([self.Nt, len(self.system.x0)])
        for i in range(N):
            self.target[:, 2*i]   = np.cos(2*i*np.pi/N)+np.sin(self.time[:self.Nt])
            self.target[:, 2*i+1] = np.sin(2*i*np.pi/N)+np.sin(self.time[:self.Nt])

            self.target[:, 6+2*i]   = np.cos(self.time[:self.Nt])
            self.target[:, 6+2*i+1] = np.cos(self.time[:self.Nt])

        self.y0 = np.concatenate([self.target[0], self.system.x0])

        self.controller = SCN(self.system)
        self.controller.setup(self.y0)
        self.run = self.run_controller
        self.controller_type = 'SCN'
    
        self.state = self.system.x0

    def run_controller(self):
        """Run the simulation with the controller for plotting purposes only.
        No displays are shown."""
        
        counter = 0

        u = self.controller.u

        x_list = [self.state]
        mu_list = []
        target_list = []
        y_list = [self.system.g(self.state)]
        u_list = [u]
        eps_list = []
        F_list = []
        Times = []
        spikes_vec = np.zeros([self.controller.N, self.Nt])

        while True:
            new_state, y = self.system.step(self.state, u, self.dt)

            start_time = time.perf_counter()
            mu, u, spikes = self.controller.update(y, self.target[counter], self.dt)
            elapsed_time = time.perf_counter() - start_time
            Times.append(elapsed_time)

            eps_y = y - self.system.C @ mu[int(len(mu)/2):]
            eps_mu = self.target[counter] - mu[int(len(mu)/2):]
            eps = np.concatenate([eps_y, eps_mu])
            F = eps.T @ eps

            x_list.append(new_state)
            mu_list.append(mu)
            target_list.append(self.target[counter])
            y_list.append(y)
            u_list.append(u)
            eps_list.append(eps)
            F_list.append(F)

            if self.controller_type == 'SCN':
                spikes_vec[:, counter] = spikes

            self.state = new_state

            counter += 1

            if counter == 6000:
                self.controller.set_voltage_noise(0.001)
                pass

            if counter == 12000:
                self.controller.perturb = True
                pass

            if counter == 18000:
                self.controller.kill()
                pass
            
            if counter == 24000:            
                for i in range(4):
                    self.controller.s_list.append(np.zeros(self.controller.N))
                pass

            if counter == self.Nt:
                break

        output_list = (np.array(x_list), np.array(u_list), np.array(mu_list), np.array(y_list), self.target, np.array(eps_list), np.array(F_list), self.time, np.array(spikes_vec), np.array(Times))
        return output_list

class Simulator_Compare_External:
    def __init__(self, T):

        self.T = T
        self.dt = 1/FPS
        self.Nt = int(T/self.dt)
        self.time = np.arange(0, self.T, self.dt)

        self.noise_0 = 1/10000

        self.noise_obs  = 1/100
        self.noise_ctrl = 1/100

        self.setup_2D_masses()
        self.system.set_noise(self.noise_0, self.noise_0)

    def setup_2D_masses(self):
        N = 3
        self.system = Plant('2D_masses', N)

        self.target = np.zeros([self.Nt, len(self.system.x0)])
        for i in range(N):
            self.target[:, 2*i]   = np.cos(2*i*np.pi/N)+np.sin(self.time[:self.Nt])
            self.target[:, 2*i+1] = np.sin(2*i*np.pi/N)+np.sin(self.time[:self.Nt])

            self.target[:, 6+2*i]   = np.cos(self.time[:self.Nt])
            self.target[:, 6+2*i+1] = np.cos(self.time[:self.Nt])

        self.y0 = np.concatenate([self.target[0], self.system.x0])

        self.controller = SCN(self.system)
        self.controller.setup(self.y0)
        self.run = self.run_controller
        self.controller_type = 'SCN'
    
        self.state = self.system.x0

        self.u_kick = np.zeros([self.Nt, self.system.u_k])
        kick_indices = rng.integers(0, self.system.u_k, 6)
        self.kick_times = np.array([1000, 5000, 9000, 13000, 17000, 21000])
        #print(self.kick_times)
        for i in range(6):
            self.u_kick[self.kick_times[i]:(self.kick_times[i]+100), kick_indices[i]] = 10

    def run_controller(self):
        """Run the simulation with the controller for plotting purposes only.
        No displays are shown."""
        
        counter = 0

        u = self.controller.u

        x_list = [self.state]
        mu_list = []
        target_list = []
        y_list = [self.system.g(self.state)]
        u_list = [u]
        eps_list = []
        F_list = []
        Times = []
        spikes_vec = np.zeros([self.controller.N, self.Nt])

        while True:
            new_state, y = self.system.step(self.state, u + self.u_kick[counter], self.dt)

            start_time = time.perf_counter()
            mu, u, spikes = self.controller.update(y, self.target[counter], self.dt)
            elapsed_time = time.perf_counter() - start_time
            Times.append(elapsed_time)

            eps_y = y - self.system.C @ mu[int(len(mu)/2):]
            eps_mu = self.target[counter] - mu[int(len(mu)/2):]
            eps = np.concatenate([eps_y, eps_mu])
            F = eps.T @ eps

            x_list.append(new_state)
            mu_list.append(mu)
            target_list.append(self.target[counter])
            y_list.append(y)
            u_list.append(u)
            eps_list.append(eps)
            F_list.append(F)

            if self.controller_type == 'SCN':
                spikes_vec[:, counter] = spikes

            self.state = new_state

            counter += 1

            if counter == 10000:
                self.system.set_noise(0, self.noise_ctrl)
                pass

            if counter == 20000:
                self.system.set_noise(self.noise_obs, self.noise_ctrl)
                pass

            if counter == self.Nt:
                break

        output_list = (np.array(x_list), np.array(u_list), np.array(mu_list), np.array(y_list), self.target, np.array(eps_list), np.array(F_list), self.time, np.array(spikes_vec), np.array(Times))
        return output_list
        
class Simulator_Compare_Noise:
    def __init__(self, T):

        self.T = T
        self.dt = 1/FPS
        self.Nt = int(T/self.dt)
        self.time = np.arange(0, self.T, self.dt)

        self.noise_0 = 1/10000

        self.setup_2D_masses()
        self.system.set_noise(self.noise_0, self.noise_0)

    def setup_2D_masses(self):
        N = 3
        self.system = Plant('2D_masses', N)

        self.target = np.zeros([self.Nt, len(self.system.x0)])
        for i in range(N):
            self.target[:, 2*i]   = np.cos(2*i*np.pi/N)+np.sin(self.time[:self.Nt])
            self.target[:, 2*i+1] = np.sin(2*i*np.pi/N)+np.sin(self.time[:self.Nt])

            self.target[:, 6+2*i]   = np.cos(self.time[:self.Nt])
            self.target[:, 6+2*i+1] = np.cos(self.time[:self.Nt])

        self.y0 = np.concatenate([self.target[0], self.system.x0])

        self.controller = SCN(self.system)
        self.controller.setup(self.y0)
        self.run = self.run_controller
        self.controller_type = 'SCN'
    
        self.state = self.system.x0

    def run_controller(self, noise_ctrl=0, noise_obs=0):
        """Run the simulation with the controller for plotting purposes only.
        No displays are shown."""

        self.system.set_noise(noise_obs, noise_ctrl)
        
        counter = 0

        u = self.controller.u

        x_list = [self.state]
        mu_list = []
        target_list = [self.target[counter]]
        y_list = [self.system.g(self.state)]
        u_list = [u]
        F_list = []
        Times = []
        spikes_vec = np.zeros([self.controller.N, self.Nt])

        while True:
            new_state, y = self.system.step(self.state, u, self.dt)

            start_time = time.perf_counter()
            mu, u, spikes = self.controller.update(y, self.target[counter], self.dt)
            elapsed_time = time.perf_counter() - start_time
            Times.append(elapsed_time)

            eps_y = y - self.system.C @ mu[int(len(mu)/2):]
            eps_mu = self.target[counter] - mu[int(len(mu)/2):]
            eps = np.concatenate([eps_y, eps_mu])
            F = eps.T @ eps

            x_list.append(new_state)
            mu_list.append(mu)
            target_list.append(self.target[counter])
            y_list.append(y)
            u_list.append(u)
            F_list.append(F)

            if self.controller_type == 'SCN':
                spikes_vec[:, counter] = spikes

            self.state = new_state

            counter += 1

            if counter >= self.Nt:
                break
        
        output_list = (np.array(x_list), np.array(u_list), np.array(mu_list), np.array(y_list), self.target, np.array(F_list), self.time, np.array(spikes_vec), np.array(Times))
        return output_list
        
    def run_compare(self):
        noise_ctrls = np.array([0.01, 0.03, 0.1, 0.3, 1,  3,  10,  30,  100,  300,  1000, 3000])/10000
        noise_obs =   np.array([0.1,  0.3,  1,   3,   10, 30, 100, 300, 1000, 3000, 10000, 30000])/10000
        MSE_heatmap = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                values = self.run_controller(noise_ctrl=noise_ctrls[i], noise_obs=noise_obs[j])
                x = values[0]
                t = values[4]
                error = x[:self.Nt] - t[:self.Nt]
                error_norm = np.linalg.norm(error, axis=1)
                MSE = np.mean(error_norm, axis=0)
                MSE_heatmap[i, j] = MSE
            print('---- ', 12*i+j+1, '/144 completed')

        return MSE_heatmap, np.array(noise_ctrls), np.array(noise_obs)