#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
import pygame, sys
from pygame.locals import *
from PlantClass import Plant
from ControllerClass import LQR, SCN, ActInf, ActInf_SCN, ActInf_Nengo
from DisplayClass import *

class Simulator_Animation:
    def __init__(self, T, system_type, controller_type = 'none', kill = False, noise_level = 0):
        
        os.environ['SDL_VIDEO_ACCELERATION'] = '0'
        pygame.init()
        font = pygame.font.SysFont("Verdana", 60)
        font_small = pygame.font.SysFont("Verdana", 20)
        global DISPLAYSURF
        DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        DISPLAYSURF.fill(WHITE)
        pygame.display.set_caption("Dynamical System Simulation")

        self.T = T
        self.dt = 1/FPS
        self.Nt = int(T/self.dt)
        self.time = np.arange(0, self.T, self.dt)
        self.kill = kill
        noise = 10**(noise_level) / 10000

        # System type - linear
        if system_type == '2D_masses':
            N = 4
            self.system = Plant('2D_masses', N)
            self.display = Masses_2D_Display(N)
            self.system.set_noise(noise, noise)

            self.target = np.zeros([self.Nt, len(self.system.x0)])
            for i in range(N):
                self.target[:10000, 2*i]   = np.cos(2*i*np.pi/N)
                self.target[:10000, 2*i+1] = np.sin(2*i*np.pi/N)

                self.target[10000:20000, 2*i]   = 1+ np.cos(2*i*np.pi/N)
                self.target[10000:20000, 2*i+1] = np.sin(2*i*np.pi/N)

                self.target[20000:30000, 2*i]   = np.cos(2*i*np.pi/N)
                self.target[20000:30000, 2*i+1] = 1 + np.sin(2*i*np.pi/N)

            y0 = np.concatenate([self.target[0], self.system.x0])

        elif system_type == 'SMD':
            self.system = Plant('SMD')
            self.display = SpringMassDamper_Display()
            self.system.set_noise(noise, noise)

            self.target = np.zeros([self.Nt, len(self.system.x0)])
            self.target[:10000, 0] = 5
            self.target[10000:20000, 0] = 10
            self.target[20000:30000, 0] = 5

            y0 = np.concatenate([self.target[0], self.system.x0])

        elif system_type == 'coupledSMD':
            self.system = Plant('coupledSMD')
            self.display = MassChain_Display()
            self.system.set_noise(noise, noise)

            self.target = np.zeros([self.Nt, len(self.system.x0)])
            for i in range(5):
                self.target[:, i] = self.system.x0[i]  + 0.5*np.sin(self.time)

            y0 = np.concatenate([self.target[0], self.system.x0])

        else:
            print("Invalid system")
            sys.exit()

        # Controller type
        if controller_type == 'LQR':
            self.controller = LQR(self.system)
            self.run = self.run_controller

        elif controller_type == 'SCN':
            self.controller = SCN(self.system)
            self.controller.setup(y0)
            self.run = self.run_controller
        
        elif controller_type == 'ActInf':
            self.controller = ActInf(self.system)
            self.run = self.run_controller

        elif controller_type == 'ActInf_SCN':
            self.controller = ActInf_SCN(self.system)
            self.run = self.run_controller
        
        self.controller_type = controller_type
        self.state = self.system.x0

    def run_controller(self):
        counter = 0

        u = self.controller.u

        x_list = [self.state]
        mu_list = []
        y_list = [self.system.g(self.state)]
        u_list = [u]
        while True:
            new_state, y = self.system.step(self.state, u, self.dt)

            mu, u, _ = self.controller.update(y, self.target[counter], self.dt)

            x_list.append(new_state)
            mu_list.append(mu)
            y_list.append(y)
            u_list.append(u)

            self.state = new_state

            counter += 1

            if self.kill == True:
                if counter == int(self.Nt/2):
                    self.controller.kill()

            if counter % 10 == 0:
                DISPLAYSURF.fill((255, 255, 255))

                for event in pygame.event.get(): 
                    if event.type == QUIT:
                        #self.state_list = np.array(x_list)
                        #for i in range(self.state_list.shape[1]):
                        #    plt.plot(self.state_list[:,i])
                        #plt.show()
                        u_list = np.array(u_list)
                        for i in range(u_list.shape[1]):
                            plt.plot(u_list[:,i])
                        plt.show()

                        x_list = np.array(x_list)
                        mu_list = np.array(mu_list)
                        for i in range(int(x_list.shape[1]/2)):
                            plt.plot(x_list[:,i])
                            plt.plot(mu_list[:,i], '--')
                        plt.show()

                        pygame.quit()
                        sys.exit()

                self.display.update(self.state, counter*self.dt)
                self.display.draw(DISPLAYSURF)

                pygame.display.flip()

T = 30
linear_systems = ['SMD', 'coupledSMD', '2D_masses']
simulator = Simulator_Animation(T, system_type = linear_systems[0], controller_type = 'SCN')
simulator.run()