import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()


plt.rcParams.update({'font.size': 30})

SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

class Plots:
    def __init__(self):
        pass

    def make_Fig2(self, values, title = 'Fig2'):
        x, u, mu, y, target, eps, F, time, spikes_vec, Times = values

        plt.figure()
        plt.clf()
        ### Image size
        plt.figure(figsize=(20, 12))
        #plt.rc('axes', prop_cycle=line_cycler)
        dims = int(len(x[0])/2)

        plt.subplot(311)
        target_lines = plt.plot(time, target[:, :dims], '--', color='grey')
        state_lines = plt.plot(time, x[:len(time), :dims], color='red')
        plt.legend(handles=[target_lines[0],  # Use the first line from the target group
                            state_lines[0],   # Use the first line from the state group
                            ], labels=['Target (dashed)', 'State (solid)'], loc='upper left')
        plt.title('States', fontsize=BIGGER_SIZE)
        plt.ylabel('States')
        plt.xticks([])
        #plt.xlabel('Time (s)')

        plt.subplot(312)
        plt.plot(time, u[:len(time), :dims], color='black')
        plt.ylabel('Control Inputs')
        plt.xticks([])

        a=np.where(spikes_vec[:, :])
        plt.subplot(313)
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron Index')
        plt.scatter(time[a[1]],a[0], marker='.',s=1,color='k',alpha=0.7,rasterized=True)
        plt.xticks([0, 10, 20, 30])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=300) #, bbox_inches='tight')
        plt.close()

    def make_Fig5(self, values, title = 'Fig5'):
        x, u, mu, y, target, eps, F, time, spikes_vec, Times = values
        dims1 = int(len(x[0])/2)

        plt.figure(figsize=(20, 12))
        plt.subplot(4, 1, 1)
        state_lines = plt.plot(time, x[:len(time), :dims1], color='red')
        target_lines = plt.plot(time, target[:len(time), :dims1], '--', color='grey')
        plt.legend(handles=[target_lines[0],state_lines[0]], 
                   labels=['Target (dashed)', 'State (solid)'], 
                   loc='upper left')
        
        plt.xlim(0, 30)
        #plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.title('Drones', fontsize=BIGGER_SIZE)
        plt.xticks([])

        plt.subplot(4, 1, 2)
        a=np.where(spikes_vec[:, :])
        plt.scatter(time[a[1]],a[0], marker='.',s=1,color='k',alpha=0.7,rasterized=True)
        plt.xlim(0, 30)
        plt.ylabel('Neuron Index')
        #plt.xlabel('Time (s)')
        plt.xticks([])

        plt.subplot(4, 1, 3)
        plt.plot(time, F[:len(time)], color='black')
        #plt.vlines([10, 20], 0, np.max(F1), color='r', linestyle='--')
        #plt.vlines(self.kick_times * self.dt, np.min(F1), np.max(F1), color='g', linestyle='--')
        plt.xlim(0, 30)
        plt.xlabel('Time (s)')
        plt.ylabel('Free Energy')
        plt.xticks([0, 10, 20, 30])

        # Calculate MSE between state and target
        MSE = np.mean((x[:len(time), :dims1] - target[:len(time), :dims1])**2, axis=1)
        plt.subplot(4, 1, 4)
        plt.plot(time, MSE)
        plt.xlim(0, 30)
        plt.xlabel('Time (s)')
        plt.ylabel('MSE')
        plt.ylim(0, 0.2)
        plt.xticks([0, 10, 20, 30])

        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=300)  # Save the figure with high resolution
        plt.close()

    def make_Fig8(self, values, title = 'Fig8'):
        x, u, mu, y, target, eps, F, time, spikes_vec, Times = values
        dims = 6

        plt.figure(figsize=(24, 10))
        plt.subplot(3, 1, 1)
        #Show the x lines with even indecies
        plt.plot(time, target[:len(time), 0], '--', color='blue', label ='Target x')
        plt.plot(time, target[:len(time), 1], '--', color='green', label='Target y')
        plt.plot(time, mu[:len(time), 4], color='blue', alpha=0.5, label='x')
        plt.plot(time, mu[:len(time), 5], color='green', alpha=0.5, label='y')
        for i in range(1, 3):
            plt.plot(time, mu[:len(time), 4+2*i], color='blue', alpha=0.5)
            plt.plot(time, mu[:len(time), 4+2*i+1], color='green', alpha=0.5)
        plt.legend(loc = 'upper right', fontsize=12)
        plt.xticks([])
        plt.ylabel('Position')
        plt.title('2D Masses with Different Dynamics', fontsize=BIGGER_SIZE)

        plt.subplot(3, 1, 2)
        a=np.where(spikes_vec[:, :])
        plt.scatter(time[a[1]],a[0], marker='.',s=1,color='k',alpha=0.7,rasterized=True)
        plt.xlim(0, 30)
        plt.ylabel('Spikes')
        plt.xticks([])
        #for i in range(3):
        #    plt.plot(time, u[:len(time), 2*i], color='blue', alpha=0.5)
        #    plt.plot(time, u[:len(time), 2*i+1], color='green', alpha=0.5)
        #plt.ylabel('Control Inputs')

        plt.subplot(3, 1, 3)
        plt.plot(time, F[:len(time)], color='black')
        plt.xlim(0, 30)
        plt.xticks([0, 10, 20, 30])
        plt.xlabel('Time (s)')
        plt.ylabel('Free Energy')

        #a=np.where(spikes_vec[:, :])
        #plt.scatter(time[a[1]],a[0], marker='.',s=1,color='k',alpha=0.7,rasterized=True)
        #plt.xlim(0, 30)
        #plt.xlabel('Time (s)')
        #plt.ylabel('Spikes')

        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=300)
        plt.close()

    def make_Fig9(self, MSE_heatmap, noises_ctrl, noises_obs, title = 'Fig9'):
        plt.figure(figsize=(20, 16))
        plt.imshow(MSE_heatmap, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Mean Squared Error')
        # Set exactly 4 ticks per dimension
        num_ticks = 4
        xtick_indices = np.linspace(0, len(noises_ctrl)-1, num_ticks, dtype=int)
        ytick_indices = np.linspace(0, len(noises_obs)-1, num_ticks, dtype=int)
        plt.xticks(ticks=xtick_indices, labels=[f'{noises_ctrl[i]:.2f}' for i in xtick_indices])
        plt.yticks(ticks=ytick_indices, labels=[f'{noises_obs[i]:.2f}' for i in ytick_indices])
        plt.xlabel('Control Noise')
        plt.ylabel('Observation Noise')
        plt.title('MSE Heatmap for Different Noise Levels')
        plt.tight_layout()
        plt.savefig(title, dpi=300)
        plt.close()

    def make_Fig10(self, values1, values2, values3, title = 'Fig10'):

        spikes_vec1 = values1[8]
        spikes_vec2 = values2[8]
        spikes_vec3 = values3[8]

        time = values1[7]
        
        window = 500  # Adjust the window size as needed
        each_moment1 = np.sum(spikes_vec1, axis=0)
        average1 = np.convolve(each_moment1, np.ones(window)/window, mode='valid')
        each_moment2 = np.sum(spikes_vec2, axis=0)
        average2 = np.convolve(each_moment2, np.ones(window)/window, mode='valid')
        each_moment3 = np.sum(spikes_vec3, axis=0)
        average3 = np.convolve(each_moment3, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(20, 10))
        plt.suptitle('Total Network Firing Rates Over Time', fontsize=BIGGER_SIZE)

        plt.subplot(211)
        plt.plot(time[:len(average1)], average1, color='black', label='Nengo') 
        plt.legend()
        plt.ylabel('Spikes/s')
        plt.xticks([])
        #plt.yticks([800, 1000])

        plt.subplot(212)
        plt.plot(time[:len(average2)], average2, color='red', label='Gradient SCN')
        plt.plot(time[:len(average3)], average3, color='blue', label='SFEC SCN')
        plt.ylabel('Spikes/s')
        plt.xlabel('Time (s)')   
        plt.legend()
        plt.xticks([0, 10, 20, 30])
        plt.yticks([0.5, 1.0])

        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=300)  # Save the figure with high resolution
        plt.close()

    def make_metrics(self, values1, values2, values3, time):
        Nt = len(time)
        MSE1 = np.sum((values1[0][:Nt] - values1[4][:Nt])**2)
        MSE2 = np.sum((values2[0][:Nt] - values2[4][:Nt])**2)
        MSE3 = np.sum((values3[0][:Nt] - values3[4][:Nt])**2)

        Total_spikes1 = np.sum(values1[8])
        Total_spikes2 = np.sum(values2[8])
        Total_spikes3 = np.sum(values3[8])

        Inference_time1 = np.mean(values1[9])*1000
        Inference_time2 = np.mean(values2[9])*1000
        Inference_time3 = np.mean(values3[9])*1000

        return MSE1, MSE2, MSE3, Total_spikes1, Total_spikes2, Total_spikes3, Inference_time1, Inference_time2, Inference_time3
    
    def make_table(self, averages):
        averages_mean, averages_std = np.mean(averages, axis=1), np.std(averages, axis=1)
        MSE1, MSE2, MSE3, Total_spikes1, Total_spikes2, Total_spikes3, Inference_time1, Inference_time2, Inference_time3 = averages_mean

        table_data = [
            ["Controller",    "MSE",                                   "Spike Count",                                "Inference Time (ms)"],
            ["Nengo",         f"{MSE1:.2f} +/- {averages_std[0]:.2f}", f"{Total_spikes1:.2f} +/- {averages_std[3]:.2f}", f"{Inference_time1:.2f} +/- {averages_std[6]:.2f}"],
            ["Gradient SCN",  f"{MSE2:.2f} +/- {averages_std[1]:.2f}", f"{Total_spikes2:.2f} +/- {averages_std[4]:.2f}", f"{Inference_time2:.2f} +/- {averages_std[7]:.2f}"],
            ["SFEC SCN",      f"{MSE3:.2f} +/- {averages_std[2]:.2f}", f"{Total_spikes3:.2f} +/- {averages_std[5]:.2f}", f"{Inference_time3:.2f} +/- {averages_std[8]:.2f}"]
                            ]
        print("\nTable 1: Performance Comparison of Controllers")
        for row in table_data:
            print("{:<20} {:<10} {:<15} {:<15}".format(*row))
    
        SCN_efficiency = (Total_spikes2 / Total_spikes1)*100
        SFEC_efficiency = (Total_spikes3 / Total_spikes1)*100
        print(f"\nGradient SCN Efficiency (Spike Count Ratio to Nengo): {SCN_efficiency:.4f}%")
        print(f"SFEC SCN Efficiency (Spike Count Ratio to Nengo): {SFEC_efficiency:.4f}%")