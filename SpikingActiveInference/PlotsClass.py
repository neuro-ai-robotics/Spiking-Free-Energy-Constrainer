import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
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
        plt.figure(figsize=(20, 15))
        #plt.rc('axes', prop_cycle=line_cycler)
        dims = int(len(x[0])/2)

        plt.subplot(311)
        target_lines = plt.plot(time, target[:, :dims], '--', color='grey', linewidth=5, label='$z$')
        state_lines = plt.plot(time, x[:len(time), :dims], color='red', linewidth=5, label='$x$')
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)
        #plt.legend(handles=[target_lines[0], state_lines[0]], labels=['z', 'x'], loc='upper left')

        plt.subplot(312)
        plt.plot(time, y[:len(time), :dims], color='blue', linewidth=5)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)
        plt.ylabel('Observations')

        #plt.subplot(313)
        #plt.plot(time, u[:len(time), :dims], color='black', linewidth=5)
        #plt.xticks([])
        #plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        #for spine in plt.gca().spines.values():
        #    spine.set_linewidth(4)

        a=np.where(spikes_vec[:, :])
        plt.subplot(313)
        plt.scatter(time[a[1]],a[0], marker='.',s=10,color='k',alpha=0.7,rasterized=True)
        #plt.xticks([0, 10, 20, 30])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=500, format='svg')
        plt.close()
    
    def make_Fig4(self, values, title = 'Fig4'):
        x, u, mu, y, target, eps, F, time, spikes_vec, Times = values

        plt.figure()
        plt.clf()
        ### Image size
        plt.figure(figsize=(20, 15))
        #plt.rc('axes', prop_cycle=line_cycler)
        dims = int(len(x[0])/2)

        plt.subplot(311)
        target_lines = plt.plot(time, target[:, :dims], '--', color='grey', linewidth=5)
        state_lines_x = plt.plot(time, x[:len(time), :dims:2], color='red', linewidth=5)
        state_lines_y = plt.plot(time, x[:len(time), 1:dims:2], color='blue', linewidth=5)
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)
        plt.legend(handles=[target_lines[0], state_lines_x[0], state_lines_y[0]], labels=['$z$', '$x_1$', '$x_2$'], loc='upper left')

        #plt.subplot(412)
        #plt.plot(time, y[:len(time), :dims], color='black', linewidth=3)
        #plt.ylabel('Observations')

        plt.subplot(312)
        plt.plot(time, u[:len(time), :dims], color='black', linewidth=5)
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        a=np.where(spikes_vec[:, :])
        plt.subplot(313)
        plt.scatter(time[a[1]],a[0], marker='.',s=10,color='k',alpha=0.7,rasterized=True)
        plt.xticks([0, 10, 20, 30])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=500, format='svg')
        plt.close()

    def make_Fig4_vertical(self, values_list, column_titles=None, filename='Fig4'):
        """
        Generates a vertical stack of plots for three different sets of simulation values,
        formatted for A4 pages.
        
        Args:
            values_list (list): A list containing exactly 3 tuples. Each tuple contains
                                the simulation data (x, u, mu, y, target, eps, F, time, 
                                spikes_vec, Times) for one condition.
            column_titles (list, optional): A list of 3 strings representing the title 
                                            for each condition group. Defaults to None.
            filename (str): Filename for the output figure.
        """
        if len(values_list) != 3:
            raise ValueError("values_list must contain exactly 3 sets of values.")

        # A4 dimensions in inches (Portrait)
        plt.figure(figsize=(8.27, 11.69))
        plt.clf()
        
        # Styling parameters scaled for A4 readability
        label_fontsize = 9
        title_fontsize = 10
        tick_fontsize = 8
        line_width = 1.0
        spine_width = 0.5
        marker_size = 1

        # Iterate through the three provided value sets
        for i, values in enumerate(values_list):
            x, u, mu, y, target, eps, F, time, spikes_vec, Times = values
            
            dims = int(len(x[0]) / 2)

            # --- Section i: Title + States ---
            # Subplot position: 3*i + 1 (indices 1, 4, 7)
            plt.subplot(9, 1, 3*i + 1)
            
            target_lines = plt.plot(time, target[:, :dims], '--', color='grey', linewidth=line_width)
            state_lines_x = plt.plot(time, x[:len(time), :dims:2], color='red', linewidth=line_width)
            state_lines_y = plt.plot(time, x[:len(time), 1:dims:2], color='blue', linewidth=line_width)
            
            # Set the group title if provided
            if column_titles and i < len(column_titles):
                plt.title(column_titles[i], fontsize=title_fontsize, fontweight='bold')

            plt.xticks([])
            plt.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=spine_width, length=4)
            for spine in plt.gca().spines.values():
                spine.set_linewidth(spine_width)
            
            # Add legend only to the top plot to save space
            if i == 0:
                plt.legend(handles=[target_lines[0], state_lines_x[0], state_lines_y[0]], 
                        labels=['$z$', '$x_1$', '$x_2$'], loc='upper left', fontsize=tick_fontsize)

            # --- Section i: Control Signals ---
            # Subplot position: 3*i + 2 (indices 2, 5, 8)
            plt.subplot(9, 1, 3*i + 2)
            plt.plot(time, u[:len(time), :dims], color='black', linewidth=line_width)
            
            plt.xticks([])
            plt.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=spine_width, length=4)
            for spine in plt.gca().spines.values():
                spine.set_linewidth(spine_width)

            # --- Section i: Spike Raster ---
            # Subplot position: 3*i + 3 (indices 3, 6, 9)
            plt.subplot(9, 1, 3*i + 3)
            
            a = np.where(spikes_vec[:, :])
            plt.scatter(time[a[1]], a[0], marker='.', s=marker_size, color='k', alpha=0.7, rasterized=True)
            
            # Show x-ticks on all raster plots for clarity within each group
            plt.xticks([0, 10, 20, 30])
            plt.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=spine_width, length=4)
            for spine in plt.gca().spines.values():
                spine.set_linewidth(spine_width)

        # Adjust layout
        plt.tight_layout()
        # Reduce vertical spacing slightly to fit all plots comfortably
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(filename, dpi=300, format='svg')
        plt.close()

    def make_Fig5(self, values, title = 'Fig5'):
        x, u, mu, y, target, eps, F, time, spikes_vec, Times = values
        dims1 = int(len(x[0])/2)

        plt.figure(figsize=(20, 20))
        plt.subplot(4, 1, 1)
        state_lines = plt.plot(time, x[:len(time), :dims1], color='red', linewidth=4)
        target_lines = plt.plot(time, target[:len(time), :dims1], '--', color='grey', linewidth=4)
        plt.xlim(0, 30)
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)
        plt.legend(handles=[target_lines[0], state_lines[0]], labels=['$z$', '$x$'], loc='upper left')

        plt.subplot(4, 1, 2)
        a=np.where(spikes_vec[:, :])
        plt.scatter(time[a[1]],a[0], marker='.',s=10,color='k',alpha=0.7,rasterized=True)
        plt.xlim(0, 30)
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.subplot(4, 1, 3)
        plt.plot(time, F[:len(time)], color='black', linewidth=5)
        plt.xlim(0, 30)
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        # Calculate MSE between state and target
        MSE = np.mean((x[:len(time), :dims1] - target[:len(time), :dims1])**2, axis=1)
        plt.subplot(4, 1, 4)
        plt.plot(time, MSE, color='blue', linewidth=5)
        plt.xlim(0, 30)
        plt.ylim(0, 0.5)
        plt.xticks([0, 10, 20, 30])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=500, format='svg')  # Save the figure with high resolution
        plt.close()

    def make_Fig8(self, values_dif, values_same, title = 'Fig8'):
        x, u, mu, y, target, eps, F, time, spikes_vec, Times = values_dif
        x_same, u_same, mu_same, y_same, target_same, eps_same, F_same, time_same, spikes_vec_same, Times_same = values_same

        plt.figure(figsize=(36, 10))
        plt.subplot(1, 2, 1)
        # Show starting positions with crosses and final positions with triangles
        plt.scatter(x_same[0, 0], x_same[0, 1], color='red', s=300, marker='x')
        plt.scatter(x_same[0, 2], x_same[0, 3], color='green', s=300, marker='x')
        plt.scatter(x_same[0, 4], x_same[0, 5], color='blue', s=300, marker='x')

        plt.plot(x_same[:, 0], x_same[:, 1], color='red', label ='Drone 1', linewidth=5)
        plt.plot(x_same[:, 2], x_same[:, 3], color='green', label ='Drone 2', linewidth=5)
        plt.plot(x_same[:, 4], x_same[:, 5], color='blue', label ='Drone 3', linewidth=5)

        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.legend()

        plt.subplot(1, 2, 2)
        # Show starting positions with crosses and final positions with triangles
        plt.scatter(x[0, 0], x[0, 1], color='red', s=300, marker='x')
        plt.scatter(x[0, 2], x[0, 3], color='green', s=300, marker='x')
        plt.scatter(x[0, 4], x[0, 5], color='blue', s=300, marker='x')

        plt.plot(x[:, 0], x[:, 1], color='red', label='Drone 1', linewidth=5)
        plt.plot(x[:, 2], x[:, 3], color='green', label='Drone 2', linewidth=5)
        plt.plot(x[:, 4], x[:, 5], color='blue', label='Drone 3', linewidth=5)

        plt.plot(target[:, 0], target[:, 1], color='black', label='Target', linewidth=5, linestyle='--')
        plt.legend()

        #less ticks for clarity
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=4)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)


        plt.subplots_adjust(hspace=0.2)
        plt.savefig(title, dpi=500, format='svg')
        plt.close()

    def make_Fig9(self, MSE_heatmap, noises_ctrl, noises_obs, title='Fig9'):
        fig, ax = plt.subplots(figsize=(20, 8))

        # Check shapes (optional debug prints)
        # print("MSE shape:", MSE_heatmap.shape)
        # print("lens:", len(noises_ctrl), len(noises_obs))

        # --- discrete color bins and colormap ---
        n_bins = 5
        cmap = plt.cm.viridis.copy()
        cmap.set_under(cmap(0))
        bounds = np.linspace(np.min(MSE_heatmap), np.max(MSE_heatmap), n_bins + 1)
        norm = BoundaryNorm(bounds, ncolors=cmap.N)

        # --- make sure rows = observation noise, cols = control noise ---
        # If MSE_heatmap.shape == (len(noises_obs), len(noises_ctrl)) we must transpose
        if MSE_heatmap.shape == (len(noises_obs), len(noises_ctrl)):
            data = MSE_heatmap
        else:
            data = MSE_heatmap.T  # assume shape already (len(noises_obs), len(noises_ctrl))

        # --- map axes to actual log-noise values for exact alignment ---
        x = np.log(noises_ctrl)
        y = np.log(noises_obs)
        extent = [x.min(), x.max(), y.min(), y.max()]

        im = ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
            origin='lower',
            aspect='auto',
            interpolation='nearest',
            extent=extent
        )

        # --- colorbar ---
        cbar = fig.colorbar(im, ax=ax, ticks=bounds)
        cbar.ax.set_yticklabels([f'{b:.2f}' for b in bounds])
        #cbar.set_label('Mean Squared Error')

        # --- ticks: exactly 4 per axis, using the log-noise coordinates ---
        # make ticks larger for visibility
        num_ticks = 3
        ax.set_xticks(np.linspace(x.min(), x.max(), num_ticks))
        ax.set_yticks(np.linspace(y.min(), y.max(), num_ticks))
        ax.set_xticklabels([f'{v:.2f}' for v in np.linspace(noises_ctrl.min(), noises_ctrl.max(), num_ticks)])
        ax.set_yticklabels([f'{v:.0f}' for v in np.linspace(noises_obs.min(), noises_obs.max(), num_ticks)])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        #ax.set_xlabel('Process Noise (LogScale)')
        #ax.set_ylabel('Observation Noise (LogScale)')
        #ax.set_title('MSE Heatmap for Different Noise Levels', fontsize=2*BIGGER_SIZE)

        fig.savefig(title, dpi=500, format='svg')
        plt.close(fig)

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
        #plt.suptitle('Total Network Firing Rates Over Time', fontsize=2*BIGGER_SIZE)

        plt.plot(time[:len(average1)], average1, color='black', label='Nengo', linewidth=3)
        plt.plot(time[:len(average2)], average2, color='red', label='Gradient SCN', linewidth=3)
        plt.plot(time[:len(average3)], average3, color='blue', label='SFEC SCN', linewidth=3)
        plt.legend()
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=45, width=4, length=10)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)

        plt.savefig(title, dpi=500, format='svg')  # Save the figure with high resolution
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
            ["Controller",    "MSE                     ",                                   "Spike Count                    ",                                "Inference Time (ms)"],
            ["Nengo",         f"{MSE1:.2f} +/- {averages_std[0]:.2f}     ", f"{Total_spikes1:.2f} +/- {averages_std[3]:.2f}           ", f"{Inference_time1:.2f} +/- {averages_std[6]:.2f}"],
            ["Gradient SCN",  f"{MSE2:.2f} +/- {averages_std[1]:.2f}     ", f"{Total_spikes2:.2f} +/- {averages_std[4]:.2f}           ", f"{Inference_time2:.2f} +/- {averages_std[7]:.2f}"],
            ["SFEC SCN",      f"{MSE3:.2f} +/- {averages_std[2]:.2f}     ", f"{Total_spikes3:.2f} +/- {averages_std[5]:.2f}           ", f"{Inference_time3:.2f} +/- {averages_std[8]:.2f}"]
                            ]
        print("\nTable 1: Performance Comparison of Controllers")
        for row in table_data:
            print("{:<20} {:<10} {:<15} {:<15}".format(*row))
    
        SFEC_efficiency_SCN = (Total_spikes3 / Total_spikes2)*100
        SFEC_efficiency_Nengo = (Total_spikes3 / Total_spikes1)*100
        print(f"SFEC SCN Efficiency (Spike Count Ratio to Nengo): {SFEC_efficiency_Nengo:.4f}%")
        print(f"SFEC SCN Efficiency (Spike Count Ratio to Gradient SCN): {SFEC_efficiency_SCN:.4f}%")
