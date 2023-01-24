import numpy as np
import matplotlib.pyplot as plt
import os



def plot_hist(z_obs, xlabel = "", ylabel = "", title = "", fileo = "", bins=20):
    """ Plot histogramme

        Parameters
        ----------
        z_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Altitudes des observations.
        xlabel :
            TYPE, optional :
                string
            DESCRIPTION :
                Label de l'axe des abscisses.
        ylabel :
            TYPE, optional :
                string
            DESCRIPTION :
                Label de l'axe des ordonnées.
        title :
            TYPE, optional :
                string
            DESCRIPTION :
                Titre du graphique."""
    import seaborn as sns

    # Histogramme
    plt.figure(figsize=(16, 8))
    sns.set(style="whitegrid")
    sns.set_palette("husl")

    sns.histplot(z_obs, kde = True, stat='density', bins = bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=16, fontweight='bold')
    if fileo != "":
        plt.savefig(fileo)

def plot_boxplot(z_obs, xlabel = "", title = "", fileo = ""):
    """ Plot boxplot

        Parameters
        ----------
        z_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Altitudes des observations.
        xlabel :
            TYPE, optional :
                string
            DESCRIPTION :
                Label de l'axe des abscisses.
        ylabel :
            TYPE, optional :
                string
            DESCRIPTION :
                Label de l'axe des ordonnées.
        title :
            TYPE, optional :
                string
            DESCRIPTION :
                Titre du graphique."""

    # Boxplot
    plt.figure(figsize=(16, 8))
    plt.boxplot(z_obs)
    plt.xlabel(xlabel)
    plt.title(title, fontsize=16, fontweight='bold')
    if fileo != "":
        plt.savefig(fileo)

def plot_surface2D(x, y, z, xlabel = "", ylabel = "", title = "", fileo = ""):
    """ Plot surface 2D

        Parameters
        ----------
        x :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Abscisses des points.
        y :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Ordonnées des points.
        z :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Altitudes des points.
        xlabel :
            TYPE, optional :
                string
            DESCRIPTION :
                Label de l'axe des abscisses.
        ylabel :
            TYPE, optional :
                string
            DESCRIPTION :
                Label de l'axe des ordonnées.
        title :
            TYPE, optional :
                string
            DESCRIPTION :
                Titre du graphique."""

    # Surface 2D
    plt.figure(figsize=(16, 8))
    plt.contourf(x, y, z, 100, cmap='jet')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=16, fontweight='bold')
    if fileo != "":
        plt.savefig(fileo)


if __name__ == '__main__':
    # Read the data
    if not os.path.exists('data/diff_frequences.txt'):
        data_38 = np.loadtxt('data/ea400_38kilo.txt', delimiter=',')
        file = np.column_stack((data_38[:,0],data_38[:,1],data_38[:,3], data_38[:,3] - data_38[:,4]))
        np.savetxt('data/diff_frequences.txt', file, fmt='%.3f')
        # with open('data/diff_frequences.txt', 'w') as f:
        #     for line in data_38:
        #         f.write(f'{line[0]} {line[1]} {line[3]} {line[3] - line[4]} \n')


    diff_freq = np.loadtxt('data/diff_frequences.txt')

    # Plot the data
    diff = diff_freq[:,-1]
    mean = np.mean(diff)
    std = np.std(diff)
    rms = np.sqrt(np.mean(diff**2))
    mini = np.min(diff)
    maxi = np.max(diff)

    print(f'Mean: {mean} Std: {std} RMS:{rms} Min: {mini} Max: {maxi}')


    plot_hist(diff, bins=np.linspace(-0.3, 0.3, 100), \
                title="Différence de mesure de profondeur entre le sondeur à 38kHz et le sondeur à 200 kHz", \
                xlabel='mètres', ylabel='densité', fileo='images/difference_hist_38_200.png')

    plt.figure(figsize=(16, 8))
    sc = plt.scatter(diff_freq[:, 0], diff_freq[:, 1], c=diff_freq[:, 3], cmap='gist_earth')
    plt.title('Différences entre les mesures de profondeur à 38kHz et 200kHz', fontsize=16, fontweight='bold')
    plt.colorbar(sc, label='Hauteur [m]')

    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.savefig('images/difference_surface_38_200.png')


    plot_boxplot(diff, title="Différence de mesure de profondeur entre le sondeur à 38kHz et le sondeur à 200 kHz", \
                xlabel='mètres', fileo='images/difference_boxplot_38_200.png')


