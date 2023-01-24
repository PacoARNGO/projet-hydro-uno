import numpy as np
import pygmt
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import CenteredNorm
from matplotlib.colors import Normalize
from matplotlib.colors import Normalize
import statsmodels.api as sm
import seaborn as sns
import inspect

""" Interpolateur griddata"""

def interpolation_nearest(pts, coord, X_grid, Y_grid):
    return griddata(coord, pts, (X_grid,Y_grid), method='nearest' )

def interpolation_linear(pts, coord, X_grid, Y_grid):
    return griddata(coord, pts, (X_grid,Y_grid), method='linear' )

def interpolation_cubic(pts, coord, X_grid, Y_grid):
    return griddata(coord, pts, (X_grid,Y_grid), method='cubic' )


"""Fonctions utiles"""


def statistiques(interpolation, sol):
    me = np.nanmean(interpolation - sol)
    stde = np.nanstd(interpolation-sol)
    rmse = np.sqrt(np.nanmean((interpolation-sol)**2))
    return me, stde, rmse

def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

def interpolate(interpolateur, pts, multifaisceaux):
    # Interpolation
    bornes_MBES = pygmt.info(multifaisceaux, spacing=0.3)  # pas de grille 30 cm
    bornes_SBES = pygmt.info(pts, spacing=1)
    grid_MBES = pygmt.xyz2grd(multifaisceaux, spacing=(0.3, 0.3), region=bornes_MBES)
    echantillon = pygmt.grdsample(grid_MBES, spacing=1, region=bornes_SBES)
    x, y = echantillon.x.values, echantillon.y.values
    Z_multi = echantillon.values
    X_grid, Y_grid = np.meshgrid(x, y)
    coord, Z_mono_process = pts[:,0:2], pts[:, 2]
    interpolation = interpolateur(Z_mono_process, coord, X_grid, Y_grid)

    return interpolation, echantillon.x.values, echantillon.y.values, Z_multi


def stats(pts, multifaisceaux):
    #Interpolateur
    interpolation_lin, X, Y, sol = interpolate(interpolation_linear, pts, multifaisceaux)
    interpolation_near, X, Y, sol = interpolate(interpolation_nearest, pts, multifaisceaux)
    interpolation_cub, X, Y, sol = interpolate(interpolation_cubic, pts, multifaisceaux)

    #Affichage
    plt.figure()
    plt.subplot(3,1,1)
    plt.contourf(X, Y, interpolation_lin, levels=50, cmap='gist_earth')
    plt.title('Modèle {}'.format(retrieve_name(interpolation_lin)))
    plt.colorbar(label='Hauteur [m]')
    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.subplot(3,1, 2)
    plt.contourf(X, Y, interpolation_near, levels=50, cmap='gist_earth')
    plt.title('Modèle {}'.format(retrieve_name(interpolation_near)))
    plt.colorbar(label='Hauteur [m]')
    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.subplot(3,1, 3)
    plt.contourf(X, Y, interpolation_cub, levels=50, cmap='gist_earth')
    plt.title('Modèle {}'.format(retrieve_name(interpolation_lin)))
    plt.colorbar(label='Hauteur [m]')
    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.suptitle("Affichage méthodes griddata ")
    plt.savefig("comparaison_interpolateur/Affichage interpolations Griddata")
    plt.show()
    # Statistiques
    interpolation_stat_lin = interpolation_lin[~np.isnan(interpolation_lin)]
    sol_stat_lin = sol[~np.isnan(interpolation_lin)]
    me_lin, stde_lin, rmse_lin = statistiques(interpolation_stat_lin, sol_stat_lin)
    #print(f'Erreur moyenne : {me_lin:.3f}m')
    #print(f'Écart-type : {stde_lin:.3f}m')
    #print(f'RMSE : {rmse_lin:.3f}m')
    interpolation_stat_near = interpolation_near[~np.isnan(interpolation_near)]
    sol_stat_near = sol[~np.isnan(interpolation_near)]
    me_near, stde_near, rmse_near = statistiques(interpolation_stat_near, sol_stat_near)

    interpolation_stat_cub = interpolation_cub[~np.isnan(interpolation_cub)]
    sol_stat_cub = sol[~np.isnan(interpolation_cub)]
    me_cub, stde_cub, rmse_cub = statistiques(interpolation_stat_cub, sol_stat_cub)


    difference_lin = (interpolation_stat_lin - sol_stat_lin).flatten()
    difference_near = (interpolation_stat_near - sol_stat_near).flatten()
    difference_cub = (interpolation_stat_cub - sol_stat_cub).flatten()

    # boite à moustache
    difference_lin = difference_lin[~np.isnan(difference_lin)]
    difference_near = difference_near[~np.isnan(difference_near)]
    difference_cub = difference_cub[~np.isnan(difference_cub)]

    plt.figure()
    plt.subplot(1,3,1)
    plt.boxplot(difference_lin)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    plt.text(0.6, 3, r'RMSE={0:.3f}m'.format(rmse_lin))
    plt.text(0.6, 2, r'Erreur_moyenne={0:.3f}m'.format(me_lin))
    plt.text(0.6, 1, r'Ecart_type={0:.3f}m'.format(stde_lin))
    plt.title("Boite à moustache : {}".format(retrieve_name(interpolation_linear)))
    plt.subplot(1, 3, 2)
    plt.boxplot(difference_near)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    plt.text(0.6, 3, r'RMSE={0:.3f}m'.format(rmse_near))
    plt.text(0.6, 2, r'Erreur_moyenne={0:.3f}m'.format(me_near))
    plt.text(0.6, 1, r'Ecart_type={0:.3f}m'.format(stde_near))
    plt.title("Boite à moustache : {}".format(retrieve_name(interpolation_nearest)))
    plt.subplot(1, 3, 3)
    plt.boxplot(difference_cub)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    plt.text(0.1, 3, r'RMSE={0:.3f}m'.format(rmse_cub))
    plt.text(0.1, 2, r'Erreur_moyenne={0:.3f}m'.format(me_cub))
    plt.text(0.1, 1, r'Ecart_type={0:.3f}m'.format(stde_cub))
    plt.title("Boite à moustache : {}".format(retrieve_name(interpolation_cubic)))
    plt.suptitle("Boites à moustache méthodes griddata ")
    plt.savefig('comparaison_interpolateur/comparaison_interpolateur_boite_moustache.png')
    plt.show()

    # Histogramme
    fig, axes = plt.subplots(1, 3, sharex=True)
    axes[ 0].hist(difference_lin, bins=50)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[ 0].set_title('griddata, {}'.format(retrieve_name(interpolation_linear)))

    axes[ 1].hist(difference_near, bins=50)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[ 1].set_title('griddata, {}'.format(retrieve_name(interpolation_nearest)))

    axes[ 2].hist(difference_cub, bins=50)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[ 2].set_title('griddata, {}'.format(retrieve_name(interpolation_cubic)))
    plt.suptitle("Histogrammes méthodes griddata ")
    plt.savefig('comparaison_interpolateur/Histogrammes.png')
    plt.show()

    # Carte des différences
    difference_n = interpolation_near - sol
    difference_l = interpolation_lin - sol
    difference_c = interpolation_cub - sol
    cmap = 'bwr'
    fig, ax = plt.subplots(1, 3, figsize=(10, 6), constrained_layout=True)
    # Normalisation avec définition du demi-intervalle
    normalize1 = CenteredNorm(0, halfrange=1)
    c1 = ax[0].pcolor(X, Y, difference_n, cmap=cmap, norm=normalize1)
    ax[0].set(title='Différences nearest')
    plt.colorbar(c1, label='Différences profondeurs [m]', ax=ax[ 0])
    c2 = ax[1].pcolor(X, Y, difference_l, cmap=cmap, norm=normalize1)
    ax[1].set(title='Différences linear')
    plt.colorbar(c2, label='Différences profondeurs [m]', ax=ax[ 1])
    c31 = ax[ 2].pcolor(X, Y, difference_c, cmap=cmap, norm=normalize1)
    ax[2].set(title='Différences cubic')
    plt.colorbar(c31, label='Différences profondeurs [m]', ax=ax[2])
    plt.suptitle('Différences interpolation / solution multifaisceaux')
    for axe in ax.flat:
        axe.set(xlabel='Est [m]', ylabel='Nord [m]')
    for axe in ax.flat:
        axe.label_outer()
    plt.savefig('comparaison_interpolateur/Cartes des différence.png')
    plt.show()





if __name__ == """__main__""":
    FILENAME = 'data/ea400_200kilo.txt'
    data = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
    data_m = np.loadtxt('data/PortCommercePasseSante_Compile_2017a2022_GPSTide_30cm.xyz')
    stats(data, data_m)
