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

def interpolation_linear38(pts, coord, X_grid, Y_grid):
    return griddata(coord, pts, (X_grid,Y_grid), method='linear' )

def interpolation_cubic(pts, coord, X_grid, Y_grid):
    return griddata(coord, pts, (X_grid,Y_grid), method='cubic' )
""""""
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


def plot_interpolation(interpolateur, pts, multifaisceaux):
    interpolation, X, Y, sol = interpolate(interpolateur, pts, multifaisceaux)
    plt.figure()
    im = plt.contourf(X, Y, interpolation, levels=50, cmap='gist_earth')
    plt.title('Modèle {}'.format(retrieve_name(interpolateur)))
    plt.colorbar(im, label='Hauteur [m]')
    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.show()
    # Statistiques
    interpolation_stat = interpolation[~np.isnan(interpolation)]
    sol_stat = sol[~np.isnan(interpolation)]
    #sol_stat = sol_stat[~np.isin(interpolation, interpolation_stat)]
    me, stde, rmse = statistiques(interpolation_stat, sol_stat)
    print(f'Erreur moyenne : {me:.3f}m')
    print(f'Écart-type : {stde:.3f}m')
    print(f'RMSE : {rmse:.3f}m')
    # boite à moustache
    difference = (interpolation_stat - sol_stat).flatten()
    difference = difference[~np.isnan(difference)]
    plt.figure()
    plt.boxplot(difference)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    plt.text(0.6, 6, r'RMSE={0:.3f}m'.format(rmse))
    plt.text(0.6, 5, r'Erreur_moyenne={0:.3f}m'.format(me))
    plt.text(0.6, 4, r'Ecart_type={0:.3f}m'.format(stde))
    plt.title("Boite à moustache interpolateur : {}".format(retrieve_name(interpolateur)))
    plt.savefig('Images/boite_moustache_{}.png'.format(retrieve_name(interpolateur).replace(' ', '_')))
    plt.show()
    return interpolation, sol

def histo_carte_diff(interpolateur, pts, multifaisceaux):
    interpolation, X, Y, sol = interpolate(interpolateur, pts, multifaisceaux)
    difference = interpolation - sol
    # Histogramme
    plt.figure()
    diff_1D = difference.flatten()
    hp = plt.hist(diff_1D, bins=50, density=True)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    plt.title('HISTOGRAMME Différences interpolation {} / multifaisceaux'.format(retrieve_name(interpolateur)))
    plt.savefig('Images/Histograme_{}.png'.format(retrieve_name(interpolateur).replace(' ', '_')))
    plt.show()
    # Carte des différences
    plt.figure()
    cmap = 'bwr'
    normalize1 = CenteredNorm(0, halfrange=1)
    plt.pcolor(X, Y, difference, cmap=cmap, norm=normalize1)
    plt.colorbar(label='Différences profondeurs [m]')
    plt.title("Carte des différences interpolateur : ".format(retrieve_name(interpolateur)))
    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.savefig('Images/Carte de différence{}.png'.format(retrieve_name(interpolateur).replace(' ', '_')))
    plt.show()












if __name__ == """__main__""":
    FILENAME = 'data/ea400_38kilo.txt'
    data = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
    data_m = np.loadtxt('data/PortCommercePasseSante_Compile_2017a2022_GPSTide_30cm.xyz')
    plot_interpolation(interpolation_linear38, data, data_m)
    histo_carte_diff(interpolation_linear38, data, data_m)










