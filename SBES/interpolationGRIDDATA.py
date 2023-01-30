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
import math
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
    bornes = [min(bornes_SBES[0], bornes_MBES[0]), max(bornes_MBES[1], bornes_SBES[1]), min(bornes_SBES[2], bornes_MBES[2]), max(bornes_MBES[3], bornes_SBES[3]) ]

    grid_MBES = pygmt.xyz2grd(multifaisceaux, spacing=(0.3, 0.3), region=bornes) #bornes_MBES
    echantillon = pygmt.grdsample(grid_MBES, spacing=1, region=bornes_SBES) #bornes_SBES
    x, y = echantillon.x.values, echantillon.y.values
    Z_multi = echantillon.values
    X_grid, Y_grid = np.meshgrid(x, y)
    coord, Z_mono_process = pts[:,0:2], pts[:, 2]
    interpolation = interpolateur(Z_mono_process, coord, X_grid, Y_grid)

    return interpolation, echantillon.x.values, echantillon.y.values, Z_multi


def stats(pts, multifaisceaux):
    #HEADLINE

    #bornes_MBES = pygmt.info(multifaisceaux, spacing=0.3)  # pas de grille 30 cm
    #bornes_SBES = pygmt.info(pts, spacing=1)
    #grid_MBES = pygmt.xyz2grd(multifaisceaux, spacing=(0.3, 0.3), region=bornes_MBES)
    #grid_multi = pygmt.grdsample(grid=grid_MBES, region=bornes_SBES, spacing=1)

    #Interpolateur
    interpolation_lin, X, Y, sol = interpolate(interpolation_linear, pts, multifaisceaux)
    interpolation_near, X, Y, sol = interpolate(interpolation_nearest, pts, multifaisceaux)
    interpolation_cub, X, Y, sol = interpolate(interpolation_cubic, pts, multifaisceaux)
    PAS_GRILLE = 1
    rayon = 35
    secteur = 8
    bornes = pygmt.info(pts, spacing=PAS_GRILLE)
    inter_tri = pygmt.triangulate.regular_grid(data=pts, spacing=PAS_GRILLE, region=bornes)
    inter_neigbour = pygmt.nearneighbor(data=pts, spacing=PAS_GRILLE, region=bornes, search_radius=rayon,
                                        sectors=secteur)
    df = pygmt.blockmean(data=pts, region=bornes, spacing=PAS_GRILLE)
    inter_surface = pygmt.surface(data=df, spacing=PAS_GRILLE, region=bornes, T=0.35, verbose="i")
    inter_surface.values[np.isnan(inter_tri)] = np.nan
    interpolation_near[np.isnan(inter_tri)] = np.nan

    ##Affichage
    #plt.figure()
    ##GRIDDATA
    #plt.subplot(6,1,1)
    #plt.contourf(X, Y, interpolation_lin, levels=50, cmap='gist_earth')
    #plt.title('Modèle {}'.format(retrieve_name(interpolation_lin)))
    #plt.colorbar(label='Hauteur [m]')
    #plt.xlabel('Est [m]')
    #plt.ylabel('Nord [m]')
    #plt.subplot(6,1, 2)
    #plt.contourf(X, Y, interpolation_near, levels=50, cmap='gist_earth')
    #plt.title('Modèle {}'.format(retrieve_name(interpolation_near)))
    #plt.colorbar(label='Hauteur [m]')
    #plt.xlabel('Est [m]')
    #plt.ylabel('Nord [m]')
    #plt.subplot(6,1, 3)
    #plt.contourf(X, Y, interpolation_cub, levels=50, cmap='gist_earth')
    #plt.title('Modèle {}'.format(retrieve_name(interpolation_cub)))
    #plt.colorbar(label='Hauteur [m]')
    #plt.xlabel('Est [m]')
    #plt.ylabel('Nord [m]')
    ## PYGMT
    ## INTERPOLATEUR DE TRIANGULATION
    #plt.subplot(6, 2, 4)
    #inter_tri = pygmt.triangulate.regular_grid(data=pts, spacing=PAS_GRILLE, region=bornes, verbose='i')
    #im = plt.contourf(inter_tri.x, inter_tri.y, inter_tri, levels=50, cmap='gist_earth')
    #plt.title('Modèle {}'.format("interpolation_triangulation"))
    #plt.colorbar(im, label='Hauteur [m]')
    #plt.xlabel('Est [m]')
    #plt.ylabel('Nord [m]')
    ## INTERPOLATEUR PLUS PROCHES VOISINS
    #plt.subplot(6, 2, 5)
    #im = plt.contourf(inter_neigbour.x, inter_neigbour.y, inter_neigbour, levels=50, cmap='gist_earth')
    #plt.title('Modèle {}'.format("interpolation +proche voisins"))
    #plt.colorbar(im, label='Hauteur [m]')
    #plt.xlabel('Est [m]')
    #plt.ylabel('Nord [m]')
    ## INTERPOLATEUR SURFACE
    #plt.subplot(6, 2, 6)
    #im = plt.contourf(inter_surface.x, inter_surface.y, inter_surface, levels=50, cmap='gist_earth')
    #plt.title('Modèle {}'.format("interpolation surface"))
    #plt.colorbar(im, label='Hauteur [m]')
    #plt.xlabel('Est [m]')
    #plt.ylabel('Nord [m]')
    #plt.suptitle("Affichage méthodes griddata ")
    #plt.savefig("comparaison_interpolateur/Affichage interpolations")
    #plt.show()
    # Statistiques
    interpolation_stat_lin = interpolation_lin[~np.isnan(interpolation_lin)]
    sol_stat_lin = sol[~np.isnan(interpolation_lin)]
    me_lin, stde_lin, rmse_lin = statistiques(interpolation_stat_lin, sol_stat_lin)
    print("-----------------------------C'est l'heure des statistiques----------------------------------------")
    print("linear")
    print(f'Erreur moyenne : {me_lin:.3f}m')
    print(f'Écart-type : {stde_lin:.3f}m')
    print(f'RMSE : {rmse_lin:.3f}m')
    print("------------------------------------------------------------------------------------------------")
    interpolation_stat_near = interpolation_near[~np.isnan(interpolation_near)]
    sol_stat_near = sol[~np.isnan(interpolation_near)]
    me_near, stde_near, rmse_near = statistiques(interpolation_stat_near, sol_stat_near)
    print("nearest")
    print(f'Erreur moyenne : {me_near:.3f}m')
    print(f'Écart-type : {stde_near:.3f}m')
    print(f'RMSE : {rmse_near:.3f}m')
    print("------------------------------------------------------------------------------------------------")
    print("cubic")
    interpolation_stat_cub = interpolation_cub[~np.isnan(interpolation_cub)]
    sol_stat_cub = sol[~np.isnan(interpolation_cub)]
    me_cub, stde_cub, rmse_cub = statistiques(interpolation_stat_cub, sol_stat_cub)
    print(f'Erreur moyenne : {me_cub:.3f}m')
    print(f'Écart-type : {stde_cub:.3f}m')
    print(f'RMSE : {rmse_cub:.3f}m')
    print("------------------------------------------------------------------------------------------------")
    print("triangulation")
    interpolation_stat_tri = inter_tri.values[~np.isnan(inter_tri.values)]
    sol_stat_tri = sol[~np.isnan(inter_tri.values)]
    me_tri, stde_tri, rmse_tri = statistiques(interpolation_stat_tri, sol_stat_tri)
    print(f'Erreur moyenne : {me_tri:.3f}m')
    print(f'Écart-type : {stde_tri:.3f}m')
    print(f'RMSE : {rmse_tri:.3f}m')
    print("------------------------------------------------------------------------------------------------")
    print("plus proche voisin")
    interpolation_stat_neigh = inter_neigbour.values[~np.isnan(inter_neigbour.values)]
    sol_stat_neigh = sol[~np.isnan(inter_neigbour.values)]
    me_neigh, stde_neigh, rmse_neigh = statistiques(interpolation_stat_neigh, sol_stat_neigh)
    print(f'Erreur moyenne : {me_neigh:.3f}m')
    print(f'Écart-type : {stde_neigh:.3f}m')
    print(f'RMSE : {rmse_neigh:.3f}m')
    print("------------------------------------------------------------------------------------------------")
    print("surface")
    interpolation_stat_surf = inter_surface.values[~np.isnan(inter_surface.values)]
    sol_stat_surf = sol[~np.isnan(inter_surface.values)]
    me_surf, stde_surf, rmse_surf = statistiques(interpolation_stat_surf, sol_stat_surf)
    print(f'Erreur moyenne : {me_surf:.3f}m')
    print(f'Écart-type : {stde_surf:.3f}m')
    print(f'RMSE : {rmse_surf:.3f}m')
    print("------------------------------------------------------------------------------------------------")
    difference_lin = (interpolation_stat_lin - sol_stat_lin).flatten()
    difference_near = (interpolation_stat_near - sol_stat_near).flatten()
    difference_cub = (interpolation_stat_cub - sol_stat_cub).flatten()
    difference_lin = difference_lin[~np.isnan(difference_lin)]
    difference_near = difference_near[~np.isnan(difference_near)]
    difference_cub = difference_cub[~np.isnan(difference_cub)]

    diff_tri = (interpolation_stat_tri - sol_stat_tri).flatten()
    diff_neigh = (interpolation_stat_neigh - sol_stat_neigh).flatten()
    diff_surf = (interpolation_stat_surf - sol_stat_surf).flatten()
    diff_tri = diff_tri[~np.isnan(diff_tri)]
    diff_neigh = diff_neigh[~np.isnan(diff_neigh)]
    diff_surf = diff_surf[~np.isnan(diff_surf)]

    # Différence pygmt
    #diff_tri = (inter_tri.values[~np.isnan(inter_tri.values)] - sol[~np.isnan(inter_tri.values)]).flatten()
    #diff_neighbor = (inter_neigbour.values[~np.isnan(inter_neigbour.values)] - sol[
    #    ~np.isnan(inter_neigbour.values)]).flatten()
    #diff_surface = (inter_surface.values[~np.isnan(inter_surface.values)] - sol[
    #    ~np.isnan(inter_surface.values)]).flatten()
#

    # boite à moustache

    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.boxplot(difference_lin)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # plt.text(0.6, 3, r'RMSE={0:.3f}m'.format(rmse_lin))
    # plt.text(0.6, 2, r'Erreur_moyenne={0:.3f}m'.format(me_lin))
    # plt.text(0.6, 1, r'Ecart_type={0:.3f}m'.format(stde_lin))
    # plt.title("Boite à moustache : {}".format(retrieve_name(interpolation_linear)))
    # plt.subplot(1, 3, 2)
    # plt.boxplot(difference_near)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # plt.text(0.6, 3, r'RMSE={0:.3f}m'.format(rmse_near))
    # plt.text(0.6, 2, r'Erreur_moyenne={0:.3f}m'.format(me_near))
    # plt.text(0.6, 1, r'Ecart_type={0:.3f}m'.format(stde_near))
    # plt.title("Boite à moustache : {}".format(retrieve_name(interpolation_nearest)))
    # plt.subplot(1, 3, 3)
    # plt.boxplot(difference_cub)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # plt.text(0.1, 3, r'RMSE={0:.3f}m'.format(rmse_cub))
    # plt.text(0.1, 2, r'Erreur_moyenne={0:.3f}m'.format(me_cub))
    # plt.text(0.1, 1, r'Ecart_type={0:.3f}m'.format(stde_cub))
    # plt.title("Boite à moustache : {}".format(retrieve_name(interpolation_cubic)))
    # plt.suptitle("Boites à moustache méthodes griddata ")

    # fig, axes = plt.subplots(2, 3, sharey=True, sharex=True)
    # axes[0, 0].boxplot(difference_near)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # #plt.text(0.6, 3, r'RMSE={0:.3f}m'.format(rmse_near))
    # #plt.text(0.6, 2, r'Erreur_moyenne={0:.3f}m'.format(me_near))
    # #plt.text(0.6, 1, r'Ecart_type={0:.3f}m'.format(stde_near))
    # axes[0, 0].set_title("Boite à moustache : {}".format(retrieve_name(interpolation_nearest)))
    #
    #
    # axes[0, 1].boxplot(difference_lin)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # #plt.text(0.6, 3, r'RMSE={0:.3f}m'.format(rmse_lin))
    # #plt.text(0.6, 2, r'Erreur_moyenne={0:.3f}m'.format(me_lin))
    # #plt.text(0.6, 1, r'Ecart_type={0:.3f}m'.format(stde_lin))
    # axes[0, 1].set_title("Boite à moustache : {}".format(retrieve_name(interpolation_linear)))
    #
    #
    # axes[0, 2].boxplot(difference_cub)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # #plt.text(0.1, 3, r'RMSE={0:.3f}m'.format(rmse_cub))
    # #plt.text(0.1, 2, r'Erreur_moyenne={0:.3f}m'.format(me_cub))
    # #plt.text(0.1, 1, r'Ecart_type={0:.3f}m'.format(stde_cub))
    # axes[0, 2].set_title("Boite à moustache : {}".format(retrieve_name(interpolation_cubic)))
    #
    #
    # axes[1, 0].boxplot(diff_tri)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # #plt.text(0.1, 3, r'RMSE={0:.3f}m'.format(rmse_tri))
    # #plt.text(0.1, 2, r'Erreur_moyenne={0:.3f}m'.format(me_tri))
    # #plt.text(0.1, 1, r'Ecart_type={0:.3f}m'.format(stde_tri))
    # axes[1, 0].set_title("Boite à moustache : interpolation_triangulation")
    #
    # axes[1, 1].boxplot(diff_neighbor)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # #plt.text(0.1, 3, r'RMSE={0:.3f}m'.format(rmse_neigh))
    # #plt.text(0.1, 2, r'Erreur_moyenne={0:.3f}m'.format(me_neigh))
    # #plt.text(0.1, 1, r'Ecart_type={0:.3f}m'.format(stde_neigh))
    # axes[1, 1].set_title("Boite à moustache : interpolation_plus_proche_voisins")
    #
    # axes[1, 2].boxplot(diff_surface)
    # plt.xlabel('Différences de profondeurs [m]')
    # plt.ylabel('Densité')
    # #plt.text(0.1, 3, r'RMSE={0:.3f}m'.format(rmse_surf))
    # #plt.text(0.1, 2, r'Erreur_moyenne={0:.3f}m'.format(me_surf))
    # #plt.text(0.1, 1, r'Ecart_type={0:.3f}m'.format(stde_surf))
    # axes[1, 2].set_title("Boite à moustache : interpolation_surface")
    #
    # plt.suptitle("Boîtes à moustaches")
    # plt.savefig('comparaison_interpolateur/comparaison_interpolateur_boite_moustache.png')
    # plt.show()
    plt.figure()
    names = ['triangulation', 'nearneighbor', 'surface', 'nearest', 'linear', 'cubic']
    plt.boxplot([diff_tri, diff_neigh, diff_surf, difference_near, difference_lin, difference_cub], labels=names, showfliers=False)
    plt.title("Comparaison des boxplots des méthodes d'interpolations avec les données multifaisceaux")
    plt.savefig('comparaison_interpolateur/comparaison_interpolateur_boite_moustache.png')
    plt.show()
    # Histogramme
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    axes[0, 0].hist(difference_lin, bins=50, density=True)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[0, 0].set_title('griddata, {}'.format(retrieve_name(interpolation_linear)))

    axes[0, 1].hist(difference_near, bins=50, density=True)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[0, 1].set_title('griddata, {}'.format(retrieve_name(interpolation_nearest)))

    axes[0, 2].hist(difference_cub, bins=50, density=True)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[0, 2].set_title('griddata, {}'.format(retrieve_name(interpolation_cubic)))
    axes[1,0].hist(diff_tri, bins=50, density=True)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[1,0].set_title('gmt, {}'.format(retrieve_name(diff_tri)))

    axes[1,1].hist(diff_neigh, bins=50, density=True)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[1,1].set_title('gmt, {}'.format(retrieve_name(diff_neigh)))

    axes[1,2].hist(diff_surf, bins=50, density=True)
    plt.xlabel('Différences de profondeurs [m]')
    plt.ylabel('Densité')
    axes[1,2].set_title('gmt, {}'.format(retrieve_name(diff_surf)))
    plt.suptitle("Histogrammes méthodes griddata ")
    plt.savefig('comparaison_interpolateur/Histogrammes.png')
    plt.show()

    # Carte des différences
    difference_n = interpolation_near - sol
    difference_l = interpolation_lin - sol
    difference_c = interpolation_cub - sol
    diff_tri = inter_tri.values - sol
    diff_neighbor = inter_neigbour - sol
    diff_surface = inter_surface - sol
    cmap = 'bwr'
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    # Normalisation avec définition du demi-intervalle
    normalize1 = CenteredNorm(0, halfrange=1)
    c1 = ax[0, 0].pcolor(X, Y, diff_tri, cmap=cmap, norm=normalize1)
    ax[0, 0].set(title='Différences triangulation')
    plt.colorbar(c1, label='Différences profondeurs [m]', ax=ax[0, 0])
    c2 = ax[0, 1].pcolor(X, Y, diff_neighbor, cmap=cmap, norm=normalize1)
    ax[0, 1].set(title='Différences plus proche voisin')
    plt.colorbar(c2, label='Différences profondeurs [m]', ax=ax[0, 1])
    c31 = ax[0, 2].pcolor(X, Y, diff_surface, cmap=cmap, norm=normalize1)
    ax[0, 2].set(title='Différences surface')
    plt.colorbar(c31, label='Différences profondeurs [m]', ax=ax[0, 2])
    c12 = ax[1,0].pcolor(X, Y, difference_n, cmap=cmap, norm=normalize1)
    ax[1,0].set(title='Différences nearest')
    plt.colorbar(c12, label='Différences profondeurs [m]', ax=ax[1, 0])
    c22 = ax[1,1].pcolor(X, Y, difference_l, cmap=cmap, norm=normalize1)
    ax[1,1].set(title='Différences linear')
    plt.colorbar(c22, label='Différences profondeurs [m]', ax=ax[1, 1])
    c32 = ax[1, 2].pcolor(X, Y, difference_c, cmap=cmap, norm=normalize1)
    ax[1,2].set(title='Différences cubic')
    plt.colorbar(c32, label='Différences profondeurs [m]', ax=ax[1,2])
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
