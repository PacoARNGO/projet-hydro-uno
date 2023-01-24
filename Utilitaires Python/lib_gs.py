#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
#    TP - Introduction à l'interpolation spatiale et aux géostatistiques #
##########################################################################

# P. Bosser / ENSTA Bretagne
# Version du 24/02/2021


# Numpy
import numpy as np
# Matplotlib / plot
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def plot_hist(z_obs, xlabel = "", ylabel = "", title = "", fileo = "", bins= tab):
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
    plt.figure()
    sns.histplot(z_obs, kde = False, bins = tab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if fileo != "":
        plt.savefig(fileo)



################## Modèle de fonction d'interpolation ##################

def interp_xxx(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par ???
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    z_int = np.nan*np.zeros(x_int.shape)
    #
    # ...
    #
    return z_int

####################### Fonctions d'interpolation ######################

def interp_inv(x_obs, y_obs, z_obs, x_grd, y_grd, p):
    """


        Parameters
        ----------
        x_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                coordonnées en Est des observations
        y_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                coordonnées en Nord des observations
        z_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                altitudes des observations
        x_grd :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Tableau à deux dimensions.
                coordonnées en Est des points que l'on souhaite interpoler.
        y_grd :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Tableau à deux dimensions.
                coordonnées en Est des points que l'on souhaite interpoler.
        p :
            TYPE, optional :
                float
            DESCRIPTION :
                facteur par lequel on multiplie l'inverse de la distance du point interpolé aux observations pour pondérer les observations.

        Returns
        -------
        z_interpoles :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Altitudes des points dont les coordonnées sont comprises dans x_grd et y_grd.
                L'interpolation par inverse des distances est utilisée.


        """
    # Interpolation par inversion des distances
    # Taille de la grille qu'on souhaite interpoler
    (m, n) = x_grd.shape
    # Liste des z interpoles
    z_interpoles = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            # Coordonnées du kième point de la grille
            x0 = x_grd[i][j];
            y0 = y_grd[i][j]
            v0 = np.array([x0, y0])
            # Numérateur et dénominateur de la formule de la moyenne pondérée
            S = 0;
            somme_poids = 0
            # Parcourt des observations
            for k in range(len(z_obs)):
                # Récupération de la valeur du kième point
                xk = x_obs[k][0];
                yk = y_obs[k][0];
                zk = z_obs[k][0]
                vk = np.array([xk, yk])
                # Distance entre le point à interpoler et le kième point
                d = np.linalg.norm(v0 - vk)
                # Poids à associer au point k
                poids = 1 / d ** p
                S += poids * zk
                somme_poids += poids
            z0 = S / somme_poids
            # Remplissage du tableau des altitudes interpolées
            z_interpoles[i][j] = z0
    return z_interpoles

def interp_lin(x_obs, y_obs, z_obs, x_grd, y_grd):
    """


        Parameters
        ----------
        x_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                coordonnées en Est des observations
        y_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                coordonnées en Nord des observations
        z_obs :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                altitudes des observations
        x_grd :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Tableau à deux dimensions.
                coordonnées en Est des points que l'on souhaite interpoler.
        y_grd :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Tableau à deux dimensions.
                coordonnées en Est des points que l'on souhaite interpoler.

        Returns
        -------
        z_interpoles :
            TYPE :
                numpy.ndarray
            DESCRIPTION :
                Altitudes des points dont les coordonnées sont comprises dans x_grd et y_grd.
                L'interpolation linéaire est utilisée.

        """
    assert len(x_obs) == len(y_obs) == len(z_obs)
    assert x_grd.shape == y_grd.shape
    # Taille de la grille qu'on souhaite interpoler
    (m, n) = x_grd.shape
    # Calcul de la triangulation à partir de coordonnées (x,y)
    tri = Delaunay(np.hstack((x_obs, y_obs)))
    # Liste des z interpoles
    z_interpoles = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            # Coordonnées du kième point de la grille
            x0 = x_grd[i][j];
            y0 = y_grd[i][j]
            # Recherche de l’index du triangle contenant le point x0, y0
            idx_t = tri.find_simplex(np.array([x0, y0]))
            if idx_t == -1:
                # La valeur par défaut quand le point n'est pas compris dans un triange
                # est trouvée par interpolation par plus proche voisin
                z0 = interp_ppv(x_obs, y_obs, z_obs, np.array([[x0]]), np.array([[y0]]))[0][0]
                z_interpoles[i][j] = z0
            else:
                # Recherche des index des sommets du triangle contenant le point x0, y0
                idx_s = tri.simplices[idx_t, :]
                # Coordonnées des sommets du triangle contenant le point x0, y0
                x1 = x_obs[idx_s[0]][0];
                y1 = y_obs[idx_s[0]][0];
                z1 = z_obs[idx_s[0]][0]
                x2 = x_obs[idx_s[1]][0];
                y2 = y_obs[idx_s[1]][0];
                z2 = z_obs[idx_s[1]][0]
                x3 = x_obs[idx_s[2]][0];
                y3 = y_obs[idx_s[2]][0];
                z3 = z_obs[idx_s[2]][0]
                # Mise du système de trois équations sous forme matricielle
                M = np.array([[x1, y1, 1],
                              [x2, y2, 1],
                              [x3, y3, 1]])
                Z = np.array([[z1],
                              [z2],
                              [z3]])
                # Calcul de la valeur interpolée par inversion du système de 3 équations.
                inv_M = np.linalg.inv(M)
                V = np.dot(inv_M, Z)
                [[a], [b], [c]] = V
                z0 = a * x0 + b * y0 + c
                # Remplissage du tableau des altitudes interpolées
                z_interpoles[i][j] = z0
    return z_interpoles

def interp_ppv(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par plus proche voisin
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    z_int = np.nan*np.zeros(x_int.shape)
    for i in np.arange(0,x_int.shape[0]):
        for j in np.arange(0,x_int.shape[1]):
            z_int[i,j] = z_obs[np.argmin(np.sqrt((x_int[i,j]-x_obs)**2+(y_int[i,j]-y_obs)**2))]
    return z_int


def interp_sfc(x_obs, y_obs, z_obs, x_grd, y_grd, p=2):
    """


    Parameters
    ----------
    x_obs :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            coordonnées en Est des observations
    y_obs :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            coordonnées en Nord des observations
    z_obs :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            altitudes des observations
    x_grd :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            Tableau à deux dimensions.
            coordonnées en Est des points que l'on souhaite interpoler.
    y_grd :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            Tableau à deux dimensions.
            coordonnées en Est des points que l'on souhaite interpoler.
    p :
        TYPE, optional :
            float
        DESCRIPTION :
            puissance de la surface de tendance utilisée

    Returns
    -------
    z_interpoles :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            Altitudes des points dont les coordonnées sont comprises dans x_grd et y_grd.
            L'interpolation par surface de tendance est utilisée.


    """
    # Interpolation par inversion des distances
    # Taille de la grille qu'on souhaite interpoler
    (m, n) = x_grd.shape
    # Liste des z interpoles
    z_interpoles = np.zeros((m, n))
    # Création des matrices utilisées lors des moindres carrés
    # nb_obs est le nombre d'observations
    nb_obs = z_obs.shape[0]
    # print('nb_obs : {}'.format(nb_obs))
    # nb_par est le nombre de paramètres
    nb_par = int((p + 1) * (p + 2) / 2)
    # print('nb_par : {}'.format(nb_par))
    # Création de la matrice A
    A = np.zeros((nb_obs, nb_par))
    # Remplissage de la matrice A
    k = 0
    for i in range(0, p + 1):
        for j in range(0, p - i + 1):
            for obs in range(nb_obs):
                x = x_obs[obs][0]  # [0] pour avoir un flottant et non un array comportant un flottant
                y = y_obs[obs][0]
                A[obs][k] = x ** i * y ** j
            # print(i, j, k)
            k += 1
    # print(A)
    # La matrice B des moindres carrés est la matrice z_obs
    X_mc = moindres_carres(A, z_obs)
    # print(X_mc)
    # print("Fin de l'exécution de l'algorithme des moindres carrés")
    for ix in range(m):
        for jy in range(n):
            x = x_grd[ix][jy]
            y = y_grd[ix][jy]
            z = 0
            kalpha = 0
            for ialpha in range(0, p + 1):
                for jalpha in range(0, p - ialpha + 1):
                    alpha = X_mc[kalpha]
                    # print(ialpha, jalpha, kalpha)
                    z += alpha * x ** ialpha * y ** jalpha
                    kalpha += 1
            z_interpoles[ix][jy] = z
    # print('*')
    return z_interpoles


def interp_spline(x_obs, y_obs, z_obs, x_grd, y_grd, ro=10 ** 3):
    """


    Parameters
    ----------
    x_obs :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            coordonnées en Est des observations
    y_obs :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            coordonnées en Nord des observations
    z_obs :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            altitudes des observations
    x_grd :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            Tableau à deux dimensions.
            coordonnées en Est des points que l'on souhaite interpoler.
    y_grd :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            Tableau à deux dimensions.
            coordonnées en Est des points que l'on souhaite interpoler.
    p :
        TYPE, optional :
            float
            La valeur par défaut est 10**3.
        DESCRIPTION :
            paramètre de lissage.

    Returns
    -------
    z_interpoles :
        TYPE :
            numpy.ndarray
        DESCRIPTION :
            Altitudes des points dont les coordonnées sont comprises dans x_grd et y_grd.
            L'interpolation par splines des distances est utilisée.


    """
    assert len(x_obs) == len(y_obs) == len(z_obs)
    assert x_grd.shape == y_grd.shape

    def phi(h):
        norme_h = np.linalg.norm(h)
        return norme_h ** 2 * np.log(norme_h)

    nb_obs = len(x_obs)
    # Taille de la grille qu'on souhaite interpoler
    (m, n) = x_grd.shape
    # Liste des z interpoles
    z_interpoles = np.zeros((m, n))
    # Remplissage de la matrice à inverser
    M = np.zeros((nb_obs + 3, nb_obs + 3))
    for k in range(nb_obs):
        M[k, 0] = 1
        M[k, 1] = x_obs[k][0]
        M[k, 2] = y_obs[k][0]
        M[nb_obs][k + 3] = 1
        M[nb_obs + 1][k + 3] = x_obs[k][0]
        M[nb_obs + 2][k + 3] = y_obs[k][0]
    for i in range(nb_obs):
        for j in range(nb_obs):
            if i == j:
                M[i][j + 3] = ro
            else:
                M[i][j + 3] = phi(np.array([x_obs[i] - x_obs[j], y_obs[i] - y_obs[j]]))
    # Inversion de la matrice
    # Nous effectuons cette opération ici pour n'avoir à la faire qu'une fois
    inv_M = np.linalg.inv(M)
    # Remplissage du vecteur colonne
    Z = np.zeros((nb_obs + 3, 1))
    for k in range(nb_obs):
        Z[k] = z_obs[k][0]
    # Calcul des coefficients de l'estimation
    V = np.dot(inv_M, Z)
    a0 = V[0][0]
    a1 = V[1][0]
    a2 = V[2][0]
    for i in range(m):
        for j in range(n):
            # Coordonnées du kième point de la grille
            x0 = x_grd[i][j];
            y0 = y_grd[i][j]
            # Calcul de la valeur interpolée
            z0 = a0 + a1 * x0 + a2 * y0
            for k in range(nb_obs):
                bk = V[k + 3]
                z0 += bk * phi(np.array([x0 - x_obs[k], y0 - y_obs[k]]))
            # Remplissage du tableau des altitudes interpolées
            z_interpoles[i][j] = z0[0]
    return z_interpoles


def moindres_carres(A, B):
    return np.dot(np.linalg.pinv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), B))


############################# Visualisation ############################

def plot_contour_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé du champ interpolé sous forme d'isolignes
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    plt.contour(x_grd, y_grd, z_grd_m, int(np.round((np.max(z_grd_m)-np.min(z_grd_m))/4)),colors ='k')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        plt.xlim(0.95*np.min(x_obs),np.max(x_obs)+0.05*np.min(x_obs))
        plt.ylim(0.95*np.min(y_obs),np.max(y_obs)+0.05*np.min(y_obs))
    else:
        plt.xlim(0.95*np.min(x_grd),np.max(x_grd)+0.05*np.min(x_grd))
        plt.ylim(0.95*np.min(y_grd),np.max(y_grd)+0.05*np.min(y_grd))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_surface_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), minmax = [0,0], xlabel = "", ylabel = "", zlabel = "", title = "", fileo = ""):
    # Tracé du champ interpolé sous forme d'une surface colorée
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # minmax : valeurs min et max de la variable interpolée (facultatif)
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    from matplotlib import cm
    
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    if minmax[0] < minmax[-1]:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cm.terrain, vmin = minmax[0], vmax = minmax[-1], shading = 'auto')
    else:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cm.terrain, shading = 'auto')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        plt.xlim(0.95*np.min(x_obs),np.max(x_obs)+0.05*np.min(x_obs))
        plt.ylim(0.95*np.min(y_obs),np.max(y_obs)+0.05*np.min(y_obs))
    else:
        plt.xlim(0.95*np.min(x_grd),np.max(x_grd)+0.05*np.min(x_grd))
        plt.ylim(0.95*np.min(y_grd),np.max(y_grd)+0.05*np.min(y_grd))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_points(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    fig = plt.figure()
    ax = plt.gca()
    plt.plot(x_obs, y_obs, 'ok', ms = 4)
    ax.set_xlim(0.95*min(x_obs),max(x_obs)+0.05*min(x_obs))
    ax.set_ylim(0.95*min(y_obs),max(y_obs)+0.05*min(y_obs))
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_patch(x_obs, y_obs, z_obs, xlabel = "", ylabel = "", zlabel = "", title = "", fileo = ""):
    # Tracé des valeurs observées
    # x_obs, y_obs, z_obs : observations
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    from matplotlib import cm
    
    fig = plt.figure()
    p=plt.scatter(x_obs, y_obs, marker = 'o', c = z_obs, s = 80, cmap=cm.terrain)
    plt.xlim(0.95*min(x_obs),max(x_obs)+0.05*min(x_obs))
    plt.ylim(0.95*min(y_obs),max(y_obs)+0.05*min(y_obs))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_triangulation(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé de la triangulation sur des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    from scipy.spatial import Delaunay as delaunay
    tri = delaunay(np.hstack((x_obs,y_obs)))
    
    plt.figure()
    plt.triplot(x_obs[:,0], y_obs[:,0], tri.simplices)
    plt.plot(x_obs, y_obs, 'or', ms=4)
    plt.xlim(0.95*min(x_obs),max(x_obs)+0.05*min(x_obs))
    plt.ylim(0.95*min(y_obs),max(y_obs)+0.05*min(y_obs))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
