import numpy as np
import pygmt
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

'''chargement du fichier monofaisceau'''
FILENAME = 'data/ea400_38kilo.txt'
data = np.loadtxt(FILENAME,delimiter=',')

'''Tracer des nuages de points chargés'''
sc = plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='gist_earth')
plt.title('Sondes')
plt.colorbar(sc, label='Hauteur [m]')
plt.xlabel('Est [m]')
plt.ylabel('Nord [m]')
plt.show()

'''Utilisation de PyGMT.info pour récuperation d'info important'''
# print('Info:',pygmt.info(data))
# renvoie nbr de pts, min et max de chaque colonnes/coord

'''Suppression de l'affichage en mode scientifique donc on arrondit car RTK précision cm donc pas besoin d'autant e précision'''
np.set_printoptions(suppress=True)
# pour pouvoir lire plus facilement les valeurs

'''Affichage des 10 premières lignes de data'''
# print(data[:10])

'''Comparaison avec données multifaisceaux'''
data_m = np.loadtxt('data/PortCommercePasseSante_Compile_2017a2022_GPSTide_30cm.xyz')
# print(data_m[:10])

'''Comparaison des données SBES'''
# print('Info MBES:',pygmt.info(data_m))
bornes_MBES = pygmt.info(data_m, spacing=0.3)  # pas de grille 30 cm
# print('region:',bornes_MBES)

bornes_SBES = pygmt.info(data, spacing=1)

grid_MBES = pygmt.xyz2grd(data_m, spacing=(0.3, 0.3), region=bornes_MBES)
# changement de format, on a des coord en xyz et on les passe sous forme de grille
# print(grid_MBES)

output_dataframe = pygmt.grdtrack(points=data, grid=grid_MBES)
#  0 = X / 1 = Y / 2 = multi / 3 = Z_200 / 4  = Z_38 / 5 = Z_proc200
# prend les colonnes x et y de multi et associe le z du mono
# print(output_dataframe.info)

output_dataframe['différence profondeur']  = output_dataframe[2] - output_dataframe[3]
#print(output_dataframe['différence profondeur'])

'''Visualisation des écarts entre les sondes SBES et modèle MBES '''
sc = plt.scatter(output_dataframe[0], output_dataframe[1], c=output_dataframe['différence profondeur'], cmap='seismic',
                 vmax=1, vmin=-1)
plt.title('écarts entre les sondes SBES et modèle MBES')
plt.colorbar(sc, label='Profondeur [m]')
plt.xlabel('Est [m]')
plt.ylabel('Nord [m]')
plt.show()

'''Visualisation des écarts entre les sondes SBES et modèle MBES avec un histogramme'''
diff_SBES_MBES = output_dataframe['différence profondeur']
plt.hist(diff_SBES_MBES, bins=50)
plt.title('écarts entre les sondes SBES et modèle MBES')
plt.show()

'''Interpolation des données monofaisceaux'''
# triangulation
interp_triangulaire = pygmt.triangulate.regular_grid(data=data, spacing=1, region=bornes_SBES)
interp_triangulaire2 = interp_triangulaire.values[~np.isnan(
    interp_triangulaire.values)]  # on enlève les valeurs nan !!! uniquement pour tracer boxplot et histogramme !!!

# print(interp_triangulaire)

# nearneighbour
interp_nearneighbor = pygmt.nearneighbor(data, spacing=1, region=bornes_SBES, search_radius=30)
interp_nearneighbor2 = interp_nearneighbor.values[~np.isnan(interp_nearneighbor.values)]

# surface
data_mean = pygmt.blockmean(data, spacing=1, region=bornes_SBES)
interp_surface = pygmt.surface(data_mean, spacing=1, region=bornes_SBES, T=0.5)
interp_surface2 = interp_surface.values[~np.isnan(interp_surface.values)]

'''Visualisation des donées interpolées aux monofaisceaux'''
plt.pcolormesh(interp_triangulaire.x, interp_triangulaire.y, interp_triangulaire.values, cmap='gist_earth')
plt.colorbar(label='Profondeur [m]')
plt.contourf(interp_triangulaire.x, interp_triangulaire.y, interp_triangulaire.values, color='black')
plt.title('Interpolation par triangulation')
plt.xlabel('Est [m]')
plt.ylabel('Nord [m]')

plt.figure()
plt.pcolormesh(interp_nearneighbor.x, interp_nearneighbor.y, interp_nearneighbor.values, cmap='gist_earth',
               shading='flat')
plt.contour(interp_surface.x, interp_surface.y, interp_surface.values, color='black')
plt.colorbar(label='Profondeur [m]')
plt.xlabel('Est [m]')
plt.ylabel('Nord [m]')
plt.title('Interpolation par voisinage proche')

plt.figure()
plt.pcolormesh(interp_surface.x, interp_surface.y, interp_surface.values, cmap='gist_earth', shading='flat')
plt.contour(interp_surface.x, interp_surface.y, interp_surface.values, color='black')  # ligne de niveau
plt.title('Interpolation surface')
plt.colorbar(label='Profondeur [m]')
plt.xlabel('Est [m]')
plt.ylabel('Nord [m]')
plt.show()

'''Boxplot et histogramme'''
grid_multi = pygmt.grdsample(grid=grid_MBES, region=bornes_SBES, spacing=1)

grid_multi1 = grid_multi.values[~np.isnan(grid_multi.values)]
grid_multi1 = grid_multi1[np.shape(interp_triangulaire2)[0]]

grid_multi2 = grid_multi.values[~np.isnan(interp_nearneighbor.values)].flatten()
grid_multi2 = grid_multi2[np.shape(interp_nearneighbor2)[0] - 1]
# print(np.shape(grid_multi2),np.shape(interp_nearneighbor2))

grid_multi3 = grid_multi.values[~np.isnan(interp_surface.values)].flatten()
grid_multi3 = grid_multi3[np.shape(interp_surface2)[0] - 1]

diff_tri = interp_triangulaire - grid_multi
diff_nearneighbor = interp_nearneighbor - grid_multi
diff_surface = interp_surface - grid_multi

f, ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax[0].hist(diff_tri.values.flatten(), bins=50, density=True)
ax[0].set(title='triangulation')
ax[1].hist(diff_nearneighbor.values.flatten(), bins=50, density=True)
ax[1].set(title='nearneighbor \n radius=100m \n secteurs=4')
ax[2].hist(diff_surface.values.flatten(), bins=50, density=True)
ax[2].set(title='surface \n T=0.5')
plt.suptitle('écarts entre interpolation et multifaisceau')
plt.show()

diff_tri_sans_nan = diff_tri.values[~np.isnan(diff_tri.values)]

'''plt.figure()
names = ['triangulation','nearneighbor','surface']
plt.boxplot([diff_tri,diff_nearneighbor,diff_surface],labels=names)
plt.title("comparaison des boxplots des méthodes d'interpolations avec les données multifaisceaux")
plt.show()'''


def statistiques(val_interp, sol):
    ei = val_interp - sol
    me = np.mean(ei)
    stde = np.std(ei)
    rmse = np.sqrt(np.sum(ei ** 2) / len(ei))
    return me, stde, rmse


me, stde, rmse = statistiques(interp_triangulaire2, grid_multi1)
me, stde, rmse = statistiques(interp_nearneighbor2, grid_multi2)
me2, stde2, rmse2 = statistiques(interp_surface2, grid_multi3)

print('statistiques sur nearneighbor:')
print(f'Erreur moyenne : {me:.3f}m')
print(f'Écart-type : {stde:.3f}m')
print(f'RMSE : {rmse:.3f}m')

print('statistiques sur surface:')
print(me2, stde2, rmse2)
