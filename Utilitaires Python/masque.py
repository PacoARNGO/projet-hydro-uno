import geopandas
import pandas
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    SHP = 'zone3_pour_lignesOBSTRN(A).shp'
    SBES = 'sbes_ch_0_xyz.txt'

    # Lecture du polygone en shp
    poly = geopandas.read_file(SHP)
    
    # Lecture des sondes xyz avec pandas
    # Il est également possible d'utiliser np.loadtxt
    p0 = pandas.read_csv(SBES, delim_whitespace=True, names=['x', 'y', 'z'])

    # Grille d'interpolation
    x = np.arange(int(p0.x.min()), int(p0.x.max())+1)
    y = np.arange(int(p0.y.min()), int(p0.y.max())+1)
    X, Y = np.meshgrid(x, y)
    # Interpolation linéaire
    Zl = griddata(p0[['x', 'y']], p0.z, (X, Y), method='linear')

    # Transformation des points en géométrie
    geo_pts = geopandas.points_from_xy(x=X.flatten(), y=Y.flatten())
    # Créatiion d'un DataFrame à partir des points
    df = pandas.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Zl.flatten()})
    # Création d'un GeoDataFrame à partir du dataframe et de la géométrie
    gdf = geopandas.GeoDataFrame(geometry=geo_pts, data=df)
    gdf.set_crs(epsg=2154, inplace=True)
    # Calcul des indices non masqués
    interieur = gdf.within(poly.at[0, 'geometry'])
    # Remise du masque sous forme de matrice
    mask = interieur.values.reshape(X.shape)
    # Passer les valeurs hors du masque à NaN
    Zl[~mask] = np.nan
    
    # Affichage de l'interpolation masquée
    plt.contourf(X, Y, Zl)
    plt.show()
