import pygmt
import numpy as np
from pyproj import Proj
import matplotlib.pyplot as plt
import seaborn as sns

'''ATTENTION : l'origine (0,0) se trouve en bas à droite'''

if __name__ == '__main__':
    # Définition du format A4 en portrait et en paysage
    region_a4_paysage = [0, 29.7, 0, 21]
    region_a4_portrait = [0, 21, 0, 29.7]
    # Projection identité
    proj_a4 = 'x1' #marche pas
    # Projection Lambert93 en RGF93v1
    proj = Proj('epsg:2154')

    # TODO : fichier SBES à utiliser
    FILE_SBES = r'C:\Users\manyl\Documents\2ALeema\Projet\Donnee_SBES/ea400_38kilo.txt'
    # TODO : choisir une orientation
    region_a4 = region_a4_paysage
    # TODO : ajuster la projection des données SBES
    # Ici on fixe la largeur de l'image à 10cm
    projection = 'X10/0'
    # TODO : Choix de la résolution du modèle interpolé
    pas = '1'
    #pas = '3' #pas de 3 m


    # Chargement des données
    sbes = np.loadtxt(FILE_SBES,delimiter=',',skiprows=1)

    sc = plt.scatter(sbes[:, 0], sbes[:, 1], c=sbes[:, 2], cmap='gist_earth')
    plt.title('Sondes')
    plt.colorbar(sc, label='Hauteur [m]')
    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.show()

    # TODO : définir la région xmin, xmax, ymin, ymax
    # Utiliser pygmt.info pour obtenir la région
    # calcul des bornes xmin,xmax,ymin,ymax
    region_sbes = pygmt.info(sbes, spacing=pas)
    # Création d'une figure pygmt
    fig = pygmt.Figure()
    # TODO : définir les paramètres x et y pour dessiner un contour à 1cm du bord de la page
    fig.plot(
        # coordonnées exprimées dans le repère de la page A4
        region=region_a4,
        projection=proj_a4,
        # Pour un rectangle, 5 valeurs pour x et 5 pour y #sachant que la figure fait 10 cm au total
        x=[1,27.7,27.7,1,1],
        y=[1,1,20,20,1],
        # Trait de 1 pixel d'epaisseur en noir
        pen="2p,black"
    )
    # TODO définir la position du texte
    # Éventuellement modifier la police utilisée. Liste des polices : 
    # https://docs.generic-mapping-tools.org/latest/cookbook/postscript-fonts.html

    fig.text(text="evé bathymétrique au monofaisceu du passe de la santé OUEST, zone 1",
             #position du titre
             x=14.5, y=19,
             # Police en 20 points en mode gras
             # Uniquement pour le titre principal
             font='20p,Helvetica-Bold')

    fig.text(text="Groupe Beautemps-Baupré, 17/01/2023",
             # position du titre
             x=14.5, y=18,
             # Police en 20 points en mode gras
             # Uniquement pour le titre principal
             font='16p,Helvetica')

    #file = open('text.txt',"w")
    #file.write("Méthode d'interpolation : Plus proche voisin R = 100m et Secteur=8\nFréquence du sondeur : 200 kHz\nEllipsoïde de référence : GRS80\nSystème de référence : RGF93\nProjection : LAMBERT_93\nRésolution : 3m\nRéférence vertical : Zéro hydrographique")
    #file.close()

    '''Creation de la cartouche avec les informations concernant la projection etc'''
    fig.text(x=16, y=11, font="10p", justify="TL", text="Méthode d'interpolation : Plus proche voisin R = 25m et Secteur=8")
    fig.text(x=16, y=10, font="10p", justify="TL", text="Fréquence du sondeur : 200 kHz")
    fig.text(x=16, y=9, font="10p", justify="TL", text="Ellipsoïde de référence : GRS80")
    fig.text(x=16, y=8, font="10p", justify="TL", text="Système de référence : RGF93")
    fig.text(x=16, y=7, font="10p", justify="TL", text="Projection : LAMBERT_93")
    fig.text(x=16, y=6, font="10p", justify="TL", text="Résolution : 3m")
    fig.text(x=16, y=5, font="10p", justify="TL", text="Référence vertical : Zéro hydrographique")

    '''logo ENSTA'''
    #fig.image(imagefile='ENSTA-LogoH-NOIR.eps', )


    # TODO : choisir une palette de couleurs (ici ocean) #on l'ajuste de façon à ce qu'on aille de la valeur min à la valeur max de profondeur
    # et définir les bornes des profondeurs (series)
    # le paramètre reverse=True permet d'inverser la palette
    pygmt.makecpt(cmap="viridis", series=(sbes[:,2].min(),sbes[:,2].max(),1), reverse=True)
    # Afficher la palette de couleurs sur la page A4
    # TODO ajuster les coordonnées (ici x=20 y=10)
    #fig.shift_origin(xshift=20,yshift=2)
    # TODO ajuster la hauteur et la largeur de la palette (ici 10cm et 0.7cm)
    # Remarque : une hauteur négative affiche le maximum en bas (utile en bathymétrie)
    fig.colorbar(position="x17/2.5+w10c/1c+h",
                 # Annotations tous les 2m, trait tous les mètres
                 # Légende en x et légende en y
                 #a2f1 : annotations toutes les 2 (ici m) et tick marks in steps of one (ie graduation toutes les 1m)
                 frame=["a2f1", "x+lProfondeur", "y+lm"])

    #position = 'x20/10+w-10/0.7'
    # TODO : interpoler les données ; ajuster les paramètres
    inter_near = pygmt.nearneighbor(data=sbes, spacing=pas, region=region_sbes, search_radius=25, sectors=8)

    # Translation de l'origine avant de tracer le MNT
    fig.shift_origin(xshift=3.5, yshift=3) #donc MNT va être placer en x = 3.5 et y=2

    # Affichage du MNT ; utilise les constantes définies au début du main
    fig.grdimage(inter_near, region=region_sbes, projection=projection, frame='ag',nan_transparent=True) #nan_transparent=True permet d'avoir du blanc dans les endroits ou il n'y a pas de data sur le MNT
    # Superposition des lignes de niveau
    # TODO ajuster les valeurs de interval et annotation
    fig.grdcontour(grid=inter_near,
                   # Lignes de niveau tous les m
                   interval=1,
                   # Annotation des lignes tous les m
                   annotation=1,
                   # Affichage des lignes en blanc
                   pen='1p,white')

    #Nouvelle translation de l'origine
    fig.shift_origin(xshift=15, yshift=10)
    # Affichage du trait de côte
    fig.coast(
        # Définition de la projection Lambert93
        # TODO ajuster la taille de la figure (ici 9cm)
        projection='L3/46.5/49/44/5c',
        #centre de la projection (lon=3,lat=46.5), les parallèles toutes à droite et toute à gauche set = (lat =49 N et lat = 44 N)
        # Région utilisée pour le trait de côte. Autour de Brest.
        region=[-5, -4, 48.2, 48.7], #les coordonnées que tu peux retrouver en abscisses et ordonnées sur la figure
        # Choix des couleurs pour la terre et l'océan
        land='lightgray', water='lightblue',
        # Si possible utiliser la résolution maximale (full)
        resolution='f',
        # Afficher les coordonnées et la grille
        frame='afg')

    # Calcul des coordonnées moyennes
    pos = np.mean(sbes, axis=0)
    # Passer de Lambert93 à des coordonnées géodésiques (transformation inverse)
    lg = proj(pos[0], pos[1], inverse=True)
    # Afficher la position sous forme de disque rouge
    fig.plot(x=lg[0], y=lg[1], style="c.3c", fill="red3", pen="1p")

    fig.show()
