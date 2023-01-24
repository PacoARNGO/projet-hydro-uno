import os
import sys


def process(filename):
    # Informations à xtraire dans le header
    header = ['SiteInfo', 'Latitude', 'Longitude']
    res = {}
    data = []
    # Lecture fi fichier ligne par ligne
    with open(filename) as fin:
        while True:
            line = fin.readline().strip()
            # Sortir de la première boucle lorsqu'on arrive sur DATA
            if line == '[DATA]':
                break
            val = line.split('=')
            if val[0] in header:
                res[val[0]] = val[1]
        # Récupération des noms de colonnes
        columns = fin.readline().strip().split('\t')
        # Ligne à passer
        _ = fin.readline()
        # Lecture des données
        for line in fin:
            data.append(line.strip().split('\t'))
    return res, columns, data


def print_res(dirname, fic, header, columns, data):
    """Écriture du fichier .csv avec les données reformatées
    """
    fic_csv = os.path.splitext(fic)[0] + '.csv'
    filename = os.path.join(dirname, fic_csv)
    with open(filename, 'w') as fout:
        # Fusion des noms de colonnes
        print(','.join(list(header.keys()) + columns), file=fout)
        for line in data:
            # Fusion des données
            print(','.join(list(header.values()) + line), file=fout)

            
if __name__ == '__main__':
    """Conversion des fichiers .vp2 en .csv
    Les données initiales doivent être dans un répertoire vp2
    Le répertoire csv doit exister. Il contiendra le résultat de la conversion.
    """
    # Répertoire contenant les fichiers .vp2
    DIRNAME_IN = 'vp2'
    # Répertoire contenant les fichiers résultat .csv
    DIRNAME_OUT = 'csv'

    # Parcours de tous les fichiers dans vp2
    for root, _, files in os.walk(DIRNAME_IN):
        for fic in files:
            filename = os.path.join(root, fic)
            head, col, dat = process(filename)
            print_res(DIRNAME_OUT, fic, head, col, dat)
