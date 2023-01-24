import pandas as pd

def convertDate(x):
    return pd.datetime.strptime(x, '%Y%m%d')


if __name__ == '__main__':
    NomFichierDataShom = 'D:/COURS/UV4.4/3_2021.txt'
    NomFichierOut = 'D:/COURS/UV4.4/Brest_5Mai21.tid'

    TideFile_In = pd.read_csv(NomFichierDataShom, sep=';', usecols=[0,1], names=['dateFr','hauteur'], skiprows=14,
                              parse_dates={'datetime': ['dateFr']})

    TideFile_In['date'] = TideFile_In['datetime'].apply(lambda x: x.strftime('%Y/%m/%d'))
    TideFile_In['time'] = TideFile_In['datetime'].apply(lambda x: x.strftime('%H:%M:%S'))

    print('nb valeurs lues : ', TideFile_In.shape[0])

    with open(NomFichierOut, 'w') as fichier:
        fichier.write('--------\n')
    fichier.close()

    TideFile_In.to_csv(NomFichierOut, mode='a', header=False, index=False, sep=' ', columns=['date','time','hauteur'])
