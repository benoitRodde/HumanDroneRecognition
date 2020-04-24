# Objectif : redimensionner en masse des images dans un dossier, avant utilisation pour blog ou partage par email par exemple
# TODO : fix si fichier non image présent dans le dossier (genre .directory), fait mais hard codé sur terminaison fichier image en "G" ou "g"

from PIL import Image
from os import listdir
from os.path import isfile, join, isdir

mypath = input("Dans quel dossier sont les images ? ")
target = int(input("Dimension maximum voulue (ex 1000) : "))


def search_in_folder(path):
    for folder in listdir(mypath):
        print(join(path,folder))
        if isdir(join(path,folder)):
            search_in_folder(join(path,folder))
        else: 
            print('isnot a dir')
            change_file(path)


def change_file(path):
    print('path', path)
    imageFiles = [ f for f in listdir(path) if isfile(join(path,f)) and (f.endswith("G") or f.endswith("g")) ]
    for im in imageFiles :
        im1 = Image.open(join(path,im))
        print('join(path,im)', join(path,im))
        originalWidth, originalHeight = im1.size
        ratio = originalWidth / originalHeight
        if ratio > 1 :
            width = target
            height = int(width / ratio)
        else :
            height = target
            width = int(height * ratio)

        im2 = im1.resize((width, height), Image.ANTIALIAS) # linear interpolation in a 2x2 environment
        im2.save(join(path, "".join(im)))
        print (im, "redimensionnée…")
    print ("Travail terminé !", len(imageFiles), "images redimensionnées.")

  
search_in_folder(mypath)
