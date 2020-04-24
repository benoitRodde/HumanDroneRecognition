import os


file_dir = input("Quel Dossier voulez-vous charger ? ")

img_hauteur = input("Hauteur de l'image ? ")
img_longueur = input("Longueur de l'image ? ")

if not os.path.exists(file_dir):
    print(f"Le dossier {file_dir} n'existe pas.")

else:
    # Dans le dossier contenant les labels des frames
    for filename in os.listdir(file_dir):
        filename_no_ext = os.path.splitext(filename)[0]

        print(f"Convertion du fichier : {filename}")
        # Lecture du fichier.txt avec les labels
        with open(os.path.join(file_dir, filename)) as f:
            content = f.readlines()

            # Ecriture du fichier train.txt a compléter
            with open (os.path.join("./label_me_txt/", f"train-Anice51.txt"), "+a") as txt_f:

                old_filename_no_ext = ""
                # Pour chaque ligne dans le fichier de label
                for first_line in content:
                    vid_name = os.path.basename(os.path.dirname(file_dir))
                    
                    if filename_no_ext in old_filename_no_ext:
                        break
                    
                    old_filename_no_ext = filename_no_ext

                    txt_f.write(f"../datasets/youtube/perso/{filename_no_ext}.jpg ")

                    xmin_list = []
                    ymin_list = []
                    xmax_list = []
                    ymax_list = []

                    # Pour chaque ligne test si la frame prise = a la frame de la ligne
                    for line in content:
                        x_center = int(float(line.split(" ")[1]) * int(img_longueur))
                        y_center = int(float(line.split(" ")[2]) * int(img_hauteur))

                        width = int(float(line.split(" ")[3]) * int(img_longueur))
                        height = int(float(line.split(" ")[4])* int(img_hauteur))
        
                        xmin = x_center - (width / 2)
                        ymin = y_center - (height / 2)
                        xmax = x_center + (width / 2)
                        ymax = y_center + (height / 2)

                        xmin_list.append(xmin)
                        ymin_list.append(ymin)
                        xmax_list.append(xmax)
                        ymax_list.append(ymax)

                    # print(xmin_list)
                    for i in range(len(xmin_list)):
                        txt_f.write(f"{xmin_list[i]}")
                        txt_f.write(f",{ymin_list[i]}")
                        txt_f.write(f",{xmax_list[i]}")
                        txt_f.write(f",{ymax_list[i]}")
                        txt_f.write(f",0 ")

                    txt_f.write(f"\n")
                    