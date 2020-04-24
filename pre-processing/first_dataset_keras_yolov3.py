import os


file_dir = input("Quel Dossier voulez-vous charger ? ")

if not os.path.exists(file_dir):
    print(f"Le dossier {file_dir} n'existe pas.")

else:
    # Dans le dossier contenant les labels
    for filename in os.listdir(file_dir):
        filename_no_ext = os.path.splitext(filename)[0]

        drone = filename.split(".")[0]
        morning = filename.split(".")[1]

        print(f"Convertion du fichier : {filename}")
        # Lecture du fichier.txt avec les labels
        with open(os.path.join(file_dir, filename)) as f:
            content = f.readlines()

            # Ecriture du fichier train.txt a compléter
            with open (os.path.join("./label_me_txt/", f"train.txt"), "+a") as txt_f:

                old_frame = -1
                # Pour chaque ligne dans le fichier de label
                for first_line in content:
                    frame = int(first_line.split(" ")[5])

                    if old_frame > frame:
                        break

                    # Toutes les x frames
                    x = 10
                    if frame % x == 0:
                        if drone == "1":
                            if morning == "1":
                                txt_f.write(f"../datasets/Drone1/Morning/{filename_no_ext}/{frame}.jpg ")
                            else:
                                txt_f.write(f"../datasets/Drone1/Noon/{filename_no_ext}/{frame}.jpg ")
                        else:
                            if morning == "1":
                                txt_f.write(f"../datasets/Drone2/Morning/{filename_no_ext}/{frame}.jpg ")
                            else:
                                txt_f.write(f"../datasets/Drone2/Noon/{filename_no_ext}/{frame}.jpg ")

                        xmin_list = []
                        ymin_list = []
                        xmax_list = []
                        ymax_list = []
                        # Pour chaque ligne test si la frame prise = a la frame de la ligne
                        for line in content:
                            new_frame = int(line.split(" ")[5])
                            if frame == new_frame:
                                xmin = int(float(line.split(" ")[1]) / 6.4)
                                ymin = int(float(line.split(" ")[2]) / 6.4)
                                xmax = int(float(line.split(" ")[3]) / 6.4)
                                ymax = int(float(line.split(" ")[4]) / 6.4)
                                
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
                    
                    old_frame = frame
