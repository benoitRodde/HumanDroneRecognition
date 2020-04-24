import os
import xmltodict


def _writing_file(txt_f, obj):
    x0 = int(list(obj["polygon"]["pt"][1]['x'])[0])
    y0 = int(list(obj["polygon"]["pt"][3]['y'])[0])
    
    x1 = int(list(obj["polygon"]["pt"][0]['x'])[0])
    y1 = int(list(obj["polygon"]["pt"][0]['y'])[0])

    if x0 < x1:
        xmax = x1
        xmin = x0
    else:
        xmax = x0
        xmin = x1

    if y0 < y1:
        ymax = y1
        ymin = y0
    else:
        ymax = y0
        ymin = y1

    txt_f.write(f"{xmin}")
    txt_f.write(f",{ymin}")
    txt_f.write(f",{xmax}")
    txt_f.write(f",{ymax}")
    txt_f.write(f",0 ")


xml_dir = input("Quel Dossier voulez-vous charger ? ")
for filename in os.listdir(xml_dir):
    with open(os.path.join(xml_dir, filename)) as xml_f:
        elem = xmltodict.parse(xml_f.read())

    if not os.path.exists("./label_me_txt"):
        os.makedirs("./label_me_txt")

    filename_jpg = os.path.splitext(filename)[0] + ".jpg"
    with open (os.path.join("./label_me_txt/", "train.txt"), "+a") as txt_f:
        print(f"Labelisation de l'image : {filename_jpg}")
        txt_f.write(f"../datasets/Semantic_dataset/images/{filename_jpg} ")

        if 'object' in elem["annotation"]:
            if isinstance(elem["annotation"]["object"], list):
                for obj in elem["annotation"]["object"]:
                    for pt in obj["polygon"]["pt"]:
                        pt['x'] = {round(int(pt['x']) / 10)}
                        pt['y'] = {round(int(pt['y']) / 10)}
                    
                    _writing_file(txt_f, obj)

            else:
                for pt in elem["annotation"]["object"]["polygon"]["pt"]:
                    pt['x'] = {round(int(pt['x']) / 10)}
                    pt['y'] = {round(int(pt['y']) / 10)}

                _writing_file(txt_f, elem["annotation"]["object"])
        
        txt_f.write(f"\n")

        txt_f.close()