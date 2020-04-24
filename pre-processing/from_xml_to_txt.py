import os
import xmltodict


def _writing_file(txt_f, obj):
    x0 = int(list(obj["polygon"]["pt"][0]['x'])[0])
    y0 = int(list(obj["polygon"]["pt"][0]['y'])[0])
    x1 = int(list(obj["polygon"]["pt"][1]['x'])[0])
    y3 = int(list(obj["polygon"]["pt"][3]['y'])[0])

    width = abs(x0 - x1) / 600
    height = abs(y0 - y3) / 400
    X_center = abs(round((x0 + x1) / 2)) / 600
    y_center = abs(round((y0 + y3) / 2)) / 400

    txt_f.write("1")
    txt_f.write(f" {X_center}")
    txt_f.write(f" {y_center}")
    txt_f.write(f" {width}")
    txt_f.write(f" {height}\n")


xml_dir = input("Quel Dossier voulez-vous charger ? ")
for filename in os.listdir(xml_dir):
    with open(os.path.join(xml_dir, filename)) as xml_f:
        elem = xmltodict.parse(xml_f.read())

    if not os.path.exists("./label_me_txt"):
        os.makedirs("./label_me_txt")

    filename_txt = os.path.splitext(filename)[0] + ".txt"
    with open (os.path.join("./label_me_txt/", filename_txt), "+w") as txt_f:
        print(f"Ecriture du fichier : {filename_txt}")

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
        
        txt_f.close()