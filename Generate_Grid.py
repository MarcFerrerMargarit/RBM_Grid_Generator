import sys
import Utils
import os
import pandas as pd
import urllib.request as urllib2
import json
from random import randrange
from Constants import Constants


def main(arg1=None, nameStyle=""):
    if os.path.exists(Constants.Path_Server + nameStyle + "_Generated.json"):
        os.remove(Constants.Path_Server + nameStyle + "_Generated.json")
    f = open(Constants.Path_Server + nameStyle + "_Generated.json", "w+")
    f.write("[]")
    f.close()
    number_grids = 1
    if int(arg1) != 1:
        number_grids = int(arg1)
    print("Se van a generar " + str(number_grids) + " moodboards")
    output_data = Utils.generateGrid(number_grids, Constants.Path_Server_Data + 'RBM' + nameStyle + '.pickle', Constants.Path_Server_Data + "OneHotData" + nameStyle + ".pickle")
    path_file = Constants.Path_Server + "Tipology.json"
    with open(path_file, 'r+') as f:
        grid = json.load(f)
        for g in range(len(output_data)):
            print("Generando moodboard...")
            final_grid = []
            for i in range(len(grid)):
                subcategories = grid[i]["subcategories"]
                id_finish = output_data[g][i]
                str_finish = ""
                if len(id_finish) == 1 and id_finish[0] == 0:
                    final_grid.append(0)
                else:
                    for k in range(len(id_finish)):
                        if id_finish[k] != 0:
                            str_finish += "finish_subcategories="
                            str_finish += str(id_finish[k])
                            if k < len(id_finish) - 1:
                                str_finish += "&&"
                    available_id = []
                    # if len(subcategories) == 0:
                    #     print("No Subcategories")
                    #     url = "https://roomdesignerdb.interiorvista.net/api/sku_object_models/?" + str_finish
                    #     print(url)
                    for j in range(len(subcategories)):
                        url = Constants.Url_Api_Objects + "?subcategory=" + \
                              str(subcategories[j]["id"])
                        if str_finish != "":
                            url += "&&" + str_finish
                        req = urllib2.Request(url)
                        opener = urllib2.build_opener()
                        f = opener.open(req)
                        json_download = json.loads(f.read())
                        for d in range(len(json_download)):
                            available_id.append(json_download[d]["id"])
                    available_id = list(set(available_id))
                    if len(available_id) == 0:
                        final_grid.append(0)
                    else:
                        random_index = randrange(0, len(available_id))
                        final_grid.append(available_id[random_index])
            d = {"status:": "Available", "elements": final_grid}
            with open(Constants.Path_Server + nameStyle + "_Generated.json", 'r+') as outfile:
                json_data = json.load(outfile)
                json_data.append(d)
                outfile.seek(0)
                json.dump(json_data, outfile, indent=4)
                outfile.truncate()
    print("Moodboards añadidos en: " +(Constants.Path_Server + nameStyle + "_Generated.json"))


if __name__ == '__main__':
    main(sys.argv[1], "Popular_Modern_Generated.json")
