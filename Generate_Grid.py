import sys
import Utils
import os
import pandas as pd
import urllib.request as urllib2
import json
from random import randrange


if __name__ == '__main__':
    number_grids = 1
    if len(sys.argv) > 1:
        number_grids = int(sys.argv[1])
    output_data = Utils.generateGrid(number_grids)
    path_file = "./Grid_subcategories.json"
    with open(path_file, 'r+') as f:
        grid = json.load(f)
        final_grid = []
        for i in range(len(grid)):
            subcategories = grid[i]["subcategories"]
            id_finish = output_data[0][i]
            str_finish = ""
            if len(id_finish) == 1 and id_finish[0] == 0:
                final_grid.append(0)
            else:
                for k in range(len(id_finish)):
                    if id_finish[k] != 0:
                        str_finish += "finish_subcategories="
                        str_finish += str(id_finish[k])
                        if k < len(id_finish)-1:
                            str_finish += "&&"
                available_id = []
                for j in range(len(subcategories)):
                    url = "https://roomdesignerdb.interiorvista.net/api/sku_object_models/?subcategory=" + \
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
    with open('./Popular_Modern_Generated.json', 'r+') as outfile:
        json_data = json.load(outfile)
        json_data.append(d)
        outfile.seek(0)
        json.dump(json_data, outfile, indent=4)
        outfile.truncate()
