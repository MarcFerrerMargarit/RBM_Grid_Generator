import sys
import pandas as pd
import urllib.request as urllib2
import json
import os
from Constants import Constants
import os.path


def main(arg1):
    # do whatever the script does
    print("Generating file...")
    path_moodboard = Constants.Path_Server + arg1
    data = pd.read_json(path_moodboard)
    path_file_to_export = Constants.Path_Server_Data + arg1
    if os.path.exists(path_file_to_export):
        os.remove(path_file_to_export)
    f = open(path_file_to_export, "w+")
    f.close()
    final_dict = []
    for j in range(len(data)):
        last_grid = data["elements"][j]
        transformed_grid = []
        for i in range(len(last_grid)):
            id = last_grid[i]
            if id != 0:
                url = "http://roomdesigner-db-devtemp.interiorvistapps.com/api/sku_object_models/" + str(id) + "/"
                req = urllib2.Request(url)
                opener = urllib2.build_opener()
                f = opener.open(req)
                json_download = json.loads(f.read())
                transformed_grid.append(json_download['finish_subcategories'])
            else:
                transformed_grid.append([0])
        d = {"style": transformed_grid}
        final_dict.append(d)
    json_string = json.dumps(final_dict)
    f = open(path_file_to_export, "w+")
    f.write(json_string)
    f.close()
    print("File generated. Saved at: " + path_file_to_export)

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("You must set name of the style to transform.")
    else:
        main(sys.argv[1])

