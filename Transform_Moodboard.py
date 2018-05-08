import sys
import pandas as pd
import urllib.request as urllib2
import json
import os

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("You must set name of the style to transform.")
    else:
        path_moodboard = "./" + sys.argv[1]
        for filename in os.listdir(path_moodboard):
            data = pd.read_json(path_moodboard + "/"+ filename)
            last_grid = data["elements"][len(data)-1]
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
            with open('./Transformed_Moodboards.json', 'r+') as f:
                data = json.load(f)
                d = {"style":transformed_grid}
                data.append(d)  # <--- add `id` value.
                f.seek(0)  # <--- should reset file position to the beginning.
                json.dump(data, f, indent=4)
                f.truncate()  # remove remaining part
