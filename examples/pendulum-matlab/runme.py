import matlab.engine
import os
import urllib.request
import zipfile

if not os.path.isdir("pilcov0.9"):
    print("Matlab implementation not found in current path.")
    print("Attempting to download now")
    urllib.request.urlretrieve("http://mlg.eng.cam.ac.uk/pilco/release/pilcoV0.9.zip", "pilcoV0.9.zip")
    zip_ref = zipfile.ZipFile("pilcoV0.9.zip", 'r')
    zip_ref.extractall("./")
    zip_ref.close()
    print("Done!")
