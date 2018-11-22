from google_images_download import google_images_download
import os, subprocess
from autocrop import autocrop

FolderList = []
CropList = []
basedir = './'
space = ' '
downloadArgument_dict = {'keywords_from_file': 'BNK48_Members.csv', 'limit': '50', 'format': 'jpg', 'prefix_keywords': 'BNK48'}

response = google_images_download.googleimagesdownload()
absolute_image_paths = response.download(downloadArgument_dict)

#Credit to phihag for answer on how to change folder name with python 
#(https://stackoverflow.com/questions/8735312/how-to-change-folder-names-in-python)
for folder in os.listdir(basedir):
    if folder.count(space) != 0:
        newName = folder.replace(space,'_')
        os.rename(os.path.join(basedir, folder), os.path.join(basedir, newName))
        FolderList.append(basedir + '/' + folder)
        CropList.append(basedir + '/' + 'Crop_' + folder)
    else:
        FolderList.append(basedir + '/' + folder)
        CropList.append(basedir + '/' + 'Crop_' + folder)

i=0
for member in FolderList:
    subprocess.Popen('cmd.exe /k autocrop -i "%s" -o "%s"' %(FolderList[i], CropList[i]))
    i+=1