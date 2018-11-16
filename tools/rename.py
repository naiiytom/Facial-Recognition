
import os

datadir = os.path.expanduser('../datasets')
subdir = os.listdir(datadir)
subdir.sort()
for i in range(len(subdir)):
    facedir = os.path.join(datadir, subdir[i])
    print(facedir)
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        images.sort()
        for j in range(len(images)):
            print('rename: ' + images[j] + ' to ===> ' + facedir.split('\\')[1] + '_%04d.jpg' % j)
            os.rename(os.path.join(facedir, images[j]), os.path.join(facedir, (facedir.split('\\')[1] + '_%04d.jpg' % j)))
