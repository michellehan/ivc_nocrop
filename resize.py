import os
from glob import glob
from PIL import Image

rd = "/data/IVC_180814/IVC_Filter_Images/test/"
bd = "/home/mihan/projects/ivc_nocrop/src/gradcam.pytorch/3_detector/results_kfold0.25/heatmaps"
od = "/home/mihan/projects/ivc_nocrop/src/gradcam.pytorch/3_detector/results_kfold0.25/heatmaps_origsize"

dirs = sorted([os.path.abspath(x) for x in glob("%s/*" %bd)])
files = sorted([os.path.abspath(x) for x in glob("%s/*/*" %bd)])

for d in dirs:
    nd = od + d.split('heatmaps')[1]
    if not os.path.exists(nd): os.makedirs(nd)

for f in files:
    img = Image.open(f)
    img_orig = Image.open('%s/%s' %(rd, os.path.basename(f)))
    dim_orig = img_orig.size
    print(f.split('heatmaps')[1], img.size, dim_orig)
    
    outfile = od + f.split('heatmaps')[1]
    img = img.resize((dim_orig[0], dim_orig[1]), Image.ANTIALIAS)
    img.save(outfile)
