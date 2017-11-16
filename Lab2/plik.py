#some magic to show the images inside the notebook
# %pylab inline
# %matplotlib inline

import matplotlib.pyplot as plt
from scipy import ndimage
import imageio as imio
import numpy as np
import subprocess

# A hepler function for displaying images within the notebook.
# It displays an image, optionally applies zoom the image.
def show_image(img, zoom=1.5):
    dpi = 77
    plt.figure(figsize=(img.shape[0]*zoom/dpi,img.shape[0]*zoom/dpi))
    if len(img.shape) == 2:
        img = np.repeat(img[:,:,np.newaxis],3,2)        
    plt.imshow(img, interpolation='nearest')
    

# A hepler function for displaying images within the notebook.
# It may display multiple images side by side, optionally apply gamma transform, and zoom the image.
def show_images(imglist, zoom=1, needs_encoding=False):
    if type(imglist) is not list:
       imglist = [imglist]
    n = len(imglist)
    first_img = imglist[0]
    dpi = 77 # pyplot default?
    plt.figure(figsize=(first_img.shape[0]*zoom*n/dpi,first_img.shape[0]*zoom*n/dpi))
    for i in range(0,n):
        img = imglist[i]
        plt.subplot(1,n,i + 1)
        plt.tight_layout()    
        plt.axis('off')
        if len(img.shape) == 2:
           img = np.repeat(img[:,:,np.newaxis],3,2)
        plt.imshow(img, interpolation='nearest')    
    

def saveHDR(filename, image):
    f = open(filename, "wb")
    f.write(str.encode("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n"))
    f.write(str.encode("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1])))
    
    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)
    
    rgbe.flatten().tofile(f)
    f.close()

def dcraw_meta(path):
    lines = subprocess.check_output(["dcraw", "-i", "-v", path]).decode("utf-8").splitlines()
    lines = [[x.strip() for x in line.split(":", 1)] for line in lines if ":" in line]
    lines = { x[0] : x[1] for x in lines }

    if "Aperture" in lines:
        lines["Aperture"] = float(lines["Aperture"].split("/")[1])

    if "ISO speed" in lines:
        lines["ISO speed"] = float(lines["ISO speed"])

    if "Shutter" in lines:
        shutter = lines["Shutter"].split()[0]
        shutter = shutter.split("/")
        lines["Shutter"] = float(shutter[0]) / float(shutter[1])

    return lines

def getExposureInSeconds(path):
    meta = dcraw_meta(path)
    result = meta["Aperture"] ** 2 / (meta["Shutter"] * meta["ISO speed"])
    return meta["Shutter"]



import subprocess, shlex

def raw2linear_tiff(path_to_file,name):
    my_cmd = './dcraw   -4 -T -c '+path_to_file + ' > '+name
    args = shlex.split(my_cmd)
    subprocess.call(args)
    subprocess.Popen(my_cmd, shell=True)
    
path="HDR/sec1/"
raw2linear_tiff(path+"_MG_4460.CR2",path+"img_1")
raw2linear_tiff(path+"_MG_4461.CR2",path+"img_2")
raw2linear_tiff(path+"_MG_4462.CR2",path+"img_3")

import imageio
img1=ndimage.imread(path+"img_1")/255.
img2=ndimage.imread(path+"img_2")/255.
img3=ndimage.imread(path+"img_3")/255.
#show_images([img1[:1800 , :2000],img2[:1800 , :2000],img3[:1800 , :2000]],0.2) #takes a region of the image 


t=range(0,3)
t[0]=1/8.
t[1]=1/30.
t[2]=1/2.
w= lambda x: np.exp(-4.*((x-.5)**2)/(0.5**2))
ilin= lambda x, t: t*x

Ilin=range(0,3)


Ilin[0]=ilin(img1,t[0])
Ilin[1]=ilin(img2,t[1])
Ilin[2]=ilin(img3,t[2])

W=range(0,3)
for i in range(0,3):
    W[i]=w(Ilin[i])

print "1"
#print Ilin[0]*t[0]
X1=t[0]*Ilin[0]*W[0]
X2=W[0]*t[0]**2.
for i in range(1,3):
    X1+=W[i]*Ilin[i]*t[i]
    X2+=W[i]*t[i]**2

print "2"    
X=np.clip(X1/X2,0.,1.)

print "3"

saveHDR("img_HDR", X)

print "4"
show_image(X[:2000,:2000],0.5)