import numpy
import cv2
import urllib.request
from mpi4py import MPI
from google.cloud import storage
import os
from pathlib import Path
import shutil
import time

comm = MPI.COMM_WORLD
r = comm.Get_rank()
size = comm.Get_size()

# Master
# for each categoria
#   get lista de imagenes - 1
#   base = dividir len entre workers
#   lista_de_indices = ...
#   comm.bcast([0, lista_de_indices[0]) 
#   comm.bcast([0, [0 - base-1], [base, (base*2)-1], lista])
#   wait

# Resto
# res[r]
# wait()


# para todos
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/mpi/credentials.json"
client = storage.Client('cloud-sergio')
bucket = client.get_bucket('proyecto_cloud_sergio')

emojis = ["Angry", "Happy", "Sad", "Poo", "Surprised"]
emojis_prefix = ["angry_", "happy_", "sad_", "poo_", "surprised_"]
if r == 0:
    for i in range(len(emojis)):
        blobs = bucket.list_blobs(prefix="Input/" + emojis[i])
        blobList = []
        for blb in blobs:
            blobList.append(blb.name)

        blobList = blobList[1:]
        base = len(blobList) // (size-1) # 500/5 = 100

        blobs_by_rank = []
        blobs_by_rank.append(0)
        ctr = 0
        for j in range(1,size):
            if j == 0:
                data = comm.isend([ctr, int(base*(j+1)),"Input/" + emojis[i] + "/" + emojis_prefix[i]], dest=j, tag=j)
                data.wait()            
            else:
                data = comm.isend([ctr+1, int(base*(j+1)), "Input/" + emojis[i] + "/" + emojis_prefix[i]], dest=j, tag=j)
                data.wait()
            ctr += base
            #blobs_by_rank.append("Input/" + emojis[i] + "/" + emojis_prefix[i])
        print("sent ", emojis[i])
    
    for i in range(1, size):
        data = comm.isend("exit", dest=i, tag=i)
        data.wait()
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/mpi/credentials.json"
    client = storage.Client('cloud-sergio')
    bucket = client.get_bucket('proyecto_cloud_sergio')
    data = None

    while True:
        req = comm.irecv(source = 0, tag = r)
        res = req.wait()
        if res == "exit":
            print("res returned exit")
            break
        prefix = res[-1:]
        a = res[0]
        b = res[1]
        a_o = a
        while a < b:
            # print("{0}{1}.jpg".format(prefix[0], a))
            filename = prefix[0] + str(a) + ".jpg"
            blob = bucket.blob(filename)
            try:
                blob.download_to_filename("imagen.jpg")
            except:
                print("Error descargando ", filename)
                a += 1
                continue
            img = cv2.imread("imagen.jpg")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(img_gray)
            ret, binarized = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)
            # Dilation for better closing
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
            closing = cv2.dilate(binarized, kernel, iterations=4)

            # Find countours
            contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            if len(contours) != 0:
                c = contours_sorted[0]
                x, y, w, h = cv2.boundingRect(c)

                # crop
                ROI = binarized[y:y+h, x:x+w]

                ROI = cv2.resize(ROI, (32,32), interpolation=cv2.INTER_AREA)
                cv2.imwrite('output.jpg', ROI)
                splits = prefix[0].split('/')
                # print(splits)
                upload_filename = "Output/" + splits[1] + '/' + splits[2] + str(a) + '.jpg'

                # Subir imagen
                # print(upload_filename)
                blob= bucket.blob(upload_filename)
                blob.upload_from_filename('output.jpg')

            a += 1
        print("Finished with {0} from {1} to {2}".format(prefix, a_o, a))
