from CascadeTabNet.Table_Structure_Recognition.border import border

from mmdetection.mmdet.apis import inference_detector, show_result_pyplot, init_detector
import cv2
from CascadeTabNet.Table_Structure_Recognition.Functions.blessFunc import borderless
import lxml.etree as etree
import glob
import os
import gdown
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

gdown_model_url = 'https://drive.google.com/file/d/1IATqgV8GAGTmCJlqTD92kINcl8r8wSeJ/view?usp=sharing'
gdown_model_output_dir = 'model_dir'
gdown.download(gdown_model_url, gdown_model_output_dir)

gdown_dataset_dir = 'dataset_dir'
# 19qMDNMWgw04T0HCQ_jADq1OvycF3zvuO
gdown_dataset_url = 'https://drive.google.com/drive/folders/14ZYVF3hxrvkMnRI0wdKveuDs9FPSlKSu?usp=sharing'
gdown.download(gdown_dataset_url, gdown_dataset_dir)

# before conduct this code, please conduct the code of mmdetection install, gdown the model file from gdrive, gdown the dataset from gdrive


envir = "local"
envir = 'hubble'
if envir == 'local':
    config_file = 'Config/config_CascadeTabNet_1.py'
    checkpoint_file = gdown_model_output_dir + '/epoch_30.pth'
    image_path = xmlPath + '/chunk_images'
    xmlPath = xmlPath + '/orig_chunk'
    xmlPath_write = xmlPath + '/infer_res/'
else:
    config_file = 'Config/config_CascadeTabNet_1.py'
    checkpoint_file = gdown_model_output_dir + '/epoch_30.pth'
    image_path = xmlPath + '/chunk_images'
    xmlPath = xmlPath + '/orig_chunk'
    xmlPath_write = xmlPath + '/infer_res/'


model = init_detector(config_file, checkpoint_file)

# List of images in the image_path
img_dir = os.listdir(image_path)
print("img_dir")
print(img_dir)
for path in img_dir:
    image_fullPath = os.path.join(image_path, path)
    print("image_fullPath")
    print(image_fullPath)
    xmlPath_write_name = xmlPath_write+image_fullPath.split('/')[-1][:-3]+'xml'
    if os.path.exists(xmlPath_write_name):
        size = os.path.getsize(xmlPath_write_name)
    else:
        size = 0
    print("size")
    print(size)
    # if size > 1000 or '10039' in image_fullPath or '10481' in image_fullPath or '10360' in image_fullPath or '10090' \
    #         in image_fullPath  or '10142'  in image_fullPath or '10044' in image_fullPath or '10314' in image_fullPath:
    #     continue
    # else:
    #     pass
    result = inference_detector(model, image_fullPath)
    print("result1")
    print(result)

    out_file = image_path+"_infer/" + path
    print("out_file:", out_file)
    show_result_pyplot(model=model, img=image_fullPath, result=result, out_file=out_file)
    # exit()
    res_border = []
    res_bless = []
    res_cell = []
    root = etree.Element("document")
    print("11111")
    ## for border
    for r in result[0][0]:
        if r[4]>0:
            res_border.append(r[:4].astype(int))
    ## for cells
    for r in result[0][1]:
        if r[4]>0:
            r[4] = r[4]*100
            res_cell.append(r.astype(int))
    ## for borderless
    for r in result[0][2]:
        if r[4]>0:
            res_bless.append(r[:4].astype(int))
    print("2222")
    ## if border tables detected
    if len(res_border) != 0:
        print("border root")
        ## call border script for each table in image
        for res in res_border:
            try:
                root.append(border(res,cv2.imread(image_fullPath)))
            except:
                pass
        print("border root")
        print(root)
    print("3333")
    print("res_bless")
    print(res_bless)
    # if borderless tables detected
    if len(res_bless) != 0:
        print("borderless root")
        if len(res_cell) != 0:
            for no,res in enumerate(res_bless):
                print("cvcvcv")
                root.append(borderless(res,cv2.imread(image_fullPath),res_cell))

        print("borderless root")
        print(root)
    # write results to XML file
    print("write image_fullPath: ",xmlPath_write+image_fullPath.split('/')[-1][:-3]+'xml')
    myfile = open(xmlPath_write+image_fullPath.split('/')[-1][:-3]+'xml', "w")
    myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    myfile.write(etree.tostring(root, pretty_print=True,encoding="unicode"))
    myfile.close()