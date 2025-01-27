# Annotations conversion and vizualisation scripts
This subfolder contains several miscellanous used for the study, such as the conversion voc-2-coco script ... Scripts for annotation (conversion and monitoring) are adapted for the LLVIP dataset, but feel free to adapt it for the FLIR-aligned dataset (minor modifications). 

# Voc-to-coco converter 
A script has been used in order to convert voc/xlm annotations from the LLVIP dataset into a COCO/Json like format, which is easier to handle with DETR-like models. 
Example of command to use it : (you need to download the original annotation files here
```bash 
python3 voc_to_coco.py --annotation_path ./dir/to/xml --json_save_path ./name/of/json/out/file
```
The script gathers and converts each xml file contained in the input folder in a json/coco-like file. 

# Annotation vizualiser script 
A script for image/image pairs vizualisation (to control LLVIP conversion). An argument mode gives the ability to monitor mono-spectrum annotations (mono) or directly on both pairs, on a randomly selected image name. 

```bash 
python3 visualize_random_img.py --images_dir /dir/to/spectrum1/test --annotation_file /dir/to/coco/LLVIP_test.json --mode multi --secondary_images_dir /dir/to/spectrum2/test
```

<figure>
<p align="center">
  <img src="monitor_anns.gif" alt="Alt Text 1">
</p>
<figcaption style="text-align: center; font-style: italic;"> <p> <i> Examples of image pairs with our converted annotations (LLVIP dataset).</i> </p> </figcaption>
</figure>

