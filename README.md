# P2-Enrich-data-with-OpenCV-python

## Dependencies

pip install opencv-python

## Detail

This script add gender and age fields to each json node list processing your url image which key must be specified, if the image do not contain faces your node is not included

## Execute command

To execute the script is necessary to specify three arguments:
1. The input json name file
2. Your field node key of image chosen to process
3. The output json name file 

python addGenderAndAgeFields.py inputJsonFilePath fieldImageJsonKey outputJsonFilePath 

## Example
python addGenderAndAgeFields.py lastoutputjson.json publicationImageUrl outPutDatass.json


