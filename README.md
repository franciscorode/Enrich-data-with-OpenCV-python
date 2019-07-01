# P2-Enrich-data-with-OpenCV-python

## Dependencies

pip install opencv-python

or

git clone https://github.com/opencv/opencv.git

## Detail

The folfer "data" must be to same level where the script is executed

This script add gender and age fields to each json node list processing your url image which key must be specified, if the image do not contain faces your node is not included

## Execute command

To execute the script is necessary to specify three arguments:
1. The input json name file
2. Your field node key of image chosen to process
3. The output json name file 

python addGenderAndAgeFields.py inputJsonFilePath fieldImageJsonKey outputJsonFilePath 

## Example
python addGenderAndAgeFields.py lastoutputjson.json publicationImageUrl outPutDatass.json

## More
You can change the input json file by other generated in the phase one

### Origin documentation
https://github.com/raunaqness/Gender-and-Age-Detection-OpenCV-Caffe

The result of predictions are sometimes not successful
