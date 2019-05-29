# Open Information Extraction

This repository holds the code for experiments being done for OIE. 

## Description

Open information extraction is a NLP problem which deals with extracting structured triples of the form (subject, relation, object) from free text. There are both supervised and unsuperivised ways to deal with this problem. At the time of writing this, this project uses stanfordcorenlp oie parser to get OIE triples.

## Getting Started

1. Download Stanfordcorenlp from here and unzip the file.
2. In config.py change the value corresponding to "CORENLP_HOME" to the absolute path to the unzipped folder.
3. Do ``` pip install -r requirements.txt ```

### Executing program

The main code is present in oie_filters.py. Change the in_file and out_file variables to the input and output file path respectively.

## Useful links

[Google drive folder](https://drive.google.com/drive/folders/1Jv-lwiIsk1vQXseGKfqqCLWTWiRPNDmB?usp=sharing)
