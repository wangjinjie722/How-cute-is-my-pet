Filter of BrainCorporation 
==== 

Brief Description 
-------
This is a project including two filters.\
The frist one is range filter. ([0.03, 50])\
The second one is temporal median filter.

Installation Instruction  
-------
Development Envrionment: Pycharm 2018.2.4\
Language: Python 2.7


API Documentation 
-------
>### BrainCorp
>>#### test
>>>##### test.py
This is the test file of the filter.\
The input are a set of float numbers listed in test_input.csv and test_result.csv.\
This function will call validate.py to validate the result.
>>>##### validate.py
This is a function used to test whether the result is equal to what we expect.\
The expected values are listed in test_result.csv.
>>#### brain.py
This function define the two filters.\
The input of the range filter is a set of measured distance.\
Its output is a set of distance ranged in [0.03.50].\
The input of the temporal median filter is a set of current scan, a set of previous scan and a number D that means we will calculate the median of previous D scans.\
Its out are median of the previous D scans and prevous D set scans.
>>#### main.py
The entry of the test file.
>>#### data
>>>##### test_input.csv
The input data.
>>>##### test_result.csv
The expected output data, which will be used to validate the correctness of the filters.

Version  
-------
> Time: 02/18/2019\
> Version: 1.0

Contact 
-------
Author: Kai Wang\
E-mail: wangjinjie722@gmail.com


TIPS: Demo for Brain Corporation Test
====
