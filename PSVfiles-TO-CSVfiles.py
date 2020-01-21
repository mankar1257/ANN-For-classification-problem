
""" 

@author: vaibhav
"""

import pandas as pd
from pandas import read_csv 
# As this is a PSV file we convert it to CSV 

import os
import glob
#set working directory
os.chdir("Add-to-the-working-diretry")
 
#find all csv files in the folder
#use glob pattern matching -> extension = ' csv'
#save result in list -> all_filenames
extension = 'psv'
fileList = [i for i in glob.glob('*.{}'.format(extension))]

dataset111 =[pd.read_csv(file,skiprows=1,sep='|').fillna(0) for file in fileList ]
# manually specify column names
i = 0 
for data in dataset111:
    data.columns = ["specify the names of the columns"]
    fileN = '/home/vaibhav/Desktop/training_setB/subject'+str(i)+'.csv'
    data.to_csv(fileN)
    i+=1
    
  
# save to file
