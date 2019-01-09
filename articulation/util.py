#the aim of this code is to read the txt file and convert to a form which LIWC
#can directly use. Additionally, add the number of text into the excel log file

import os
import csv

def write_to_csv(file_name,features,write_name,feature_name=[]):

    if len(file_name) != len(features):
        print "Error, file name and feature size not match"
    else:

        with open(write_name, "wb") as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if not feature_name==[]:
                writer.writerow(["ID"]+feature_name)
                
            for i in range(len(file_name)):
                  file_=[file_name[i]]
                  file_.extend(features[i])
                  writer.writerow(file_)
        f.close()

def write_dynamic_to_csv(file_name,features,write_name,feature_name=[]):

    if len(file_name) != len(features):
        print "Error, file name and feature size not match"
        print len(file_name),len(features)
    else:
        with open(write_name, "wb") as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if not feature_name==[]:
                writer.writerow(["ID"]+feature_name)
                
            for i in range(len(file_name)):
                file_=[file_name[i]]
                for j in range(len(features[i])):
                      file_.append(str(features[i][j]))
                writer.writerow(file_)
        f.close()


