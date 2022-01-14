import csv
import glob

path = '../_data/project data/' 
merge_path = '../_data/project data/merge.csv'

file_list = glob.glob(path + '*')

with open(merge_path, 'w') as f:
    for i, file in enumerate(file_list):
        if i ==0:
            with open(file, 'r') as f2:
                while True:
                    line = f2.readline()
                    
                    if not line:
                        break
                    f.write(line)
            file_name = file.split('\\')[-1]
            print(file.split('//')[-1] + 'write complete...')
        else:
            with open(file, 'r') as f2:
                n = 0
                while True :
                    line = f2.readline()
                    if n !=0:
                        f.write(line)
                    if not line:
                        break
                    n += 1
            file_name = file.split('//')[-1]
            print(file.split('//')[-1]+ 'write complete...')
print('>>>all file merge complete...')
            
    


