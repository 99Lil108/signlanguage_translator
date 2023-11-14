import os
import csv
import re

folder_path = 'P_skeleton'
csv_file_path = 'mapper.csv'

def mk_mapper():
    file_names = os.listdir(folder_path)
    pattern = re.compile(r'(\d+)([\u4e00-\u9fa5]+)')

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for file_name in file_names:
            file_name, extension = os.path.splitext(file_name)
            matches = pattern.match(file_name)
            id = matches.group(1)
            word = matches.group(2)

            csvwriter.writerow([word, id])

def modidy_mapper():
    file_names = os.listdir(folder_path)
    pattern = re.compile(r'(\d+)([\u4e00-\u9fa5]+)')
    ab_file_names = os.path.join(os.getcwd(), folder_path)

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([0,"<pad>"])
        csvwriter.writerow([1,"<unk>"])
        csvwriter.writerow([2,"<s>"])
        csvwriter.writerow([3,"</>"])
        for file_name in file_names:
            name, extension = os.path.splitext(file_name)
            matches = pattern.match(name)
            id = int(matches.group(1)) + 3
            word = matches.group(2)

            os.rename(os.path.join(ab_file_names,file_name), os.path.join(ab_file_names,f'{id}{word}{extension}'))

            csvwriter.writerow([id, word])

if __name__ == '__main__':
    # mk_mapper()
    modidy_mapper()
    pass