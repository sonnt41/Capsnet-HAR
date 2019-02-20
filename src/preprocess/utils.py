# -*- coding: utf-8 -*-

import csv
import os
from os import listdir
from os.path import join
from xml.etree import ElementTree




def read_eaf(path):
    try:
        root = ElementTree.parse(path).getroot()
    except:
        print(path)
    ts_mapping = dict()

    ts_label = dict()
    for element in root:
        if element.tag == "TIME_ORDER":
            for ele in element:
                ts_mapping[ele.attrib['TIME_SLOT_ID']] = int(ele.attrib['TIME_VALUE'])
            #print(ts_mapping)
            print('-------------')
        elif element.tag == "TIER":
            for annotation in element:
                alignable_annotation = annotation[0]
                id = alignable_annotation.attrib['ANNOTATION_ID']
                ts_start = alignable_annotation.attrib['TIME_SLOT_REF1']
                ts_end = alignable_annotation.attrib['TIME_SLOT_REF2']
                #label = alignable_annotation[0].text.encode('utf-8')
                try:
                    label = alignable_annotation[0].text.encode('utf-8')
                except:
                    print('Error at file: %s'%(path))
                ts_label[ts_mapping[ts_start], ts_mapping[ts_end]] = label
                print(id, ts_start, ts_end, label.decode('utf-8'))
    return ts_label



def find_path_with_ext(directory, extension):
    paths = []
    for annotator in listdir('data'):
        p = join(directory, annotator)
        if os.path.isdir(p):
            for position in listdir(p):
                position_path = join(p, position)
                if os.path.isfile(position_path):
                    continue
                for fname in listdir(position_path):
                    if fname.endswith(extension):
                        paths.append(join(position_path, fname))
    return paths

if __name__ == '__main__':

    for path in find_path_with_ext('data','eaf'):
        a = read_eaf(path)
