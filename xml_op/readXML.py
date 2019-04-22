# -*- coding:utf-8 -*-
# 作者：chen
# 日期：19-4-19 下午7:56
# 工具：PyCharm
# Python版本：2.7.12

import  xml.etree.ElementTree as ET

def parse_laji_annotation(xml_file):
    tree = ET.parse(xml_file)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        # obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                            int(float(bbox.find('ymin').text)),
                            int(float(bbox.find('xmax').text)),
                            int(float(bbox.find('ymax').text))]
        objects.append(obj_dict)
    return objects







