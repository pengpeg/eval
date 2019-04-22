# -*- coding:utf-8 -*-
# 作者：chen
# 日期：19-4-19 下午7:16
# 工具：PyCharm
# Python版本：2.7.12

import os
import numpy as np
from xml_op.readXML import parse_laji_annotation

# 检测结果格式
# 图片路径
# 检测到的目标个数
# 类别 xmin ymin xmax ymax (confidence)
def parse_dets(det_file):
    flag = 1  #
    dets = []
    with open(det_file, 'r') as f:
        for line in f:
            if flag == 1:
                det = {}
                line = line[:line.find('.')]
                file_name = line[line.find('/')+1:]
                det['file_name'] = file_name
                flag = 2
                continue
            if flag == 2:
                num_obj = int(line.strip())
                objs = []
                if num_obj == 0:
                    det['objs'] = objs
                    dets.append(det)
                    flag = 1
                else:
                    flag = 3
                continue
            if flag == 3:
                obj = dict()
                strs = line.split(' ')
                obj['name'] = strs[0]
                obj['bbox'] = [int(strs[2].strip()), int(strs[3].strip()), int(strs[4].strip()), int(strs[5].strip())]
                objs.append(obj)
                num_obj = num_obj - 1
                if num_obj == 0:
                    det['objs'] = objs
                    dets.append(det)
                    flag = 1
                continue
    return dets

def det_anno_compare(det_objs, anno_objs, ovp_thresh=0.5):
    """
    处理单张图片的检测结果和标注对照。
    :param det_objs: [dict,]
    :param anno_objs: [dict,]
    :return:
    """
    def iou(x, y):
        """
        Calculate intersection-over-union overlap
        Params:
        ----------
        x : list
            single box [xmin, ymin ,xmax, ymax]
        y : list
            single box [xmin, ymin, xmax, ymax]
        Returns:
        -----------
        numpy.array
            [iou1, iou2, ...], size == ys.shape[0]
        """
        ixmin = max(y[0], x[0])
        iymin = max(y[1], x[1])
        ixmax = min(y[2], x[2])
        iymax = min(y[3], x[3])
        iw = max(ixmax - ixmin, 0.)
        ih = max(iymax - iymin, 0.)
        inters = iw * ih
        uni = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * \
              (y[3] - y[1]) - inters
        iou = float(inters) / uni

        if uni < 1e-12: # in case bad boxes
            iou = 0
        return iou
    tp = 0
    fp = 0
    for det_obj in det_objs:
        cls_name = det_obj['name'].split('_')[0]
        x = det_obj['bbox']
        max_iou = 0.0
        index = -1
        for i in range(len(anno_objs)):
            anno_obj = anno_objs[i]
            if cls_name == anno_obj['name'].split('_')[0]:
                y = anno_obj['bbox']
                iou_cur = iou(x, y)
                if iou_cur > max_iou:
                    max_iou = iou_cur
                    index = i
        if max_iou > ovp_thresh and index > -1:
            anno_objs.pop(index)
            tp = tp + 1
        else:
            fp = fp + 1
    fn = len(anno_objs)
    return tp, fp, fn


if __name__ == '__main__':
    det_file = '/home/chen/models/datasets/laji/pr.txt'
    anno_path = '/home/chen/models/datasets/laji/hebing/Annotations/'
    dets = parse_dets(det_file)
    image_anno_tmpl = os.path.join(anno_path, '{}.xml')
    total_obj_num = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for det in dets:
        anno_objs = parse_laji_annotation(image_anno_tmpl.format(det['file_name']))
        tp, fp, fn = det_anno_compare(det['objs'], anno_objs)
        tp_total = tp_total + tp
        fp_total = fp_total + fp
        fn_total = fn_total + fn

    print("precision: %f" % (float(tp_total) / (tp_total + fp_total)))
    print("recall: %f" % (float(tp_total) / (tp_total + fn_total)))




