#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TIFS 审稿 M18：查看验证集前 2000 张（旧 JPEG 子集）的类别构成。CPU，几秒。
用法：python check_val_order.py <DATA_ROOT>/data_npy/validation_meta_named.csv"""
import os

import sys
import pandas as pd
m = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('DATA_ROOT', '<DATA_ROOT>'), 'data_npy/validation_meta_named.csv'))
print('全部 7000：', m['gen_name'].value_counts().to_dict())
print('前 2000（旧 JPEG 子集）：', m['gen_name'].iloc[:2000].value_counts().to_dict())
print('前 2000 中真图占比：%.1f%%' % (100 * (m['gen_name'].iloc[:2000] == 'Real').mean()))
