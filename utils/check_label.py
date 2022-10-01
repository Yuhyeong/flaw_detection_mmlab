import os
import numpy as np


def check_abel(label_dir):
    label_list = os.listdir(label_dir)
    for label_name in label_list:
        label_path = os.path.join(label_dir, label_name)

        f = open(label_path, 'r', encoding='utf-8')
        lines = f.readlines()

        for line in lines:
            info = line.strip().split(',')
            if info[0] == 'Perfect':
                continue


def minimumAverageDifference(nums) -> int:
    if len(nums) == 1:
        return 0

    before = nums[0]
    before_num = 1
    after = sum(nums) - before
    after_num = len(nums) - 1

    min_dist = dist = abs(int(before / before_num) - int(after / after_num))

    pos = 0
    i = 1
    while i < len(nums) - 1:
        before += nums[i]
        before_num += 1
        after -= nums[i]
        after_num -= 1
        dist = abs(int(before / before_num) - int(after / after_num))

        if min_dist > dist:
            pos = i
            min_dist = dist

        i += 1

    return pos


minimumAverageDifference([4,2,0])
