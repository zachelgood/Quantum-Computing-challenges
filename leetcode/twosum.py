#!/usr/bin/env python3
class Solution(object):
    def twoSum(self, nums, target):
        value = {}
        for i,n in enumerate(nums):
            diff = target - n
            if diff in value:
                return [value[diff], i]
            value[n] = i
