#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:58:54 2024

@author: george
"""

import unittest
import numpy as np
from your_module import VolumeProcessor

class TestVolumeProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = VolumeProcessor()

    def test_apply_threshold(self):
        data = np.array([0, 0.5, 1])
        result = self.processor.apply_threshold(data, 0.5)
        np.testing.assert_array_equal(result, [0, 0.5, 1])

    def test_calculate_statistics(self):
        data = np.array([1, 2, 3, 4, 5])
        stats = self.processor.calculate_statistics(data)
        self.assertEqual(stats['mean'], 3)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 5)

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
