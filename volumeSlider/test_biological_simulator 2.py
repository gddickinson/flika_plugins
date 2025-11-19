#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:26:57 2024

@author: george
"""

import unittest
import numpy as np
from simulateLightSheetData_2 import BiologicalSimulator
from scipy.ndimage import distance_transform_edt, center_of_mass

class TestBiologicalSimulator(unittest.TestCase):
    def setUp(self):
        self.size = (10, 10, 10)
        self.num_time_points = 5
        self.simulator = BiologicalSimulator(self.size, self.num_time_points)

    def test_generate_cell_membrane(self):
        size = (50, 50, 50)
        simulator = BiologicalSimulator(size, num_time_points=1)
        center = (25, 25, 25)
        radius = 20
        thickness = 1

        membrane = simulator.generate_cell_membrane(center, radius, thickness)

        # Check shape
        self.assertEqual(membrane.shape, size)

        # Check that the membrane is a hollow sphere
        self.assertTrue(np.any(membrane))  # Membrane is not empty
        self.assertFalse(np.all(membrane))  # Membrane is not solid

        # Check membrane thickness
        membrane_bool = membrane > 0
        z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
        dist_from_center = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        expected_membrane = (dist_from_center >= radius - thickness/2) & (dist_from_center <= radius + thickness/2)

        # Compare with actual membrane
        agreement = np.sum(expected_membrane == membrane_bool) / membrane_bool.size
        print(f"Membrane agreement: {agreement * 100:.2f}%")

        self.assertGreater(agreement, 0.99)  # Allow for small discrepancies

        # Check that the membrane is centered correctly
        com = np.array(center_of_mass(membrane))
        np.testing.assert_allclose(com, center, atol=1)

        # Print some debug information
        print(f"Membrane volume: {np.sum(membrane_bool)}")
        print(f"Expected membrane volume: {np.sum(expected_membrane)}")
        print(f"Center of mass: {com}")

    def test_generate_nucleus(self):
        size = (50, 50, 50)
        simulator = BiologicalSimulator(size, num_time_points=1)
        center = (25, 25, 25)
        cell_radius = 20
        nucleus_radius = 10
        thickness = 2

        nucleus = simulator.generate_nucleus(center, cell_radius, nucleus_radius, thickness)

        # Check shape
        self.assertEqual(nucleus.shape, size)

        # Check that the nucleus is a hollow sphere
        self.assertTrue(np.any(nucleus))
        self.assertLess(np.sum(nucleus), np.prod(size))  # Not filling the entire volume

        # Check nucleus size and hollowness
        filled_volume = np.sum(nucleus)
        total_volume = 4/3 * np.pi * nucleus_radius**3
        hollow_volume = total_volume - (4/3 * np.pi * (nucleus_radius - thickness)**3)
        self.assertAlmostEqual(filled_volume, hollow_volume, delta=hollow_volume*0.1)

        # Check that the center is empty
        self.assertEqual(nucleus[center[0], center[1], center[2]], 0)

        # Check that the nucleus is centered correctly
        com = np.array(center_of_mass(nucleus))
        np.testing.assert_allclose(com, center, atol=1)

        print(f"Nucleus volume: {filled_volume}")
        print(f"Nucleus center of mass: {com}")

    def test_protein_diffusion(self):
        #print("\nDebug: Entering test_protein_diffusion")
        size = (10, 10, 10)
        num_time_points = 5
        #print(f"Debug: Creating simulator with size {size} and {num_time_points} time points")
        simulator = BiologicalSimulator(size, num_time_points)

        #print("Debug: Creating initial concentration")
        initial_concentration = np.zeros(size)
        initial_concentration[5, 5, 5] = 1.0

        D = 0.1  # Diffusion coefficient
        #print(f"Debug: D = {D}")

        try:
            #print("Debug: Calling simulate_protein_diffusion")
            result = simulator.simulate_protein_diffusion(D, initial_concentration)

            #print("Debug: Checking result shape")
            self.assertEqual(result.shape, (num_time_points, *size))

            #print("Debug: Checking conservation of mass")
            for t in range(num_time_points):
                self.assertAlmostEqual(np.sum(result[t]), 1.0, places=6)

            #print("Debug: Checking diffusion occurrence")
            self.assertGreater(np.count_nonzero(result[-1]), np.count_nonzero(result[0]))

        except Exception as e:
            #print(f"Debug: Exception caught in test: {str(e)}")
            #print(f"Debug: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

        #print(f"Debug: Simulator size: {simulator.size}")
        #print(f"Debug: Initial concentration shape: {initial_concentration.shape}")
        #if 'result' in locals():
        #    print(f"Debug: Result shape: {result.shape}")

        #print("Debug: Test completed successfully")

    def test_protein_diffusion_invalid_input(self):
        #print("\nDebug: Entering test_protein_diffusion_invalid_input")
        size = (10, 10, 10)
        num_time_points = 5
        simulator = BiologicalSimulator(size, num_time_points)

        # Create an initial concentration with the wrong shape
        invalid_initial_concentration = np.zeros((5, 5, 5))
        D = 0.1

        #print("Debug: Attempting to simulate with invalid input")
        with self.assertRaises(ValueError) as context:
            simulator.simulate_protein_diffusion(D, invalid_initial_concentration)

        #print(f"Debug: Caught expected ValueError: {str(context.exception)}")
        self.assertTrue("Initial concentration shape (5, 5, 5) does not match simulator size (10, 10, 10)" in str(context.exception))
        #print("Debug: Test completed successfully")

if __name__ == '__main__':
    unittest.main(verbosity=2)
