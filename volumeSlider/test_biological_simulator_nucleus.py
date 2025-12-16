import unittest
import numpy as np
from simulateLightSheetData_2 import BiologicalSimulator

class TestBiologicalSimulator(unittest.TestCase):
    def setUp(self):
        self.size = (30, 100, 100)
        self.cell_center = (15, 50, 50)
        self.simulator = BiologicalSimulator(self.size, num_time_points=1)

        # Create a simple spherical cell shape
        z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
        dist_from_center = np.sqrt((z - self.cell_center[0])**2 +
                                   (y - self.cell_center[1])**2 +
                                   (x - self.cell_center[2])**2)
        cell_radius = min(self.size) // 4
        self.cell_shape = dist_from_center <= cell_radius

    def test_generate_nucleus(self):
        for nucleus_radius in [1, 2, 3, 5, 10]:  # Test various radii
            with self.subTest(nucleus_radius=nucleus_radius):
                nucleus, nucleus_center = self.simulator.generate_nucleus(self.cell_shape, self.cell_center, nucleus_radius)

                # Assertions
                self.assertEqual(nucleus.shape, self.size, "Nucleus shape should match the cell shape")
                self.assertGreater(np.sum(nucleus), 0, "Nucleus should have at least one voxel")
                self.assertLessEqual(np.sum(nucleus), np.sum(self.cell_shape), "Nucleus should not be larger than the cell")

                if nucleus_radius > 1:
                    self.assertGreater(np.sum(nucleus), 1, f"Nucleus with radius {nucleus_radius} should have more than one voxel")

                # Check if the nucleus is centered correctly (allow for small variations)
                np.testing.assert_allclose(nucleus_center, self.cell_center, atol=5,
                                           err_msg="Nucleus should be approximately centered")

                # Check that the nucleus is entirely contained within the cell
                self.assertTrue(np.all(nucleus[self.cell_shape == 0] == 0), "Nucleus should be contained within the cell")

                print(f"Nucleus radius: {nucleus_radius}")
                print(f"Nucleus volume: {np.sum(nucleus)}")
                print(f"Nucleus shape: {nucleus.shape}")
                print(f"Cell shape volume: {np.sum(self.cell_shape)}")
                print(f"Cell center: {self.cell_center}")
                print(f"Nucleus center: {nucleus_center}")
                print("Nucleus cross-section:")
                print(nucleus[self.cell_center[0], self.cell_center[1]-5:self.cell_center[1]+6, self.cell_center[2]-5:self.cell_center[2]+6])
                print("---")

    def test_generate_nucleus_edge_cases(self):
        # Test with a very small cell
        small_cell = np.zeros(self.size)
        small_cell[0, 0, 0] = 1
        nucleus, _ = self.simulator.generate_nucleus(small_cell, (0, 0, 0), 5)
        self.assertEqual(np.sum(nucleus), 1, "Should create a single voxel nucleus for a single-voxel cell")

        # Test with an empty cell
        empty_cell = np.zeros(self.size)
        nucleus, _ = self.simulator.generate_nucleus(empty_cell, (15, 50, 50), 5)
        self.assertEqual(np.sum(nucleus), 0, "Should not create a nucleus for an empty cell")

        # Test with a cell that has a hole in the center
        holey_cell = np.ones(self.size)
        hole_center = (15, 50, 50)
        hole_radius = 2
        z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
        dist_from_hole_center = np.sqrt((z - hole_center[0])**2 + (y - hole_center[1])**2 + (x - hole_center[2])**2)
        holey_cell[dist_from_hole_center <= hole_radius] = 0
        nucleus, nucleus_center = self.simulator.generate_nucleus(holey_cell, hole_center, 5)

        print("Holey cell cross-section:")
        print(holey_cell[hole_center[0], hole_center[1]-5:hole_center[1]+6, hole_center[2]-5:hole_center[2]+6])
        print("Nucleus cross-section:")
        print(nucleus[hole_center[0], hole_center[1]-5:hole_center[1]+6, hole_center[2]-5:hole_center[2]+6])
        print(f"Nucleus center: {nucleus_center}")

        self.assertGreater(np.sum(nucleus), 0, "Should create a nucleus even with a hole in the cell center")
        self.assertTrue(np.all(nucleus[holey_cell == 0] == 0), "Nucleus should not fill the hole in the cell")
        self.assertLess(np.sum(nucleus), np.sum(holey_cell), "Nucleus should be smaller than the cell")

        # Test nucleus size limitation
        large_cell = np.ones(self.size)
        large_nucleus, _ = self.simulator.generate_nucleus(large_cell, hole_center, 50)  # Very large radius
        self.assertGreater(np.sum(large_nucleus), 0, "Should create a nucleus even with a very large radius")
        self.assertLess(np.sum(large_nucleus), np.sum(large_cell), "Nucleus should not be larger than the cell")

        print("Large nucleus cross-section:")
        print(large_nucleus[hole_center[0], hole_center[1]-5:hole_center[1]+6, hole_center[2]-5:hole_center[2]+6])



if __name__ == '__main__':
    unittest.main()
