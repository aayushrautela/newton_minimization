import unittest
import numpy as np
from lab1_v4 import function, gradient, hessian, newton_method

class TestNewtonMethod(unittest.TestCase):
    def test_function_evaluation(self):
        # Test function values at known points
        self.assertAlmostEqual(function(0, 0), 3.0)
        self.assertAlmostEqual(function(np.pi/2, 0), 5.0)  # 2*1 + 3*1 = 5
        self.assertAlmostEqual(function(0, np.pi/2), 0.0)  # 2*0 + 3*0 = 0
        
    def test_gradient_calculation(self):
        # Test gradient at known points
        np.testing.assert_array_almost_equal(
            gradient(0, 0),
            np.array([2.0, 0.0])
        )
        np.testing.assert_array_almost_equal(
            gradient(np.pi/2, np.pi/2),
            np.array([0.0, -3.0])
        )
        
    def test_hessian_calculation(self):
        # Test Hessian at known points
        np.testing.assert_array_almost_equal(
            hessian(0, 0),
            np.array([[0.0, 0.0], [0.0, -3.0]])
        )
        np.testing.assert_array_almost_equal(
            hessian(np.pi/2, np.pi/2),
            np.array([[-2.0, 0.0], [0.0, 0.0]])
        )
        
    def test_newton_convergence(self):
        # Test convergence behavior
        test_cases = [
            ([2.0, 2.0], True),  # Should converge to minimum
            ([-3.0, 3.0], True),  # Should converge to minimum
            ([0.0, 0.0], False)   # Hessian is singular here
        ]
        for initial_guess, should_converge in test_cases:
            (x, y), iterations, _ = newton_method(initial_guess, 1.0)
            if should_converge:
                self.assertTrue(iterations < 100, "Should converge in reasonable iterations")
                self.assertLessEqual(function(x, y), -1.0,
                                   f"Should converge to minimum (got {function(x,y)})")
            else:
                self.assertEqual(iterations, 1, "Should terminate immediately at singular point")

if __name__ == '__main__':
    unittest.main()