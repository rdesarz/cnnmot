import unittest
import numpy as np
from prediction import prediction_step, dynamic_model


class PredictionTestCase(unittest.TestCase):
    def test_prediction_step_unitary_matrix(self):
        state_vector = np.array([[1, 1]]).transpose()
        delta_t = 1

        result = prediction_step(state_vector, dynamic_model, delta_t)
        self.assertTrue(np.array_equal(result, np.array([[1, 1]]).transpose()))


if __name__ == '__main__':
    unittest.main()
