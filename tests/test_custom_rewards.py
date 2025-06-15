import unittest
from src.open_r1.custom_rewards import accuracy_reward


class TestAccuracyReward(unittest.TestCase):
    def test_perfect_match(self):
        # Test case with perfect match
        completions = [
            [
                {
                    "content": "<answer>\n{(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)}\n</answer>"
                }
            ]
        ]
        ground_truth = [
            "(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)"
        ]
        rewards = accuracy_reward(completions, ground_truth)
        self.assertEqual(rewards[0], 1.0)

    def test_small_numerical_errors(self):
        # Test case with small numerical errors (within tolerance)
        completions = [
            [
                {
                    "content": "<answer>\n{(1.000001,-1.000001),(2.01,-2.01),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)}\n</answer>"
                }
            ]
        ]
        ground_truth = [
            "(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)"
        ]
        rewards = accuracy_reward(completions, ground_truth)
        self.assertEqual(rewards[0], 1.0)

    def test_partial_correct(self):
        # Test case with only some coordinates correct
        completions = [
            [
                {
                    "content": "<answer>\n{(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(20.0,-20.0)}\n</answer>"
                }
            ]
        ]
        ground_truth = [
            "(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)"
        ]
        rewards = accuracy_reward(completions, ground_truth)
        self.assertEqual(rewards[0], 0.9)  # 9 out of 10 correct = 0.9

    def test_wrong_format(self):
        # Test case with wrong format (missing curly braces)
        completions = [
            [
                {
                    "content": "<answer>\n(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)\n</answer>"
                }
            ]
        ]
        ground_truth = [
            "(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)"
        ]
        rewards = accuracy_reward(completions, ground_truth)
        self.assertEqual(rewards[0], 0.0)

    def test_missing_answer_tags(self):
        # Test case with missing answer tags
        completions = [
            [
                {
                    "content": "{(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)}"
                }
            ]
        ]
        ground_truth = [
            "(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)"
        ]
        rewards = accuracy_reward(completions, ground_truth)
        self.assertEqual(rewards[0], 0.0)

    def test_wrong_number_of_coordinates(self):
        # Test case with wrong number of coordinates
        completions = [
            [
                {
                    "content": "<answer>\n{(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0)}\n</answer>"
                }
            ]
        ]
        ground_truth = [
            "(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)"
        ]
        rewards = accuracy_reward(completions, ground_truth)
        self.assertEqual(rewards[0], 0.0)

    def test_invalid_coordinate_format(self):
        # Test case with invalid coordinate format
        completions = [
            [
                {
                    "content": "<answer>\n{(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(invalid)}\n</answer>"
                }
            ]
        ]
        ground_truth = [
            "(1.0,-1.0),(2.0,-2.0),(3.0,-3.0),(4.0,-4.0),(5.0,-5.0),(6.0,-6.0),(7.0,-7.0),(8.0,-8.0),(9.0,-9.0),(10.0,-10.0)"
        ]
        rewards = accuracy_reward(completions, ground_truth)
        self.assertEqual(rewards[0], 0.0)


if __name__ == "__main__":
    unittest.main()
