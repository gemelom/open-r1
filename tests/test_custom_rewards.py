import unittest
from src.open_r1.custom_rewards import accuracy_reward, format_reward


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


class TestFormatReward(unittest.TestCase):
    def test_perfect_format(self):
        # Test case with perfect format (both think and answer tags)
        completions = [
            [
                {
                    "content": "<think>\nThis is my reasoning.\n</think>\n<answer>\nThis is my answer.\n</answer>"
                }
            ]
        ]
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 1.0)

    def test_missing_think_tags(self):
        # Test case with missing think tags
        completions = [[{"content": "<answer>\nThis is my answer.\n</answer>"}]]
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 0.0)

    def test_missing_answer_tags(self):
        # Test case with missing answer tags
        completions = [[{"content": "<think>\nThis is my reasoning.\n</think>"}]]
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 0.0)

    def test_wrong_tag_order(self):
        # Test case with wrong tag order (answer before think)
        completions = [
            [
                {
                    "content": "<answer>\nThis is my answer.\n</answer>\n<think>\nThis is my reasoning.\n</think>"
                }
            ]
        ]
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 0.0)

    def test_empty_content(self):
        # Test case with empty content
        completions = [[{"content": ""}]]
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 0.0)

    def test_malformed_tags(self):
        # Test case with malformed tags
        completions = [
            [
                {
                    "content": "<think>This is my reasoning.</think>\n<answer>This is my answer.</answer>"
                }
            ]
        ]
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 0.0)

    def test_extra_content(self):
        # Test case with extra content outside tags
        completions = [
            [
                {
                    "content": "Some text before\n<think>\nThis is my reasoning.\n</think>\n<answer>\nThis is my answer.\n</answer>\nSome text after"
                }
            ]
        ]
        rewards = format_reward(completions)
        self.assertEqual(rewards[0], 0.0)


if __name__ == "__main__":
    unittest.main()
