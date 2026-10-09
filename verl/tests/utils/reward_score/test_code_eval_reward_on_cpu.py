import unittest
from unittest.mock import patch

from verl.utils.reward_score.code_eval_reward import (
    _is_code,
    reward_func,
    reward_func_batched,
)


class _ImmediateExecutor:
    """Run ProcessPool tasks inline for unit tests."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args
        return False

    def submit(self, fn, arg):
        class _Future:
            def result(self):
                return fn(arg)

        return _Future()


class TestCodeEvalReward(unittest.TestCase):
    def test_is_code_by_source(self):
        self.assertTrue(_is_code("taco", None))
        self.assertFalse(_is_code("DeepMath-103K", None))

    @patch("verl.utils.reward_score.code_eval_reward._score_code")
    def test_reward_func_code(self, mock_score):
        mock_score.return_value = {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "pass_rate": 1.0,
        }
        result = reward_func("taco", "print(1)", {"inputs": [], "outputs": []})
        self.assertEqual(result["score"], 1.0)
        mock_score.assert_called_once()

    @patch("verl.utils.reward_score.code_eval_reward.as_completed", lambda futures: futures)
    @patch("verl.utils.reward_score.code_eval_reward.ProcessPoolExecutor", _ImmediateExecutor)
    @patch("verl.utils.reward_score.code_eval_reward._score_code")
    def test_reward_func_batched_parallel(self, mock_score):
        mock_score.side_effect = [
            {"score": 1.0, "format_score": 1.0, "acc": True, "pass_rate": 1.0},
            {"score": 0.0, "format_score": 1.0, "acc": False, "pass_rate": 0.0},
        ]
        results = reward_func_batched(
            data_sources=["taco", "apps"],
            solution_strs=["a", "b"],
            ground_truths=[{"inputs": [], "outputs": []}, {"inputs": [], "outputs": []}],
            code_eval_workers=2,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["score"], 1.0)
        self.assertEqual(results[1]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
