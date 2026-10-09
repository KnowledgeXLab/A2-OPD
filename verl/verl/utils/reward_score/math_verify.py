# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from math_verify import parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


import os
import requests


# Set by training scripts via math_eval_server/start_math_verify_server.sh, or manually.
MATH_VERIFY_SERVER_URL = os.getenv(
    "MATH_VERIFY_SERVER_URL",
    "http://127.0.0.1:7642/verify",    # http://10.140.45.27:8008/verify, http://10.140.37.23:8132/verify, http://10.140.37.8:8132/verify
)

MATH_VERIFY_HTTP_CONNECT_TIMEOUT = float(
    os.getenv("MATH_VERIFY_HTTP_CONNECT_TIMEOUT", "1") 
)

MATH_VERIFY_HTTP_READ_TIMEOUT = float(
    os.getenv("MATH_VERIFY_HTTP_READ_TIMEOUT", "8")
)


def remote_math_verify(ground_truth: str, answer: str) -> bool:
    session = requests.Session()
    session.trust_env = False
    try:
        resp = session.post(
            MATH_VERIFY_SERVER_URL,
            json={
                "ground_truth": ground_truth,
                "answer": answer,
                "strict": True,
            },
            timeout=(
                MATH_VERIFY_HTTP_CONNECT_TIMEOUT,
                MATH_VERIFY_HTTP_READ_TIMEOUT,
            ),
        )

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    ) as e:
        raise RuntimeError(
            f"Math verify server is unavailable: {MATH_VERIFY_SERVER_URL} ({e})"
        ) from e

    if 500 <= resp.status_code < 600:
        raise RuntimeError(
            f"Math verify server returned HTTP {resp.status_code}"
        )

    if resp.status_code != 200:
        return False

    try:
        data = resp.json()
    except ValueError:
        return False

    return bool(data.get("result", False))


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None

# reward model
def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    result = False
    answer = remove_boxed(last_boxed_only_string(model_output))
    if answer is None:
        return 0.0

    if answer == ground_truth:
        return 1.0

    try:
        if len(answer) > 100:
            answer = answer[:100]
        result = remote_math_verify(ground_truth, answer)
    except Exception as e:
        print(f"[WARNING] math_verify remote call failed: {e}", flush=True)

    if result:
        return 1.0
    else:
        return 0.0

