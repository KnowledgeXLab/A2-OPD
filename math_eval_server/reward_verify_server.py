import os
import socket
import subprocess
import logging
import threading
import ipaddress
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError as FuturesTimeoutError

from math_verify import parse, verify


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reward_verify_server")


MATH_VERIFY_HOST = os.getenv("MATH_VERIFY_HOST", "0.0.0.0")
MATH_VERIFY_PORT = int(os.getenv("MATH_VERIFY_PORT", "7683"))
MATH_VERIFY_WORKERS = int(os.getenv("MATH_VERIFY_WORKERS", "16"))
MATH_VERIFY_TIMEOUT = float(os.getenv("MATH_VERIFY_TIMEOUT", "3"))
MATH_VERIFY_MAX_TASKS = int(os.getenv("MATH_VERIFY_MAX_TASKS", "4096"))
MATH_VERIFY_MAX_INFLIGHT = int(os.getenv("MATH_VERIFY_MAX_INFLIGHT", str(MATH_VERIFY_WORKERS * 2)))
MATH_VERIFY_QUEUE_TIMEOUT = float(os.getenv("MATH_VERIFY_QUEUE_TIMEOUT", "2"))
MAX_ANSWER_LEN = int(os.getenv("MATH_VERIFY_MAX_ANSWER_LEN", "100"))


class VerifyRequest(BaseModel):
    ground_truth: str
    answer: str
    strict: bool = True


class VerifyResponse(BaseModel):
    ok: bool
    result: bool = False
    timed_out: bool = False
    busy: bool = False
    error: Optional[str] = None


def _is_valid_ipv4(ip: str) -> bool:
    try:
        obj = ipaddress.ip_address(ip.strip())
        return obj.version == 4 and not obj.is_loopback and not obj.is_link_local
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_machine_ips() -> tuple[str, ...]:
    ips = []

    try:
        output = subprocess.check_output(
            ["hostname", "-I"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        for item in output.strip().split():
            if _is_valid_ipv4(item) and item not in ips:
                ips.append(item)
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if _is_valid_ipv4(ip) and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    for target in ["8.8.8.8", "1.1.1.1"]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            s.connect((target, 80))
            ip = s.getsockname()[0]
            s.close()
            if _is_valid_ipv4(ip) and ip not in ips:
                ips.append(ip)
        except Exception:
            pass

    return tuple(ips)


def print_server_address():
    hostname = socket.gethostname()
    ips = get_machine_ips()

    print("=" * 80, flush=True)
    print("[Math-Verify Reward Server]", flush=True)
    print(f"Hostname: {hostname}", flush=True)
    print(f"Bind host: {MATH_VERIFY_HOST}", flush=True)
    print(f"Port: {MATH_VERIFY_PORT}", flush=True)
    print(f"Workers: {MATH_VERIFY_WORKERS}", flush=True)
    print(f"Task timeout: {MATH_VERIFY_TIMEOUT}", flush=True)

    if ips:
        print("Detected server URLs:", flush=True)
        for ip in ips:
            print(f"  http://{ip}:{MATH_VERIFY_PORT}", flush=True)
    else:
        print("Detected server URLs: <none>", flush=True)

    print("=" * 80, flush=True)


@lru_cache(maxsize=20000)
def _parse_ground_truth_cached(ground_truth: str):
    from math_verify import parse

    return parse(
        "\\boxed{" + ground_truth + "}",
        parsing_timeout=None,
        raise_on_error=False,
    )


def _verify_job(ground_truth: str, answer: str, strict: bool = True) -> bool:
    

    try:
        if answer is None:
            return False

        if len(answer) > MAX_ANSWER_LEN:
            answer = answer[:MAX_ANSWER_LEN]

        gold = _parse_ground_truth_cached(ground_truth)

        target = parse(
            "\\boxed{" + answer + "}",
            parsing_timeout=None,
            raise_on_error=False,
        )

        if not gold or not target:
            return False

        result = verify(
            gold,
            target,
            strict=strict,
            timeout_seconds=None,
            raise_on_error=False,
        )

        return bool(result)

    except BaseException:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print_server_address()

    logger.info(
        "Starting math-verify process pool: workers=%s, timeout=%s, max_tasks=%s, max_inflight=%s",
        MATH_VERIFY_WORKERS,
        MATH_VERIFY_TIMEOUT,
        MATH_VERIFY_MAX_TASKS,
        MATH_VERIFY_MAX_INFLIGHT,
    )

    app.state.pool = ProcessPool(
        max_workers=MATH_VERIFY_WORKERS,
        max_tasks=MATH_VERIFY_MAX_TASKS,
    )
    app.state.semaphore = threading.BoundedSemaphore(MATH_VERIFY_MAX_INFLIGHT)

    try:
        yield
    finally:
        logger.info("Stopping math-verify process pool...")
        app.state.pool.stop()
        app.state.pool.join()
        logger.info("Math-verify process pool stopped.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "host": MATH_VERIFY_HOST,
        "port": MATH_VERIFY_PORT,
        "workers": MATH_VERIFY_WORKERS,
        "timeout": MATH_VERIFY_TIMEOUT,
        "max_tasks": MATH_VERIFY_MAX_TASKS,
        "max_inflight": MATH_VERIFY_MAX_INFLIGHT,
        "urls": [f"http://{ip}:{MATH_VERIFY_PORT}" for ip in get_machine_ips()],
    }


@app.post("/verify", response_model=VerifyResponse)
def verify_endpoint(req: VerifyRequest):
    acquired = app.state.semaphore.acquire(timeout=MATH_VERIFY_QUEUE_TIMEOUT)

    if not acquired:
        return VerifyResponse(
            ok=False,
            result=False,
            busy=True,
            error="server busy",
        )

    try:
        future = app.state.pool.schedule(
            _verify_job,
            args=(req.ground_truth, req.answer, req.strict),
            timeout=MATH_VERIFY_TIMEOUT,
        )

        try:
            result = future.result()
            return VerifyResponse(
                ok=True,
                result=bool(result),
            )

        except FuturesTimeoutError:
            return VerifyResponse(
                ok=False,
                result=False,
                timed_out=True,
                error="math verify timeout",
            )

        except ProcessExpired as e:
            return VerifyResponse(
                ok=False,
                result=False,
                error=f"worker process expired: {e}",
            )

        except BaseException as e:
            return VerifyResponse(
                ok=False,
                result=False,
                error=f"internal error: {type(e).__name__}",
            )

    finally:
        app.state.semaphore.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reward_verify_server:app",
        host=MATH_VERIFY_HOST,
        port=MATH_VERIFY_PORT,
        workers=1,
        log_level="info",
    )