#!/usr/bin/env python3

import json
import pathlib
import subprocess
import sys


def run(command, parse_json=False):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return json.loads(result.stdout) if parse_json else result.stdout


def main():
    binary = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./sycl-bench")
    binary = binary.resolve()
    devices = run([str(binary), "devices", "--format", "json"], True)
    tested = 0
    for device in devices:
        if not device["matrix"]:
            continue
        selector = device["selector"]
        reports = run(
            [
                str(binary),
                "benchmark",
                "--device",
                selector,
                "--tests",
                "matrix-fp16,matrix-bf16,matrix-tf32,matrix-int8,matrix-fp64",
                "--format",
                "json",
            ],
            True,
        )
        report = reports[0]
        errors = [
            item for item in report["measurements"] if item["status"] == "error"
        ]
        successful = [
            item for item in report["measurements"] if item["status"] == "ok"
        ]
        if report["status"] != "pass" or errors or not successful:
            raise RuntimeError(f"matrix smoke test failed for {selector}: {report}")

        bf16_ok = any(
            item["name"] == "BF16->FP32" and item["status"] == "ok"
            for item in report["measurements"]
        )
        if bf16_ok:
            run(
                [
                    str(binary),
                    "stress",
                    "--device",
                    selector,
                    "--duration",
                    "1s",
                    "--profile",
                    "compute",
                    "--compute-workload",
                    "matrix-bf16",
                    "--report-interval",
                    "1s",
                ]
            )
        tested += 1
        print(f"matrix smoke: {selector}: pass")

    print(f"smoke test passed ({tested} matrix-capable device(s))")


if __name__ == "__main__":
    main()
