#!/usr/bin/env python
"""Trim one of CUDAARCHS, CF_TORCH_CUDA_ARCH_LIST, or NVCC_GENCODE to a minimum architecture.

Prints the trimmed value of a single named environment variable to stdout, and a report of
which architectures were dropped to stderr. A subprocess cannot modify its parent shell's
environment directly, so the caller captures stdout via command substitution and assigns it
itself; this works the same way in bash, PowerShell, and cmd.exe. If the variable is unset or
empty, nothing is printed to stdout (the assignment becomes empty) and nothing to stderr.

Usage (bash):
    export CUDAARCHS="$(python trim_cuda_archs.py CUDAARCHS 7.5)"

Usage (PowerShell):
    $env:CUDAARCHS = python trim_cuda_archs.py CUDAARCHS 7.5

Usage (cmd.exe):
    for /f "delims=" %i in ('python trim_cuda_archs.py CUDAARCHS 7.5') do set CUDAARCHS=%i

Expected formats (see conda-forge/cuda-nvcc-feedstock's conda_build_config.yaml and
conda-forge/pytorch-cpu-feedstock's activate.sh, which are the source of these variables):
    CUDAARCHS               "75-real;80-real;90a-real;100f-real;121-virtual" (a bare "75" is
                             CMake shorthand for "75-real;75-virtual" and is also accepted)
    CF_TORCH_CUDA_ARCH_LIST  "7.5 8.0 9.0+PTX" (';' is also accepted as a separator, and an
                             arch may carry an "a" suffix, e.g. "9.0a", for arch-conditional
                             targets)
    NVCC_GENCODE             "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_90a,code=sm_90a"

Exit codes: 0 = trimmed (or the variable was unset/empty), 2 = bad invocation.
"""

import argparse
import os
import re
import sys

ARCH_MIN_RE = re.compile(r"^([1-9][0-9]*)\.([0-9])$")
# 75-real, 90a-real, 100f-real, 121-virtual, ... ; a bare "75" (no -real/-virtual) is CMake
# shorthand for "75-real;75-virtual", so it is valid too and must be trimmed the same way.
CUDAARCHS_TOKEN_RE = re.compile(r"^(\d+)[a-z]*(?:-(?:real|virtual))?$")
# 7.5, 8.6+PTX, 9.0a, 9.0a+PTX, ... ("a" marks an arch-conditional target, e.g. sm_90a)
TORCH_ARCH_TOKEN_RE = re.compile(r"^(\d+)\.(\d+)[a-z]?(?:\+PTX)?$")
# pytorch's cpp_extension.py accepts TORCH_CUDA_ARCH_LIST separated by spaces or semicolons
# (it does `.replace(' ', ';')` before splitting on ';'); CF_TORCH_CUDA_ARCH_LIST is a sed
# substitution of that same variable, so it carries the same rule.
TORCH_ARCH_LIST_SEP_RE = re.compile(r"[ ;]+")
# -gencode arch=compute_90a,code=sm_90a
GENCODE_CLAUSE_RE = re.compile(r"-gencode\s+arch=compute_(\d+)[a-z]*,code=\S+")


def parse_arch(text):
    """Convert a dotted architecture such as '7.5' into the key 75."""
    match = ARCH_MIN_RE.match(text.strip())
    if match is None:
        return None
    return int(match.group(1)) * 10 + int(match.group(2))


def trim_cudaarchs(value, min_arch):
    """Split CUDAARCHS into (kept, dropped) tokens (e.g. '90a-real') around min_arch.

    Unrecognized tokens are kept, since we cannot judge whether they belong.
    """
    kept, dropped = [], []
    for token in value.split(";"):
        match = CUDAARCHS_TOKEN_RE.match(token)
        if match is None or int(match.group(1)) >= min_arch:
            kept.append(token)
        else:
            dropped.append(token)
    return ";".join(kept), dropped


def trim_torch_arch_list(value, min_arch):
    """Split TORCH_CUDA_ARCH_LIST into (kept, dropped) entries (e.g. '8.6+PTX') around min_arch.

    Accepts space- or semicolon-separated input and re-joins with ';' if the input used
    any, else with ' ', matching whichever style was already in use.
    """
    sep = ";" if ";" in value else " "
    kept, dropped = [], []
    for token in TORCH_ARCH_LIST_SEP_RE.split(value.strip()):
        if not token:
            continue
        match = TORCH_ARCH_TOKEN_RE.match(token)
        if match is None:
            kept.append(token)
            continue
        arch = int(match.group(1)) * 10 + int(match.group(2))
        (kept if arch >= min_arch else dropped).append(token)
    return sep.join(kept), dropped


def trim_nvcc_gencode(value, min_arch):
    """Split NVCC_GENCODE into (kept, dropped) -gencode clauses around min_arch."""
    kept, dropped = [], []
    for match in GENCODE_CLAUSE_RE.finditer(value):
        clause = match.group(0)
        (kept if int(match.group(1)) >= min_arch else dropped).append(clause)
    return " ".join(kept), dropped


TRIMMERS = {
    "CUDAARCHS": trim_cudaarchs,
    "CF_TORCH_CUDA_ARCH_LIST": trim_torch_arch_list,
    "NVCC_GENCODE": trim_nvcc_gencode,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Trim a CUDA-architecture environment variable to a minimum architecture."
    )
    parser.add_argument(
        "var",
        choices=sorted(TRIMMERS),
        metavar="VAR",
        help="environment variable to trim: {}".format(", ".join(sorted(TRIMMERS))),
    )
    parser.add_argument("min_arch", metavar="ARCH", help="minimum architecture to keep, dotted (e.g. 7.5)")
    args = parser.parse_args(argv)

    min_arch = parse_arch(args.min_arch)
    if min_arch is None:
        parser.error('ARCH must look like 7.5, got "{}"'.format(args.min_arch))

    value = os.environ.get(args.var)
    if not value:
        return 0

    kept, dropped = TRIMMERS[args.var](value, min_arch)
    print(kept)
    if dropped:
        print("{}: dropped {}".format(args.var, " ".join(dropped)), file=sys.stderr)
    else:
        print("{}: unchanged".format(args.var), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Unit tests. Run with: python -m unittest trim_cuda_archs
# --------------------------------------------------------------------------

import io
import unittest
import unittest.mock

CUDAARCHS_EXAMPLE = "75-real;80-real;86-real;89-real;90a-real;100f-real;103f-real;120a-real;121f-real;121-virtual"
NVCC_GENCODE_EXAMPLE = (
    "-gencode arch=compute_75,code=sm_75 "
    "-gencode arch=compute_80,code=sm_80 "
    "-gencode arch=compute_86,code=sm_86 "
    "-gencode arch=compute_89,code=sm_89 "
    "-gencode arch=compute_90a,code=sm_90a "
    "-gencode arch=compute_100f,code=sm_100f "
    "-gencode arch=compute_103f,code=sm_103f "
    "-gencode arch=compute_120a,code=sm_120a "
    "-gencode arch=compute_121f,code=sm_121f "
    "-gencode arch=compute_121,code=compute_121"
)
TORCH_ARCH_LIST_EXAMPLE = "5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"


class ParseArchTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_arch("7.5"), 75)
        self.assertEqual(parse_arch("10.0"), 100)
        self.assertEqual(parse_arch("12.1"), 121)

    def test_invalid(self):
        for text in ["75", "7", "7.5.0", "abc", ""]:
            self.assertIsNone(parse_arch(text))


class TrimCudaarchsTests(unittest.TestCase):
    def test_trims_below_minimum(self):
        kept, dropped = trim_cudaarchs(CUDAARCHS_EXAMPLE, 90)
        self.assertEqual(kept, "90a-real;100f-real;103f-real;120a-real;121f-real;121-virtual")
        self.assertEqual(dropped, ["75-real", "80-real", "86-real", "89-real"])

    def test_keeps_everything_at_floor(self):
        kept, dropped = trim_cudaarchs(CUDAARCHS_EXAMPLE, 0)
        self.assertEqual(kept, CUDAARCHS_EXAMPLE)
        self.assertEqual(dropped, [])

    def test_keeps_unrecognized_tokens(self):
        kept, dropped = trim_cudaarchs("bogus;90a-real", 90)
        self.assertEqual(kept, "bogus;90a-real")
        self.assertEqual(dropped, [])

    def test_trims_bare_shorthand_tokens(self):
        # A bare "75" (no -real/-virtual) is CMake shorthand for "75-real;75-virtual".
        # The same applies to family-specific bare tokens like "90a".
        kept, dropped = trim_cudaarchs("75;90a;100f-real", 90)
        self.assertEqual(kept, "90a;100f-real")
        self.assertEqual(dropped, ["75"])
        kept, dropped = trim_cudaarchs("75;90a;100f-real", 0)
        self.assertEqual(kept, "75;90a;100f-real")
        self.assertEqual(dropped, [])

    def test_out_of_order_and_duplicate_tokens_are_preserved(self):
        # Tokens are not assumed sorted or unique; each is judged on its own merits and
        # kept/dropped in place, so order and duplicates in the input carry through.
        kept, dropped = trim_cudaarchs("90a-real;75-real;90a-real;121-virtual", 90)
        self.assertEqual(kept, "90a-real;90a-real;121-virtual")
        self.assertEqual(dropped, ["75-real"])


class TrimTorchArchListTests(unittest.TestCase):
    def test_trims_below_minimum_and_keeps_ptx_suffix(self):
        kept, dropped = trim_torch_arch_list(TORCH_ARCH_LIST_EXAMPLE, 75)
        self.assertEqual(kept, "7.5 8.0 8.6+PTX")
        self.assertEqual(dropped, ["5.2", "6.0", "6.1", "7.0"])

    def test_keeps_everything_at_floor(self):
        kept, dropped = trim_torch_arch_list(TORCH_ARCH_LIST_EXAMPLE, 0)
        self.assertEqual(kept, TORCH_ARCH_LIST_EXAMPLE)
        self.assertEqual(dropped, [])

    def test_out_of_order_and_duplicate_tokens_are_preserved(self):
        kept, dropped = trim_torch_arch_list("9.0+PTX 7.5 9.0+PTX 8.0", 75)
        self.assertEqual(kept, "9.0+PTX 7.5 9.0+PTX 8.0")
        self.assertEqual(dropped, [])

    def test_semicolon_separator_is_accepted_and_preserved(self):
        # pytorch's cpp_extension.py accepts ';' as well as ' ' (it normalizes ' ' to ';').
        kept, dropped = trim_torch_arch_list("7.5;8.0;8.6+PTX", 80)
        self.assertEqual(kept, "8.0;8.6+PTX")
        self.assertEqual(dropped, ["7.5"])

    def test_arch_conditional_suffix_is_kept_and_compared(self):
        # e.g. "9.0a" targets sm_90a specifically, as opposed to plain "9.0".
        kept, dropped = trim_torch_arch_list("7.5 9.0a", 90)
        self.assertEqual(kept, "9.0a")
        self.assertEqual(dropped, ["7.5"])
        kept, dropped = trim_torch_arch_list("7.5 9.0a+PTX", 90)
        self.assertEqual(kept, "9.0a+PTX")
        self.assertEqual(dropped, ["7.5"])


class TrimNvccGencodeTests(unittest.TestCase):
    def test_trims_below_minimum(self):
        kept, dropped = trim_nvcc_gencode(NVCC_GENCODE_EXAMPLE, 90)
        self.assertEqual(
            kept,
            "-gencode arch=compute_90a,code=sm_90a "
            "-gencode arch=compute_100f,code=sm_100f "
            "-gencode arch=compute_103f,code=sm_103f "
            "-gencode arch=compute_120a,code=sm_120a "
            "-gencode arch=compute_121f,code=sm_121f "
            "-gencode arch=compute_121,code=compute_121",
        )
        self.assertEqual(
            dropped,
            [
                "-gencode arch=compute_75,code=sm_75",
                "-gencode arch=compute_80,code=sm_80",
                "-gencode arch=compute_86,code=sm_86",
                "-gencode arch=compute_89,code=sm_89",
            ],
        )

    def test_out_of_order_and_duplicate_tokens_are_preserved(self):
        kept, dropped = trim_nvcc_gencode(
            "-gencode arch=compute_90a,code=sm_90a "
            "-gencode arch=compute_75,code=sm_75 "
            "-gencode arch=compute_90a,code=sm_90a",
            90,
        )
        self.assertEqual(
            kept, "-gencode arch=compute_90a,code=sm_90a -gencode arch=compute_90a,code=sm_90a"
        )
        self.assertEqual(dropped, ["-gencode arch=compute_75,code=sm_75"])

    def test_keeps_everything_at_floor(self):
        kept, dropped = trim_nvcc_gencode(NVCC_GENCODE_EXAMPLE, 0)
        self.assertEqual(kept, NVCC_GENCODE_EXAMPLE)
        self.assertEqual(dropped, [])


class MainTests(unittest.TestCase):
    def test_unset_or_empty_var_prints_nothing(self):
        env = {"CUDAARCHS": ""}
        with unittest.mock.patch.dict(os.environ, env, clear=True), unittest.mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout, unittest.mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            rc = main(["CUDAARCHS", "7.5"])
        self.assertEqual(rc, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_prints_bare_value_to_stdout_and_report_to_stderr(self):
        env = {"CF_TORCH_CUDA_ARCH_LIST": TORCH_ARCH_LIST_EXAMPLE}
        with unittest.mock.patch.dict(os.environ, env, clear=True), unittest.mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout, unittest.mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            rc = main(["CF_TORCH_CUDA_ARCH_LIST", "7.5"])
        self.assertEqual(rc, 0)
        self.assertEqual(stdout.getvalue().strip(), "7.5 8.0 8.6+PTX")
        self.assertIn("dropped 5.2 6.0 6.1 7.0", stderr.getvalue())

    def test_unchanged_is_reported_on_stderr(self):
        env = {"CF_TORCH_CUDA_ARCH_LIST": "8.0 8.6+PTX"}
        with unittest.mock.patch.dict(os.environ, env, clear=True), unittest.mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ), unittest.mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            main(["CF_TORCH_CUDA_ARCH_LIST", "7.5"])
        self.assertIn("unchanged", stderr.getvalue())

    def test_bad_arch_exits_2(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["CUDAARCHS", "not-an-arch"])
        self.assertEqual(ctx.exception.code, 2)

    def test_bad_var_name_exits_2(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["NOT_A_REAL_VAR", "7.5"])
        self.assertEqual(ctx.exception.code, 2)
