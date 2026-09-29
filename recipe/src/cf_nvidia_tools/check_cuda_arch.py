#!/usr/bin/env python
"""Check that ``cuda_arch_version`` matches the lowest common CUDA architecture across a
set of binaries.

Usage:
    check-cuda-arch <file-or-glob> [<file-or-glob> ...]

The "lowest common CUDA architecture" is the lowest architecture that every binary in the
set still supports — not the lowest architecture present in any single binary. A binary
that happens to also carry kernels for older GPUs than the rest of the group does not lower
this value: the group as a whole cannot run on a GPU that any one of its binaries doesn't
support, so the floor is the highest of the per-file minimums, not the lowest.

The expected architecture is read from the ``cuda_arch_version`` environment variable
(dotted, e.g. ``8.2``) unless ``--arch-min`` is given.  ``cuobjdump`` must be on PATH. The
check fails if ``cuda_arch_version`` does not exactly match the computed lowest common
architecture, in either direction: too low means the recipe under-claims what the binaries
actually require; too high means the recipe claims support the binaries don't actually have.

Both SASS (``sm_XX``) and PTX (``compute_XX``) targets count; arch-conditional targets
(``sm_90a``, ``sm_100f``) count as their base architecture. Files without device code are
skipped.  The report lists SASS and PTX separately for the log, but the check uses the
lowest of the two.

Exit codes: 0 = match, 1 = mismatch or unreadable binary, 2 = bad invocation.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

# sm_90, compute_90, sm_90a, sm_100f, ...
ARCH_RE = re.compile(r"(?:sm|compute)_(\d{2,3})[af]?(?![0-9A-Za-z_])")
ARCH_MIN_RE = re.compile(r"^([1-9][0-9]*)\.([0-9])$")
NO_DEVICE_CODE = "does not contain device code"


def parse_arch(text):
    """Convert a dotted architecture such as '8.2' into the key 82."""
    match = ARCH_MIN_RE.match(text.strip())
    if match is None:
        return None
    return int(match.group(1)) * 10 + int(match.group(2))


def format_arch(key):
    """Convert an architecture key such as 82 into the dotted form '8.2'."""
    return "{}.{}".format(key // 10, key % 10)


def expand(patterns):
    """Expand globs, keeping literal arguments that contain no glob characters.

    Unix shells expand globs themselves; cmd.exe does not.  Literal arguments are kept even
    when they do not exist, so that a mistyped path is reported by name instead of being
    silently dropped; a glob that matches nothing is reported as "no files matched".
    """
    paths = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            paths.extend(sorted(glob.glob(pattern)))
        else:
            paths.append(pattern)
    return paths


def architectures(cuobjdump, path, flag):
    """Return (sorted arch keys, cuobjdump output) for one listing mode.

    ``flag`` is ``--list-elf`` for SASS or ``--list-ptx`` for PTX.  The two are
    listed separately because cuobjdump names both kinds of image ``sm_XX``, so
    they cannot be told apart once the output is merged.
    """
    result = subprocess.run(
        [cuobjdump, flag, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    keys = {int(digits) for digits in ARCH_RE.findall(result.stdout)}
    return sorted(keys), result.stdout


def render(keys):
    """Render architecture keys for the report, e.g. '7.5 8.0 9.0'."""
    return " ".join(format_arch(key) for key in keys) or "-"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Check that the lowest CUDA architecture in a set of binaries "
            "matches the expected minimum."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="FILE",
        help="binaries to inspect; globs are expanded",
    )
    parser.add_argument(
        "--arch-min",
        default=os.environ.get("cuda_arch_version"),
        help="expected minimum architecture, dotted (default: $cuda_arch_version)",
    )
    args = parser.parse_args(argv)

    if not args.arch_min:
        parser.error("cuda_arch_version is not set and --arch-min was not given")

    expected = parse_arch(args.arch_min)
    if expected is None:
        parser.error(
            'cuda_arch_version must look like 8.2, got "{}"'.format(args.arch_min)
        )

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        parser.error("cuobjdump was not found on PATH")

    paths = expand(args.paths)
    if not paths:
        parser.error("no files matched")

    print(
        "Checking CUDA architectures against cuda_arch_version={}".format(args.arch_min)
    )

    failed = False
    found = []

    for path in paths:
        name = os.path.basename(path)
        sass, sass_output = architectures(cuobjdump, path, "--list-elf")
        ptx, ptx_output = architectures(cuobjdump, path, "--list-ptx")
        keys = sorted(set(sass) | set(ptx))
        if not keys:
            output = sass_output + ptx_output
            if NO_DEVICE_CODE in output:
                print("  {} : no device code, skipped".format(name))
            else:
                print("  {} : ERROR - could not read device code".format(name))
                print(output.rstrip("\n"))
                failed = True
            continue
        # SASS and PTX are reported separately for the log; the check itself
        # uses the lowest of the two.
        print("  {} :".format(name))
        print("      SASS : {}".format(render(sass)))
        print("      PTX  : {}".format(render(ptx)))
        found.append((keys[0], name))

    if not found:
        print(
            "ERROR: none of the {} file(s) contain CUDA device code".format(len(paths))
        )
        return 1

    # The group is only usable on a GPU that every binary supports, so the floor
    # is the highest of the per-file minimums, not the lowest.
    minimum, _ = max(found, key=lambda item: item[0])

    print(
        "lowest common arch = {}   expected = {}".format(
            format_arch(minimum), format_arch(expected)
        )
    )

    if failed:
        print("FAILED: one or more files could not be inspected")
        return 1

    if minimum != expected:
        print(
            "FAILED: lowest common CUDA architecture is {}, expected {}".format(
                format_arch(minimum), format_arch(expected)
            )
        )
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Unit tests. Run with: python -m unittest cf_nvidia_tools.check_cuda_arch
# --------------------------------------------------------------------------

import io
import tempfile
import unittest
import unittest.mock


class ParseArchTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_arch("7.5"), 75)
        self.assertEqual(parse_arch("10.0"), 100)
        self.assertEqual(parse_arch("12.1"), 121)

    def test_invalid(self):
        for text in ["75", "7", "7.5.0", "abc", ""]:
            self.assertIsNone(parse_arch(text))


class FormatArchTests(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_arch(75), "7.5")
        self.assertEqual(format_arch(100), "10.0")
        self.assertEqual(format_arch(121), "12.1")


class ExpandTests(unittest.TestCase):
    def test_glob_is_expanded_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, name) for name in ("b.so", "a.so", "c.so")]
            for path in paths:
                open(path, "w").close()
            self.assertEqual(expand([os.path.join(tmp, "*.so")]), sorted(paths))

    def test_literal_argument_is_kept_even_if_missing(self):
        self.assertEqual(expand(["/no/such/file.so"]), ["/no/such/file.so"])

    def test_glob_matching_nothing_yields_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(expand([os.path.join(tmp, "*.so")]), [])


class RenderTests(unittest.TestCase):
    def test_renders_dotted_architectures(self):
        self.assertEqual(render([75, 80, 90]), "7.5 8.0 9.0")

    def test_empty_renders_as_dash(self):
        self.assertEqual(render([]), "-")


class ArchitecturesTests(unittest.TestCase):
    def _run(self, stdout):
        with unittest.mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout),
        ):
            return architectures("cuobjdump", "a.so", "--list-elf")

    def test_extracts_sm_and_compute_keys(self):
        keys, output = self._run("arch = sm_75\narch = compute_90\n")
        self.assertEqual(keys, [75, 90])
        self.assertEqual(output, "arch = sm_75\narch = compute_90\n")

    def test_arch_conditional_suffix_is_stripped(self):
        # sm_90a and sm_100f target base architectures 90 and 100 respectively.
        keys, _ = self._run("arch = sm_90a\narch = sm_100f\n")
        self.assertEqual(keys, [90, 100])

    def test_no_matches_yields_empty_keys(self):
        keys, output = self._run(NO_DEVICE_CODE + "\n")
        self.assertEqual(keys, [])
        self.assertEqual(output, NO_DEVICE_CODE + "\n")


class MainTests(unittest.TestCase):
    def _patched(self, sass_by_path, ptx_by_path=None, which="cuobjdump"):
        """Patch subprocess.run so --list-elf/--list-ptx return per-path canned stdout."""
        ptx_by_path = ptx_by_path or {}

        def fake_run(cmd, **kwargs):
            _, flag, path = cmd
            stdout = (sass_by_path if flag == "--list-elf" else ptx_by_path).get(path, "")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout)

        return (
            unittest.mock.patch("subprocess.run", side_effect=fake_run),
            unittest.mock.patch("shutil.which", return_value=which),
        )

    def test_missing_arch_min_exits_2(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                main(["a.so"])
        self.assertEqual(ctx.exception.code, 2)

    def test_bad_arch_min_exits_2(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--arch-min", "not-an-arch", "a.so"])
        self.assertEqual(ctx.exception.code, 2)

    def test_missing_cuobjdump_exits_2(self):
        with unittest.mock.patch("shutil.which", return_value=None):
            with self.assertRaises(SystemExit) as ctx:
                main(["--arch-min", "7.5", "a.so"])
        self.assertEqual(ctx.exception.code, 2)

    def test_no_files_matched_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch(
            "shutil.which", return_value="cuobjdump"
        ):
            with self.assertRaises(SystemExit) as ctx:
                main(["--arch-min", "7.5", os.path.join(tmp, "*.so")])
        self.assertEqual(ctx.exception.code, 2)

    def test_matching_architecture_returns_0(self):
        run_patch, which_patch = self._patched({"a.so": "arch = sm_75\n"})
        with run_patch, which_patch, unittest.mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            rc = main(["--arch-min", "7.5", "a.so"])
        self.assertEqual(rc, 0)
        self.assertIn("OK", stdout.getvalue())

    def test_mismatched_architecture_returns_1(self):
        run_patch, which_patch = self._patched({"a.so": "arch = sm_75\n"})
        with run_patch, which_patch, unittest.mock.patch("sys.stdout", new_callable=io.StringIO):
            rc = main(["--arch-min", "8.0", "a.so"])
        self.assertEqual(rc, 1)

    def test_no_device_code_in_any_file_returns_1(self):
        run_patch, which_patch = self._patched({"a.so": NO_DEVICE_CODE + "\n"})
        with run_patch, which_patch, unittest.mock.patch("sys.stdout", new_callable=io.StringIO):
            rc = main(["--arch-min", "7.5", "a.so"])
        self.assertEqual(rc, 1)

    def test_unreadable_binary_returns_1(self):
        run_patch, which_patch = self._patched({"a.so": "garbage output\n"})
        with run_patch, which_patch, unittest.mock.patch("sys.stdout", new_callable=io.StringIO):
            rc = main(["--arch-min", "7.5", "a.so"])
        self.assertEqual(rc, 1)

    def test_floor_is_the_highest_of_the_per_file_minimums(self):
        # A binary that also carries older kernels than the rest of the group must not lower
        # the computed floor: the group's floor is max(per-file minimum), not min(...).
        run_patch, which_patch = self._patched(
            {"a.so": "arch = sm_75\narch = sm_90\n", "b.so": "arch = sm_80\n"}
        )
        with run_patch, which_patch, unittest.mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            rc = main(["--arch-min", "8.0", "a.so", "b.so"])
        self.assertEqual(rc, 0)
        self.assertIn("lowest common arch = 8.0", stdout.getvalue())
