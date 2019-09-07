#!/usr/bin/env python3
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from io import BytesIO, TextIOWrapper
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Generator, List, Tuple, Iterator, TypeVar
import unittest
from unittest.mock import patch, MagicMock

from click import unstyle
from click.testing import CliRunner

import gray
from gray import Feature, TargetVersion

try:
    from gray import grayd
    from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
    from aiohttp import web
except ImportError:
    has_grayd_deps = False
else:
    has_grayd_deps = True

ff = partial(gray.format_file_in_place, mode=gray.FileMode(), fast=True)
fs = partial(gray.format_str, mode=gray.FileMode())
THIS_FILE = Path(__file__)
THIS_DIR = THIS_FILE.parent
EMPTY_LINE = "# EMPTY LINE WITH WHITESPACE" + " (this comment will be removed)"
PY36_ARGS = [
    f"--target-version={version.name.lower()}" for version in gray.PY36_VERSIONS
]
T = TypeVar("T")
R = TypeVar("R")


def dump_to_stderr(*output: str) -> str:
    return "\n" + "\n".join(output) + "\n"


def read_data(name: str, data: bool = True) -> Tuple[str, str]:
    """read_data('test_name') -> 'input', 'output'"""
    if not name.endswith((".py", ".pyi", ".out", ".diff")):
        name += ".py"
    _input: List[str] = []
    _output: List[str] = []
    base_dir = THIS_DIR / "data" if data else THIS_DIR
    with open(base_dir / name, "r", encoding="utf8") as test:
        lines = test.readlines()
    result = _input
    for line in lines:
        line = line.replace(EMPTY_LINE, "")
        if line.rstrip() == "# output":
            result = _output
            continue

        result.append(line)
    if _input and not _output:
        # If there's no output marker, treat the entire file as already pre-formatted.
        _output = _input[:]
    return "".join(_input).strip() + "\n", "".join(_output).strip() + "\n"


@contextmanager
def cache_dir(exists: bool = True) -> Iterator[Path]:
    with TemporaryDirectory() as workspace:
        cache_dir = Path(workspace)
        if not exists:
            cache_dir = cache_dir / "new"
        with patch("gray.CACHE_DIR", cache_dir):
            yield cache_dir


@contextmanager
def event_loop(close: bool) -> Iterator[None]:
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield

    finally:
        if close:
            loop.close()


@contextmanager
def skip_if_exception(e: str) -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        if exc.__class__.__name__ == e:
            unittest.skip(f"Encountered expected exception {exc}, skipping")
        else:
            raise


class grayRunner(CliRunner):
    """Modify CliRunner so that stderr is not merged with stdout.

    This is a hack that can be removed once we depend on Click 7.x"""

    def __init__(self) -> None:
        self.stderrbuf = BytesIO()
        self.stdoutbuf = BytesIO()
        self.stdout_bytes = b""
        self.stderr_bytes = b""
        super().__init__()

    @contextmanager
    def isolation(self, *args: Any, **kwargs: Any) -> Generator[BinaryIO, None, None]:
        with super().isolation(*args, **kwargs) as output:
            try:
                hold_stderr = sys.stderr
                sys.stderr = TextIOWrapper(self.stderrbuf, encoding=self.charset)
                yield output
            finally:
                self.stdout_bytes = sys.stdout.buffer.getvalue()  # type: ignore
                self.stderr_bytes = sys.stderr.buffer.getvalue()  # type: ignore
                sys.stderr = hold_stderr


class grayTestCase(unittest.TestCase):
    maxDiff = None

    def assertFormatEqual(self, expected: str, actual: str) -> None:
        if actual != expected and not os.environ.get("SKIP_AST_PRINT"):
            bdv: gray.DebugVisitor[Any]
            gray.out("Expected tree:", fg="green")
            try:
                exp_node = gray.lib2to3_parse(expected)
                bdv = gray.DebugVisitor()
                list(bdv.visit(exp_node))
            except Exception as ve:
                gray.err(str(ve))
            gray.out("Actual tree:", fg="red")
            try:
                exp_node = gray.lib2to3_parse(actual)
                bdv = gray.DebugVisitor()
                list(bdv.visit(exp_node))
            except Exception as ve:
                gray.err(str(ve))
        self.assertEqual(expected, actual)

    def invokegray(
        self, args: List[str], exit_code: int = 0, ignore_config: bool = True
    ) -> None:
        runner = grayRunner()
        if ignore_config:
            args = ["--config", str(THIS_DIR / "empty.toml"), *args]
        result = runner.invoke(gray.main, args)
        self.assertEqual(result.exit_code, exit_code, msg=runner.stderr_bytes.decode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_empty(self) -> None:
        source = expected = ""
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    def test_empty_ff(self) -> None:
        expected = ""
        tmp_file = Path(gray.dump_to_file())
        try:
            self.assertFalse(ff(tmp_file, write_back=gray.WriteBack.YES))
            with open(tmp_file, encoding="utf8") as f:
                actual = f.read()
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)

    @patch("gray.dump_to_file", dump_to_stderr)
    @unittest.expectedFailure
    def test_self(self) -> None:
        source, expected = read_data("test_gray", data=False)
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        self.assertFalse(ff(THIS_FILE))

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_gray(self) -> None:
        source, expected = read_data("../__init__", data=False)
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        self.assertFalse(ff(THIS_DIR / ".." / "__init__.py"))

    def test_piping(self) -> None:
        source, expected = read_data("../__init__.py", data=False)
        result = grayRunner().invoke(
            gray.main,
            ["-", "--fast", f"--line-length={gray.DEFAULT_LINE_LENGTH}"],
            input=BytesIO(source.encode("utf8")),
        )
        self.assertEqual(result.exit_code, 0)
        self.assertFormatEqual(expected, result.output)
        gray.assert_equivalent(source, result.output)
        gray.assert_stable(source, result.output, gray.FileMode())

    def test_piping_diff(self) -> None:
        diff_header = re.compile(
            rf"(STDIN|STDOUT)\t\d\d\d\d-\d\d-\d\d "
            rf"\d\d:\d\d:\d\d\.\d\d\d\d\d\d \+\d\d\d\d"
        )
        source, _ = read_data("expression.py")
        expected, _ = read_data("expression.diff")
        config = THIS_DIR / "data" / "empty_pyproject.toml"
        args = [
            "-",
            "--fast",
            f"--line-length={gray.DEFAULT_LINE_LENGTH}",
            "--diff",
            f"--config={config}",
        ]
        result = grayRunner().invoke(
            gray.main, args, input=BytesIO(source.encode("utf8"))
        )
        self.assertEqual(result.exit_code, 0)
        actual = diff_header.sub("[Deterministic header]", result.output)
        actual = actual.rstrip() + "\n"  # the diff output has a trailing space
        self.assertEqual(expected, actual)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_setup(self) -> None:
        source, expected = read_data("../../setup.py", data=False)
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        self.assertFalse(ff(THIS_DIR / ".." / ".." / "setup.py"))

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_function(self) -> None:
        source, expected = read_data("function")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_function2(self) -> None:
        source, expected = read_data("function2")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_function_trailing_comma(self) -> None:
        source, expected = read_data("function_trailing_comma")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_expression(self) -> None:
        source, expected = read_data("expression")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_pep_572(self) -> None:
        source, expected = read_data("pep_572")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        if sys.version_info >= (3, 8):
            gray.assert_equivalent(source, actual)

    def test_pep_572_version_detection(self) -> None:
        source, _ = read_data("pep_572")
        root = gray.lib2to3_parse(source)
        features = gray.get_features_used(root)
        self.assertIn(gray.Feature.ASSIGNMENT_EXPRESSIONS, features)
        versions = gray.detect_target_versions(root)
        self.assertIn(gray.TargetVersion.PY38, versions)

    def test_expression_ff(self) -> None:
        source, expected = read_data("expression")
        tmp_file = Path(gray.dump_to_file(source))
        try:
            self.assertTrue(ff(tmp_file, write_back=gray.WriteBack.YES))
            with open(tmp_file, encoding="utf8") as f:
                actual = f.read()
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)
        with patch("gray.dump_to_file", dump_to_stderr):
            gray.assert_equivalent(source, actual)
            gray.assert_stable(source, actual, gray.FileMode())

    def test_expression_diff(self) -> None:
        source, _ = read_data("expression.py")
        expected, _ = read_data("expression.diff")
        tmp_file = Path(gray.dump_to_file(source))
        diff_header = re.compile(
            rf"{re.escape(str(tmp_file))}\t\d\d\d\d-\d\d-\d\d "
            rf"\d\d:\d\d:\d\d\.\d\d\d\d\d\d \+\d\d\d\d"
        )
        try:
            result = grayRunner().invoke(gray.main, ["--diff", str(tmp_file)])
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmp_file)
        actual = result.output
        actual = diff_header.sub("[Deterministic header]", actual)
        actual = actual.rstrip() + "\n"  # the diff output has a trailing space
        if expected != actual:
            dump = gray.dump_to_file(actual)
            msg = (
                f"Expected diff isn't equal to the actual. If you made changes "
                f"to expression.py and this is an anticipated difference, "
                f"overwrite tests/data/expression.diff with {dump}"
            )
            self.assertEqual(expected, actual, msg)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_fstring(self) -> None:
        source, expected = read_data("fstring")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_pep_570(self) -> None:
        source, expected = read_data("pep_570")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        if sys.version_info >= (3, 8):
            gray.assert_equivalent(source, actual)

    def test_detect_pos_only_arguments(self) -> None:
        source, _ = read_data("pep_570")
        root = gray.lib2to3_parse(source)
        features = gray.get_features_used(root)
        self.assertIn(gray.Feature.POS_ONLY_ARGUMENTS, features)
        versions = gray.detect_target_versions(root)
        self.assertIn(gray.TargetVersion.PY38, versions)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_string_quotes(self) -> None:
        source, expected = read_data("string_quotes")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        mode = gray.FileMode(string_normalization=False)
        not_normalized = fs(source, mode=mode)
        self.assertFormatEqual(source, not_normalized)
        gray.assert_equivalent(source, not_normalized)
        gray.assert_stable(source, not_normalized, mode=mode)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_slices(self) -> None:
        source, expected = read_data("slices")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comments(self) -> None:
        source, expected = read_data("comments")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comments2(self) -> None:
        source, expected = read_data("comments2")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comments3(self) -> None:
        source, expected = read_data("comments3")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comments4(self) -> None:
        source, expected = read_data("comments4")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comments5(self) -> None:
        source, expected = read_data("comments5")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comments6(self) -> None:
        source, expected = read_data("comments6")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comments7(self) -> None:
        source, expected = read_data("comments7")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_comment_after_escaped_newline(self) -> None:
        source, expected = read_data("comment_after_escaped_newline")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_cantfit(self) -> None:
        source, expected = read_data("cantfit")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_import_spacing(self) -> None:
        source, expected = read_data("import_spacing")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_composition(self) -> None:
        source, expected = read_data("composition")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_empty_lines(self) -> None:
        source, expected = read_data("empty_lines")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_remove_parens(self) -> None:
        source, expected = read_data("remove_parens")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_string_prefixes(self) -> None:
        source, expected = read_data("string_prefixes")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_numeric_literals(self) -> None:
        source, expected = read_data("numeric_literals")
        mode = gray.FileMode(target_versions=gray.PY36_VERSIONS)
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, mode)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_numeric_literals_ignoring_underscores(self) -> None:
        source, expected = read_data("numeric_literals_skip_underscores")
        mode = gray.FileMode(target_versions=gray.PY36_VERSIONS)
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, mode)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_numeric_literals_py2(self) -> None:
        source, expected = read_data("numeric_literals_py2")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_python2(self) -> None:
        source, expected = read_data("python2")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_python2_print_function(self) -> None:
        source, expected = read_data("python2_print_function")
        mode = gray.FileMode(target_versions={TargetVersion.PY27})
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, mode)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_python2_unicode_literals(self) -> None:
        source, expected = read_data("python2_unicode_literals")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_stub(self) -> None:
        mode = gray.FileMode(is_pyi=True)
        source, expected = read_data("stub.pyi")
        actual = fs(source, mode=mode)
        self.assertFormatEqual(expected, actual)
        gray.assert_stable(source, actual, mode)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_async_as_identifier(self) -> None:
        source_path = (THIS_DIR / "data" / "async_as_identifier.py").resolve()
        source, expected = read_data("async_as_identifier")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        major, minor = sys.version_info[:2]
        if major < 3 or (major <= 3 and minor < 7):
            gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        # ensure gray can parse this when the target is 3.6
        self.invokegray([str(source_path), "--target-version", "py36"])
        # but not on 3.7, because async/await is no longer an identifier
        self.invokegray([str(source_path), "--target-version", "py37"], exit_code=123)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_python37(self) -> None:
        source_path = (THIS_DIR / "data" / "python37.py").resolve()
        source, expected = read_data("python37")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        major, minor = sys.version_info[:2]
        if major > 3 or (major == 3 and minor >= 7):
            gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())
        # ensure gray can parse this when the target is 3.7
        self.invokegray([str(source_path), "--target-version", "py37"])
        # but not on 3.6, because we use async as a reserved keyword
        self.invokegray([str(source_path), "--target-version", "py36"], exit_code=123)

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_fmtonoff(self) -> None:
        source, expected = read_data("fmtonoff")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_fmtonoff2(self) -> None:
        source, expected = read_data("fmtonoff2")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_remove_empty_parentheses_after_class(self) -> None:
        source, expected = read_data("class_blank_parentheses")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_new_line_between_class_and_code(self) -> None:
        source, expected = read_data("class_methods_new_line")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_bracket_match(self) -> None:
        source, expected = read_data("bracketmatch")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_tuple_assign(self) -> None:
        source, expected = read_data("tupleassign")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    @patch("gray.dump_to_file", dump_to_stderr)
    def test_beginning_backslash(self) -> None:
        source, expected = read_data("beginning_backslash")
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        gray.assert_equivalent(source, actual)
        gray.assert_stable(source, actual, gray.FileMode())

    def test_tab_comment_indentation(self) -> None:
        contents_tab = "if 1:\n\tif 2:\n\t\tpass\n\t# comment\n\tpass\n"
        contents_spc = "if 1:\n    if 2:\n        pass\n    # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))

        contents_tab = "if 1:\n\tif 2:\n\t\tpass\n\t\t# comment\n\tpass\n"
        contents_spc = "if 1:\n    if 2:\n        pass\n        # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))

        # mixed tabs and spaces (valid Python 2 code)
        contents_tab = "if 1:\n        if 2:\n\t\tpass\n\t# comment\n        pass\n"
        contents_spc = "if 1:\n    if 2:\n        pass\n    # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))

        contents_tab = "if 1:\n        if 2:\n\t\tpass\n\t\t# comment\n        pass\n"
        contents_spc = "if 1:\n    if 2:\n        pass\n        # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))

    def test_report_verbose(self) -> None:
        report = gray.Report(verbose=True)
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("gray.out", out), patch("gray.err", err):
            report.done(Path("f1"), gray.Changed.NO)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "f1 already well formatted, good job.")
            self.assertEqual(unstyle(str(report)), "1 file left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.done(Path("f2"), gray.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "reformatted f2")
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 1 file left unchanged."
            )
            report.done(Path("f3"), gray.Changed.CACHED)
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(
                out_lines[-1], "f3 wasn't modified on disk since last run."
            )
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 2 files left unchanged."
            )
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path("e1"), "boom")
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], "error: cannot format e1: boom")
            self.assertEqual(
                unstyle(str(report)),
                "1 file reformatted, 2 files left unchanged, "
                "1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f3"), gray.Changed.YES)
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[-1], "reformatted f3")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.failed(Path("e2"), "boom")
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], "error: cannot format e2: boom")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path("wat"), "no match")
            self.assertEqual(len(out_lines), 5)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(out_lines[-1], "wat ignored: no match")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f4"), gray.Changed.NO)
            self.assertEqual(len(out_lines), 6)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(out_lines[-1], "f4 already well formatted, good job.")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 3 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, "
                "2 files would fail to reformat.",
            )

    def test_report_quiet(self) -> None:
        report = gray.Report(quiet=True)
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("gray.out", out), patch("gray.err", err):
            report.done(Path("f1"), gray.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), "1 file left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.done(Path("f2"), gray.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 1 file left unchanged."
            )
            report.done(Path("f3"), gray.Changed.CACHED)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 2 files left unchanged."
            )
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path("e1"), "boom")
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], "error: cannot format e1: boom")
            self.assertEqual(
                unstyle(str(report)),
                "1 file reformatted, 2 files left unchanged, "
                "1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f3"), gray.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.failed(Path("e2"), "boom")
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], "error: cannot format e2: boom")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path("wat"), "no match")
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f4"), gray.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 3 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, "
                "2 files would fail to reformat.",
            )

    def test_report_normal(self) -> None:
        report = gray.Report()
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("gray.out", out), patch("gray.err", err):
            report.done(Path("f1"), gray.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), "1 file left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.done(Path("f2"), gray.Changed.YES)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "reformatted f2")
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 1 file left unchanged."
            )
            report.done(Path("f3"), gray.Changed.CACHED)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "reformatted f2")
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 2 files left unchanged."
            )
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path("e1"), "boom")
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], "error: cannot format e1: boom")
            self.assertEqual(
                unstyle(str(report)),
                "1 file reformatted, 2 files left unchanged, "
                "1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f3"), gray.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[-1], "reformatted f3")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.failed(Path("e2"), "boom")
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], "error: cannot format e2: boom")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path("wat"), "no match")
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f4"), gray.Changed.NO)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 3 files left unchanged, "
                "2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, "
                "2 files would fail to reformat.",
            )

    def test_lib2to3_parse(self) -> None:
        with self.assertRaises(gray.InvalidInput):
            gray.lib2to3_parse("invalid syntax")

        straddling = "x + y"
        gray.lib2to3_parse(straddling)
        gray.lib2to3_parse(straddling, {TargetVersion.PY27})
        gray.lib2to3_parse(straddling, {TargetVersion.PY36})
        gray.lib2to3_parse(straddling, {TargetVersion.PY27, TargetVersion.PY36})

        py2_only = "print x"
        gray.lib2to3_parse(py2_only)
        gray.lib2to3_parse(py2_only, {TargetVersion.PY27})
        with self.assertRaises(gray.InvalidInput):
            gray.lib2to3_parse(py2_only, {TargetVersion.PY36})
        with self.assertRaises(gray.InvalidInput):
            gray.lib2to3_parse(py2_only, {TargetVersion.PY27, TargetVersion.PY36})

        py3_only = "exec(x, end=y)"
        gray.lib2to3_parse(py3_only)
        with self.assertRaises(gray.InvalidInput):
            gray.lib2to3_parse(py3_only, {TargetVersion.PY27})
        gray.lib2to3_parse(py3_only, {TargetVersion.PY36})
        gray.lib2to3_parse(py3_only, {TargetVersion.PY27, TargetVersion.PY36})

    def test_get_features_used(self) -> None:
        node = gray.lib2to3_parse("def f(*, arg): ...\n")
        self.assertEqual(gray.get_features_used(node), set())
        node = gray.lib2to3_parse("def f(*, arg,): ...\n")
        self.assertEqual(gray.get_features_used(node), {Feature.TRAILING_COMMA_IN_DEF})
        node = gray.lib2to3_parse("f(*arg,)\n")
        self.assertEqual(
            gray.get_features_used(node), {Feature.TRAILING_COMMA_IN_CALL}
        )
        node = gray.lib2to3_parse("def f(*, arg): f'string'\n")
        self.assertEqual(gray.get_features_used(node), {Feature.F_STRINGS})
        node = gray.lib2to3_parse("123_456\n")
        self.assertEqual(gray.get_features_used(node), {Feature.NUMERIC_UNDERSCORES})
        node = gray.lib2to3_parse("123456\n")
        self.assertEqual(gray.get_features_used(node), set())
        source, expected = read_data("function")
        node = gray.lib2to3_parse(source)
        expected_features = {
            Feature.TRAILING_COMMA_IN_CALL,
            Feature.TRAILING_COMMA_IN_DEF,
            Feature.F_STRINGS,
        }
        self.assertEqual(gray.get_features_used(node), expected_features)
        node = gray.lib2to3_parse(expected)
        self.assertEqual(gray.get_features_used(node), expected_features)
        source, expected = read_data("expression")
        node = gray.lib2to3_parse(source)
        self.assertEqual(gray.get_features_used(node), set())
        node = gray.lib2to3_parse(expected)
        self.assertEqual(gray.get_features_used(node), set())

    def test_get_future_imports(self) -> None:
        node = gray.lib2to3_parse("\n")
        self.assertEqual(set(), gray.get_future_imports(node))
        node = gray.lib2to3_parse("from __future__ import gray\n")
        self.assertEqual({"gray"}, gray.get_future_imports(node))
        node = gray.lib2to3_parse("from __future__ import multiple, imports\n")
        self.assertEqual({"multiple", "imports"}, gray.get_future_imports(node))
        node = gray.lib2to3_parse("from __future__ import (parenthesized, imports)\n")
        self.assertEqual({"parenthesized", "imports"}, gray.get_future_imports(node))
        node = gray.lib2to3_parse(
            "from __future__ import multiple\nfrom __future__ import imports\n"
        )
        self.assertEqual({"multiple", "imports"}, gray.get_future_imports(node))
        node = gray.lib2to3_parse("# comment\nfrom __future__ import gray\n")
        self.assertEqual({"gray"}, gray.get_future_imports(node))
        node = gray.lib2to3_parse('"""docstring"""\nfrom __future__ import gray\n')
        self.assertEqual({"gray"}, gray.get_future_imports(node))
        node = gray.lib2to3_parse("some(other, code)\nfrom __future__ import gray\n")
        self.assertEqual(set(), gray.get_future_imports(node))
        node = gray.lib2to3_parse("from some.module import gray\n")
        self.assertEqual(set(), gray.get_future_imports(node))
        node = gray.lib2to3_parse(
            "from __future__ import unicode_literals as _unicode_literals"
        )
        self.assertEqual({"unicode_literals"}, gray.get_future_imports(node))
        node = gray.lib2to3_parse(
            "from __future__ import unicode_literals as _lol, print"
        )
        self.assertEqual({"unicode_literals", "print"}, gray.get_future_imports(node))

    def test_debug_visitor(self) -> None:
        source, _ = read_data("debug_visitor.py")
        expected, _ = read_data("debug_visitor.out")
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("gray.out", out), patch("gray.err", err):
            gray.DebugVisitor.show(source)
        actual = "\n".join(out_lines) + "\n"
        log_name = ""
        if expected != actual:
            log_name = gray.dump_to_file(*out_lines)
        self.assertEqual(
            expected,
            actual,
            f"AST print out is different. Actual version dumped to {log_name}",
        )

    def test_format_file_contents(self) -> None:
        empty = ""
        mode = gray.FileMode()
        with self.assertRaises(gray.NothingChanged):
            gray.format_file_contents(empty, mode=mode, fast=False)
        just_nl = "\n"
        with self.assertRaises(gray.NothingChanged):
            gray.format_file_contents(just_nl, mode=mode, fast=False)
        same = "j = [1, 2, 3]\n"
        with self.assertRaises(gray.NothingChanged):
            gray.format_file_contents(same, mode=mode, fast=False)
        different = "j = [1,2,3]"
        expected = same
        actual = gray.format_file_contents(different, mode=mode, fast=False)
        self.assertEqual(expected, actual)
        invalid = "return if you can"
        with self.assertRaises(gray.InvalidInput) as e:
            gray.format_file_contents(invalid, mode=mode, fast=False)
        self.assertEqual(str(e.exception), "Cannot parse: 1:7: return if you can")

    def test_endmarker(self) -> None:
        n = gray.lib2to3_parse("\n")
        self.assertEqual(n.type, gray.syms.file_input)
        self.assertEqual(len(n.children), 1)
        self.assertEqual(n.children[0].type, gray.token.ENDMARKER)

    @unittest.skipIf(os.environ.get("SKIP_AST_PRINT"), "user set SKIP_AST_PRINT")
    def test_assertFormatEqual(self) -> None:
        out_lines = []
        err_lines = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("gray.out", out), patch("gray.err", err):
            with self.assertRaises(AssertionError):
                self.assertFormatEqual("j = [1, 2, 3]", "j = [1, 2, 3,]")

        out_str = "".join(out_lines)
        self.assertTrue("Expected tree:" in out_str)
        self.assertTrue("Actual tree:" in out_str)
        self.assertEqual("".join(err_lines), "")

    def test_cache_broken_file(self) -> None:
        mode = gray.FileMode()
        with cache_dir() as workspace:
            cache_file = gray.get_cache_file(mode)
            with cache_file.open("w") as fobj:
                fobj.write("this is not a pickle")
            self.assertEqual(gray.read_cache(mode), {})
            src = (workspace / "test.py").resolve()
            with src.open("w") as fobj:
                fobj.write("print('hello')")
            self.invokegray([str(src)])
            cache = gray.read_cache(mode)
            self.assertIn(src, cache)

    def test_cache_single_file_already_cached(self) -> None:
        mode = gray.FileMode()
        with cache_dir() as workspace:
            src = (workspace / "test.py").resolve()
            with src.open("w") as fobj:
                fobj.write("print('hello')")
            gray.write_cache({}, [src], mode)
            self.invokegray([str(src)])
            with src.open("r") as fobj:
                self.assertEqual(fobj.read(), "print('hello')")

    @event_loop(close=False)
    def test_cache_multiple_files(self) -> None:
        mode = gray.FileMode()
        with cache_dir() as workspace, patch(
            "gray.ProcessPoolExecutor", new=ThreadPoolExecutor
        ):
            one = (workspace / "one.py").resolve()
            with one.open("w") as fobj:
                fobj.write("print('hello')")
            two = (workspace / "two.py").resolve()
            with two.open("w") as fobj:
                fobj.write("print('hello')")
            gray.write_cache({}, [one], mode)
            self.invokegray([str(workspace)])
            with one.open("r") as fobj:
                self.assertEqual(fobj.read(), "print('hello')")
            with two.open("r") as fobj:
                self.assertEqual(fobj.read(), 'print("hello")\n')
            cache = gray.read_cache(mode)
            self.assertIn(one, cache)
            self.assertIn(two, cache)

    def test_no_cache_when_writeback_diff(self) -> None:
        mode = gray.FileMode()
        with cache_dir() as workspace:
            src = (workspace / "test.py").resolve()
            with src.open("w") as fobj:
                fobj.write("print('hello')")
            self.invokegray([str(src), "--diff"])
            cache_file = gray.get_cache_file(mode)
            self.assertFalse(cache_file.exists())

    def test_no_cache_when_stdin(self) -> None:
        mode = gray.FileMode()
        with cache_dir():
            result = CliRunner().invoke(
                gray.main, ["-"], input=BytesIO(b"print('hello')")
            )
            self.assertEqual(result.exit_code, 0)
            cache_file = gray.get_cache_file(mode)
            self.assertFalse(cache_file.exists())

    def test_read_cache_no_cachefile(self) -> None:
        mode = gray.FileMode()
        with cache_dir():
            self.assertEqual(gray.read_cache(mode), {})

    def test_write_cache_read_cache(self) -> None:
        mode = gray.FileMode()
        with cache_dir() as workspace:
            src = (workspace / "test.py").resolve()
            src.touch()
            gray.write_cache({}, [src], mode)
            cache = gray.read_cache(mode)
            self.assertIn(src, cache)
            self.assertEqual(cache[src], gray.get_cache_info(src))

    def test_filter_cached(self) -> None:
        with TemporaryDirectory() as workspace:
            path = Path(workspace)
            uncached = (path / "uncached").resolve()
            cached = (path / "cached").resolve()
            cached_but_changed = (path / "changed").resolve()
            uncached.touch()
            cached.touch()
            cached_but_changed.touch()
            cache = {cached: gray.get_cache_info(cached), cached_but_changed: (0.0, 0)}
            todo, done = gray.filter_cached(
                cache, {uncached, cached, cached_but_changed}
            )
            self.assertEqual(todo, {uncached, cached_but_changed})
            self.assertEqual(done, {cached})

    def test_write_cache_creates_directory_if_needed(self) -> None:
        mode = gray.FileMode()
        with cache_dir(exists=False) as workspace:
            self.assertFalse(workspace.exists())
            gray.write_cache({}, [], mode)
            self.assertTrue(workspace.exists())

    @event_loop(close=False)
    def test_failed_formatting_does_not_get_cached(self) -> None:
        mode = gray.FileMode()
        with cache_dir() as workspace, patch(
            "gray.ProcessPoolExecutor", new=ThreadPoolExecutor
        ):
            failing = (workspace / "failing.py").resolve()
            with failing.open("w") as fobj:
                fobj.write("not actually python")
            clean = (workspace / "clean.py").resolve()
            with clean.open("w") as fobj:
                fobj.write('print("hello")\n')
            self.invokegray([str(workspace)], exit_code=123)
            cache = gray.read_cache(mode)
            self.assertNotIn(failing, cache)
            self.assertIn(clean, cache)

    def test_write_cache_write_fail(self) -> None:
        mode = gray.FileMode()
        with cache_dir(), patch.object(Path, "open") as mock:
            mock.side_effect = OSError
            gray.write_cache({}, [], mode)

    @event_loop(close=False)
    def test_check_diff_use_together(self) -> None:
        with cache_dir():
            # Files which will be reformatted.
            src1 = (THIS_DIR / "data" / "string_quotes.py").resolve()
            self.invokegray([str(src1), "--diff", "--check"], exit_code=1)
            # Files which will not be reformatted.
            src2 = (THIS_DIR / "data" / "composition.py").resolve()
            self.invokegray([str(src2), "--diff", "--check"])
            # Multi file command.
            self.invokegray([str(src1), str(src2), "--diff", "--check"], exit_code=1)

    def test_no_files(self) -> None:
        with cache_dir():
            # Without an argument, gray exits with error code 0.
            self.invokegray([])

    def test_broken_symlink(self) -> None:
        with cache_dir() as workspace:
            symlink = workspace / "broken_link.py"
            try:
                symlink.symlink_to("nonexistent.py")
            except OSError as e:
                self.skipTest(f"Can't create symlinks: {e}")
            self.invokegray([str(workspace.resolve())])

    def test_read_cache_line_lengths(self) -> None:
        mode = gray.FileMode()
        short_mode = gray.FileMode(line_length=1)
        with cache_dir() as workspace:
            path = (workspace / "file.py").resolve()
            path.touch()
            gray.write_cache({}, [path], mode)
            one = gray.read_cache(mode)
            self.assertIn(path, one)
            two = gray.read_cache(short_mode)
            self.assertNotIn(path, two)

    def test_single_file_force_pyi(self) -> None:
        reg_mode = gray.FileMode()
        pyi_mode = gray.FileMode(is_pyi=True)
        contents, expected = read_data("force_pyi")
        with cache_dir() as workspace:
            path = (workspace / "file.py").resolve()
            with open(path, "w") as fh:
                fh.write(contents)
            self.invokegray([str(path), "--pyi"])
            with open(path, "r") as fh:
                actual = fh.read()
            # verify cache with --pyi is separate
            pyi_cache = gray.read_cache(pyi_mode)
            self.assertIn(path, pyi_cache)
            normal_cache = gray.read_cache(reg_mode)
            self.assertNotIn(path, normal_cache)
        self.assertEqual(actual, expected)

    @event_loop(close=False)
    def test_multi_file_force_pyi(self) -> None:
        reg_mode = gray.FileMode()
        pyi_mode = gray.FileMode(is_pyi=True)
        contents, expected = read_data("force_pyi")
        with cache_dir() as workspace:
            paths = [
                (workspace / "file1.py").resolve(),
                (workspace / "file2.py").resolve(),
            ]
            for path in paths:
                with open(path, "w") as fh:
                    fh.write(contents)
            self.invokegray([str(p) for p in paths] + ["--pyi"])
            for path in paths:
                with open(path, "r") as fh:
                    actual = fh.read()
                self.assertEqual(actual, expected)
            # verify cache with --pyi is separate
            pyi_cache = gray.read_cache(pyi_mode)
            normal_cache = gray.read_cache(reg_mode)
            for path in paths:
                self.assertIn(path, pyi_cache)
                self.assertNotIn(path, normal_cache)

    def test_pipe_force_pyi(self) -> None:
        source, expected = read_data("force_pyi")
        result = CliRunner().invoke(
            gray.main, ["-", "-q", "--pyi"], input=BytesIO(source.encode("utf8"))
        )
        self.assertEqual(result.exit_code, 0)
        actual = result.output
        self.assertFormatEqual(actual, expected)

    def test_single_file_force_py36(self) -> None:
        reg_mode = gray.FileMode()
        py36_mode = gray.FileMode(target_versions=gray.PY36_VERSIONS)
        source, expected = read_data("force_py36")
        with cache_dir() as workspace:
            path = (workspace / "file.py").resolve()
            with open(path, "w") as fh:
                fh.write(source)
            self.invokegray([str(path), *PY36_ARGS])
            with open(path, "r") as fh:
                actual = fh.read()
            # verify cache with --target-version is separate
            py36_cache = gray.read_cache(py36_mode)
            self.assertIn(path, py36_cache)
            normal_cache = gray.read_cache(reg_mode)
            self.assertNotIn(path, normal_cache)
        self.assertEqual(actual, expected)

    @event_loop(close=False)
    def test_multi_file_force_py36(self) -> None:
        reg_mode = gray.FileMode()
        py36_mode = gray.FileMode(target_versions=gray.PY36_VERSIONS)
        source, expected = read_data("force_py36")
        with cache_dir() as workspace:
            paths = [
                (workspace / "file1.py").resolve(),
                (workspace / "file2.py").resolve(),
            ]
            for path in paths:
                with open(path, "w") as fh:
                    fh.write(source)
            self.invokegray([str(p) for p in paths] + PY36_ARGS)
            for path in paths:
                with open(path, "r") as fh:
                    actual = fh.read()
                self.assertEqual(actual, expected)
            # verify cache with --target-version is separate
            pyi_cache = gray.read_cache(py36_mode)
            normal_cache = gray.read_cache(reg_mode)
            for path in paths:
                self.assertIn(path, pyi_cache)
                self.assertNotIn(path, normal_cache)

    def test_pipe_force_py36(self) -> None:
        source, expected = read_data("force_py36")
        result = CliRunner().invoke(
            gray.main,
            ["-", "-q", "--target-version=py36"],
            input=BytesIO(source.encode("utf8")),
        )
        self.assertEqual(result.exit_code, 0)
        actual = result.output
        self.assertFormatEqual(actual, expected)

    def test_include_exclude(self) -> None:
        path = THIS_DIR / "data" / "include_exclude_tests"
        include = re.compile(r"\.pyi?$")
        exclude = re.compile(r"/exclude/|/\.definitely_exclude/")
        report = gray.Report()
        sources: List[Path] = []
        expected = [
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/dont_exclude/a.pyi"),
        ]
        this_abs = THIS_DIR.resolve()
        sources.extend(
            gray.gen_python_files_in_dir(path, this_abs, include, exclude, report)
        )
        self.assertEqual(sorted(expected), sorted(sources))

    def test_empty_include(self) -> None:
        path = THIS_DIR / "data" / "include_exclude_tests"
        report = gray.Report()
        empty = re.compile(r"")
        sources: List[Path] = []
        expected = [
            Path(path / "b/exclude/a.pie"),
            Path(path / "b/exclude/a.py"),
            Path(path / "b/exclude/a.pyi"),
            Path(path / "b/dont_exclude/a.pie"),
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/dont_exclude/a.pyi"),
            Path(path / "b/.definitely_exclude/a.pie"),
            Path(path / "b/.definitely_exclude/a.py"),
            Path(path / "b/.definitely_exclude/a.pyi"),
        ]
        this_abs = THIS_DIR.resolve()
        sources.extend(
            gray.gen_python_files_in_dir(
                path, this_abs, empty, re.compile(gray.DEFAULT_EXCLUDES), report
            )
        )
        self.assertEqual(sorted(expected), sorted(sources))

    def test_empty_exclude(self) -> None:
        path = THIS_DIR / "data" / "include_exclude_tests"
        report = gray.Report()
        empty = re.compile(r"")
        sources: List[Path] = []
        expected = [
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/dont_exclude/a.pyi"),
            Path(path / "b/exclude/a.py"),
            Path(path / "b/exclude/a.pyi"),
            Path(path / "b/.definitely_exclude/a.py"),
            Path(path / "b/.definitely_exclude/a.pyi"),
        ]
        this_abs = THIS_DIR.resolve()
        sources.extend(
            gray.gen_python_files_in_dir(
                path, this_abs, re.compile(gray.DEFAULT_INCLUDES), empty, report
            )
        )
        self.assertEqual(sorted(expected), sorted(sources))

    def test_invalid_include_exclude(self) -> None:
        for option in ["--include", "--exclude"]:
            self.invokegray(["-", option, "**()(!!*)"], exit_code=2)

    def test_preserves_line_endings(self) -> None:
        with TemporaryDirectory() as workspace:
            test_file = Path(workspace) / "test.py"
            for nl in ["\n", "\r\n"]:
                contents = nl.join(["def f(  ):", "    pass"])
                test_file.write_bytes(contents.encode())
                ff(test_file, write_back=gray.WriteBack.YES)
                updated_contents: bytes = test_file.read_bytes()
                self.assertIn(nl.encode(), updated_contents)
                if nl == "\n":
                    self.assertNotIn(b"\r\n", updated_contents)

    def test_preserves_line_endings_via_stdin(self) -> None:
        for nl in ["\n", "\r\n"]:
            contents = nl.join(["def f(  ):", "    pass"])
            runner = grayRunner()
            result = runner.invoke(
                gray.main, ["-", "--fast"], input=BytesIO(contents.encode("utf8"))
            )
            self.assertEqual(result.exit_code, 0)
            output = runner.stdout_bytes
            self.assertIn(nl.encode("utf8"), output)
            if nl == "\n":
                self.assertNotIn(b"\r\n", output)

    def test_assert_equivalent_different_asts(self) -> None:
        with self.assertRaises(AssertionError):
            gray.assert_equivalent("{}", "None")

    def test_symlink_out_of_root_directory(self) -> None:
        path = MagicMock()
        root = THIS_DIR
        child = MagicMock()
        include = re.compile(gray.DEFAULT_INCLUDES)
        exclude = re.compile(gray.DEFAULT_EXCLUDES)
        report = gray.Report()
        # `child` should behave like a symlink which resolved path is clearly
        # outside of the `root` directory.
        path.iterdir.return_value = [child]
        child.resolve.return_value = Path("/a/b/c")
        child.is_symlink.return_value = True
        try:
            list(gray.gen_python_files_in_dir(path, root, include, exclude, report))
        except ValueError as ve:
            self.fail(f"`get_python_files_in_dir()` failed: {ve}")
        path.iterdir.assert_called_once()
        child.resolve.assert_called_once()
        child.is_symlink.assert_called_once()
        # `child` should behave like a strange file which resolved path is clearly
        # outside of the `root` directory.
        child.is_symlink.return_value = False
        with self.assertRaises(ValueError):
            list(gray.gen_python_files_in_dir(path, root, include, exclude, report))
        path.iterdir.assert_called()
        self.assertEqual(path.iterdir.call_count, 2)
        child.resolve.assert_called()
        self.assertEqual(child.resolve.call_count, 2)
        child.is_symlink.assert_called()
        self.assertEqual(child.is_symlink.call_count, 2)

    def test_shhh_click(self) -> None:
        try:
            from click import _unicodefun  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Incompatible Click version")
        if not hasattr(_unicodefun, "_verify_python3_env"):
            self.skipTest("Incompatible Click version")
        # First, let's see if Click is crashing with a preferred ASCII charset.
        with patch("locale.getpreferredencoding") as gpe:
            gpe.return_value = "ASCII"
            with self.assertRaises(RuntimeError):
                _unicodefun._verify_python3_env()
        # Now, let's silence Click...
        gray.patch_click()
        # ...and confirm it's silent.
        with patch("locale.getpreferredencoding") as gpe:
            gpe.return_value = "ASCII"
            try:
                _unicodefun._verify_python3_env()
            except RuntimeError as re:
                self.fail(f"`patch_click()` failed, exception still raised: {re}")

    def test_root_logger_not_used_directly(self) -> None:
        def fail(*args: Any, **kwargs: Any) -> None:
            self.fail("Record created with root logger")

        with patch.multiple(
            logging.root,
            debug=fail,
            info=fail,
            warning=fail,
            error=fail,
            critical=fail,
            log=fail,
        ):
            ff(THIS_FILE)

    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    def test_grayd_main(self) -> None:
        with patch("gray.grayd.web.run_app"):
            result = CliRunner().invoke(grayd.main, [])
            if result.exception is not None:
                raise result.exception
            self.assertEqual(result.exit_code, 0)


class grayDTestCase(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        return grayd.make_app()

    # TODO: remove these decorators once the below is released
    # https://github.com/aio-libs/aiohttp/pull/3727
    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_request_needs_formatting(self) -> None:
        response = await self.client.post("/", data=b"print('hello world')")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.charset, "utf8")
        self.assertEqual(await response.read(), b'print("hello world")\n')

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_request_no_change(self) -> None:
        response = await self.client.post("/", data=b'print("hello world")\n')
        self.assertEqual(response.status, 204)
        self.assertEqual(await response.read(), b"")

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_request_syntax_error(self) -> None:
        response = await self.client.post("/", data=b"what even ( is")
        self.assertEqual(response.status, 400)
        content = await response.text()
        self.assertTrue(
            content.startswith("Cannot parse"),
            msg=f"Expected error to start with 'Cannot parse', got {repr(content)}",
        )

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_unsupported_version(self) -> None:
        response = await self.client.post(
            "/", data=b"what", headers={grayd.VERSION_HEADER: "2"}
        )
        self.assertEqual(response.status, 501)

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_supported_version(self) -> None:
        response = await self.client.post(
            "/", data=b"what", headers={grayd.VERSION_HEADER: "1"}
        )
        self.assertEqual(response.status, 200)

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_invalid_python_variant(self) -> None:
        async def check(header_value: str, expected_status: int = 400) -> None:
            response = await self.client.post(
                "/", data=b"what", headers={grayd.PYTHON_VARIANT_HEADER: header_value}
            )
            self.assertEqual(response.status, expected_status)

        await check("lol")
        await check("ruby3.5")
        await check("pyi3.6")
        await check("py1.5")
        await check("2.8")
        await check("py2.8")
        await check("3.0")
        await check("pypy3.0")
        await check("jython3.4")

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_pyi(self) -> None:
        source, expected = read_data("stub.pyi")
        response = await self.client.post(
            "/", data=source, headers={grayd.PYTHON_VARIANT_HEADER: "pyi"}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.text(), expected)

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_python_variant(self) -> None:
        code = (
            "def f(\n"
            "    and_has_a_bunch_of,\n"
            "    very_long_arguments_too,\n"
            "    and_lots_of_them_as_well_lol,\n"
            "    **and_very_long_keyword_arguments\n"
            "):\n"
            "    pass\n"
        )

        async def check(header_value: str, expected_status: int) -> None:
            response = await self.client.post(
                "/", data=code, headers={grayd.PYTHON_VARIANT_HEADER: header_value}
            )
            self.assertEqual(
                response.status, expected_status, msg=await response.text()
            )

        await check("3.6", 200)
        await check("py3.6", 200)
        await check("3.6,3.7", 200)
        await check("3.6,py3.7", 200)
        await check("py36,py37", 200)
        await check("36", 200)
        await check("3.6.4", 200)

        await check("2", 204)
        await check("2.7", 204)
        await check("py2.7", 204)
        await check("3.4", 204)
        await check("py3.4", 204)
        await check("py34,py36", 204)
        await check("34", 204)

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_line_length(self) -> None:
        response = await self.client.post(
            "/", data=b'print("hello")\n', headers={grayd.LINE_LENGTH_HEADER: "7"}
        )
        self.assertEqual(response.status, 200)

    @skip_if_exception("ClientOSError")
    @unittest.skipUnless(has_grayd_deps, "grayd's dependencies are not installed")
    @unittest_run_loop
    async def test_grayd_invalid_line_length(self) -> None:
        response = await self.client.post(
            "/", data=b'print("hello")\n', headers={grayd.LINE_LENGTH_HEADER: "NaN"}
        )
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main(module="test_gray")
