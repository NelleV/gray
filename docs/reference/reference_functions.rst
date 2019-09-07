*gray* functions
=================

*Contents are subject to change.*

.. currentmodule:: gray

Assertions and checks
---------------------

.. autofunction:: gray.assert_equivalent

.. autofunction:: gray.assert_stable

.. autofunction:: gray.can_be_split

.. autofunction:: gray.can_omit_invisible_parens

.. autofunction:: gray.is_empty_tuple

.. autofunction:: gray.is_import

.. autofunction:: gray.is_line_short_enough

.. autofunction:: gray.is_multiline_string

.. autofunction:: gray.is_one_tuple

.. autofunction:: gray.is_python36

.. autofunction:: gray.is_split_after_delimiter

.. autofunction:: gray.is_split_before_delimiter

.. autofunction:: gray.is_stub_body

.. autofunction:: gray.is_stub_suite

.. autofunction:: gray.is_vararg

.. autofunction:: gray.is_yield


Formatting
----------

.. autofunction:: gray.format_file_contents

.. autofunction:: gray.format_file_in_place

.. autofunction:: gray.format_stdin_to_stdout

.. autofunction:: gray.format_str

.. autofunction:: gray.reformat_one

.. autofunction:: gray.schedule_formatting

File operations
---------------

.. autofunction:: gray.dump_to_file

.. autofunction:: gray.find_project_root

.. autofunction:: gray.gen_python_files_in_dir

.. autofunction:: gray.read_pyproject_toml

Parsing
-------

.. autofunction:: gray.decode_bytes

.. autofunction:: gray.lib2to3_parse

.. autofunction:: gray.lib2to3_unparse

Split functions
---------------

.. autofunction:: gray.bracket_split_build_line

.. autofunction:: gray.bracket_split_succeeded_or_raise

.. autofunction:: gray.delimiter_split

.. autofunction:: gray.left_hand_split

.. autofunction:: gray.right_hand_split

.. autofunction:: gray.standalone_comment_split

.. autofunction:: gray.split_line

Caching
-------

.. autofunction:: gray.filter_cached

.. autofunction:: gray.get_cache_file

.. autofunction:: gray.get_cache_info

.. autofunction:: gray.read_cache

.. autofunction:: gray.write_cache

Utilities
---------

.. py:function:: gray.DebugVisitor.show(code: str) -> None

    Pretty-print the lib2to3 AST of a given string of `code`.

.. autofunction:: gray.cancel

.. autofunction:: gray.child_towards

.. autofunction:: gray.container_of

.. autofunction:: gray.convert_one_fmt_off_pair

.. autofunction:: gray.diff

.. autofunction:: gray.dont_increase_indentation

.. autofunction:: gray.format_float_or_int_string

.. autofunction:: gray.format_int_string

.. autofunction:: gray.ensure_visible

.. autofunction:: gray.enumerate_reversed

.. autofunction:: gray.enumerate_with_length

.. autofunction:: gray.generate_comments

.. autofunction:: gray.generate_ignored_nodes

.. autofunction:: gray.generate_trailers_to_omit

.. autofunction:: gray.get_future_imports

.. autofunction:: gray.list_comments

.. autofunction:: gray.make_comment

.. autofunction:: gray.maybe_make_parens_invisible_in_atom

.. autofunction:: gray.max_delimiter_priority_in_atom

.. autofunction:: gray.normalize_fmt_off

.. autofunction:: gray.normalize_numeric_literal

.. autofunction:: gray.normalize_prefix

.. autofunction:: gray.normalize_string_prefix

.. autofunction:: gray.normalize_string_quotes

.. autofunction:: gray.normalize_invisible_parens

.. autofunction:: gray.patch_click

.. autofunction:: gray.preceding_leaf

.. autofunction:: gray.re_compile_maybe_verbose

.. autofunction:: gray.should_explode

.. autofunction:: gray.shutdown

.. autofunction:: gray.sub_twice

.. autofunction:: gray.whitespace
