" gray.vim
" Author: Łukasz Langa
" Created: Mon Mar 26 23:27:53 2018 -0700
" Requires: Vim Ver7.0+
" Version:  1.1
"
" Documentation:
"   This plugin formats Python files.
"
" History:
"  1.0:
"    - initial version
"  1.1:
"    - restore cursor/window position after formatting

if v:version < 700 || !has('python3')
    echo "This script requires vim7.0+ with Python 3.6 support."
    finish
endif

if exists("g:load_gray")
   finish
endif

let g:load_gray = "py1.0"
if !exists("g:gray_virtualenv")
  let g:gray_virtualenv = "~/.vim/gray"
endif
if !exists("g:gray_fast")
  let g:gray_fast = 0
endif
if !exists("g:gray_linelength")
  let g:gray_linelength = 88
endif
if !exists("g:gray_skip_string_normalization")
  let g:gray_skip_string_normalization = 0
endif

python3 << endpython3
import os
import sys
import vim

def _get_python_binary(exec_prefix):
  try:
    default = vim.eval("g:pymode_python").strip()
  except vim.error:
    default = ""
  if default and os.path.exists(default):
    return default
  if sys.platform[:3] == "win":
    return exec_prefix / 'python.exe'
  return exec_prefix / 'bin' / 'python3'

def _get_pip(venv_path):
  if sys.platform[:3] == "win":
    return venv_path / 'Scripts' / 'pip.exe'
  return venv_path / 'bin' / 'pip'

def _get_virtualenv_site_packages(venv_path, pyver):
  if sys.platform[:3] == "win":
    return venv_path / 'Lib' / 'site-packages'
  return venv_path / 'lib' / f'python{pyver[0]}.{pyver[1]}' / 'site-packages'

def _initialize_gray_env(upgrade=False):
  pyver = sys.version_info[:2]
  if pyver < (3, 6):
    print("Sorry, gray requires Python 3.6+ to run.")
    return False

  from pathlib import Path
  import subprocess
  import venv
  virtualenv_path = Path(vim.eval("g:gray_virtualenv")).expanduser()
  virtualenv_site_packages = str(_get_virtualenv_site_packages(virtualenv_path, pyver))
  first_install = False
  if not virtualenv_path.is_dir():
    print('Please wait, one time setup for gray.')
    _executable = sys.executable
    try:
      sys.executable = str(_get_python_binary(Path(sys.exec_prefix)))
      print(f'Creating a virtualenv in {virtualenv_path}...')
      print('(this path can be customized in .vimrc by setting g:gray_virtualenv)')
      venv.create(virtualenv_path, with_pip=True)
    finally:
      sys.executable = _executable
    first_install = True
  if first_install:
    print('Installing gray with pip...')
  if upgrade:
    print('Upgrading gray with pip...')
  if first_install or upgrade:
    subprocess.run([str(_get_pip(virtualenv_path)), 'install', '-U', 'gray'], stdout=subprocess.PIPE)
    print('DONE! You are all set, thanks for waiting ✨ 🍰 ✨')
  if first_install:
    print('Pro-tip: to upgrade gray in the future, use the :grayUpgrade command and restart Vim.\n')
  if sys.path[0] != virtualenv_site_packages:
    sys.path.insert(0, virtualenv_site_packages)
  return True

if _initialize_gray_env():
  import gray
  import time

def gray():
  start = time.time()
  fast = bool(int(vim.eval("g:gray_fast")))
  mode = gray.FileMode(
    line_length=int(vim.eval("g:gray_linelength")),
    string_normalization=not bool(int(vim.eval("g:gray_skip_string_normalization"))),
    is_pyi=vim.current.buffer.name.endswith('.pyi'),
  )
  buffer_str = '\n'.join(vim.current.buffer) + '\n'
  try:
    new_buffer_str = gray.format_file_contents(buffer_str, fast=fast, mode=mode)
  except gray.NothingChanged:
    print(f'Already well formatted, good job. (took {time.time() - start:.4f}s)')
  except Exception as exc:
    print(exc)
  else:
    cursor = vim.current.window.cursor
    vim.current.buffer[:] = new_buffer_str.split('\n')[:-1]
    try:
      vim.current.window.cursor = cursor
    except vim.error:
      vim.current.window.cursor = (len(vim.current.buffer), 0)
    print(f'Reformatted in {time.time() - start:.4f}s.')

def grayUpgrade():
  _initialize_gray_env(upgrade=True)

def grayVersion():
  print(f'gray, version {gray.__version__} on Python {sys.version}.')

endpython3

command! gray :py3 gray()
command! grayUpgrade :py3 grayUpgrade()
command! grayVersion :py3 grayVersion()
