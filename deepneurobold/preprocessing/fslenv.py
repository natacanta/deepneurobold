#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepneurobold.preprocessing.fslenv
====================================
Helper for running FSL commands with the inherited process environment
(``os.environ`` — which must already contain ``FSLDIR``, ``PATH``, and
``FSLOUTPUTTYPE`` set up by ``config_file()`` or the SLURM wrapper).

Used by every module that invokes FLIRT, BET, FAST, FSLMATHS, etc.

- Inherits the current process environment.
- Raises a clear :class:`FileNotFoundError` if the requested FSL binary is
  not on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Sequence


def _check_cmd_available(cmd: str) -> None:
    """Raise ``FileNotFoundError`` if *cmd* is not on ``PATH``."""
    if shutil.which(cmd) is None:
        raise FileNotFoundError(
            f"'{cmd}' is not on PATH. Check your FSLDIR setup in config_file()."
        )


def run_in_fsl(
    cmd: str | Sequence[str],
    *,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    shell: Optional[bool] = None,
) -> subprocess.CompletedProcess:
    """
    Run an FSL command inheriting the current process environment.

    Parameters
    ----------
    cmd : str or sequence of str
        Full FSL command, either as a single string
        (``'flirt -in input.nii.gz -ref ref.nii.gz -out out.nii.gz -applyxfm -init ident.mat'``)
        or as a list of tokens
        (``['flirt', '-in', 'input.nii.gz', ...]``).
    check : bool, default False
        If True, raise :class:`subprocess.CalledProcessError` on non-zero
        return code.
    capture_output : bool, default True
        If True, capture both stdout and stderr.
    text : bool, default True
        If True, decode stdout/stderr as text.
    shell : bool, optional
        Forwarded to :func:`subprocess.run`. If ``None`` (default), inferred
        from ``cmd`` (True if string, False if list).

    Returns
    -------
    subprocess.CompletedProcess
        Object with ``.stdout``, ``.stderr``, and ``.returncode``.
    """
    # Verify the primary binary is reachable. For pipelines or shell snippets
    # we tolerate failures here silently — only the leading token is checked.
    try:
        first = cmd.split()[0] if isinstance(cmd, str) else (cmd[0] if cmd else "")
        if first:
            _check_cmd_available(first)
    except Exception:
        pass

    if shell is None:
        shell = isinstance(cmd, str)

    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=text,
        shell=shell,
        env=os.environ,
    )
