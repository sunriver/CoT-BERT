"""
获取指定目录下 Git 仓库当前分支与 commit，用于评估/训练日志复现。
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional


def get_git_repo_info(cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    在 cwd 下执行 git，获取当前分支与完整/短 commit hash。

    Args:
        cwd: Git 工作区内的任意目录；为 None 时使用当前工作目录。

    Returns:
        dict，包含 branch, commit, commit_short, cwd；失败时对应字符串字段为 None。
    """
    resolved = cwd if cwd else os.getcwd()
    out: Dict[str, Any] = {
        "branch": None,
        "commit": None,
        "commit_short": None,
        "cwd": resolved,
    }
    if not resolved or not os.path.isdir(resolved):
        return out

    def _run_git(args_list):
        try:
            p = subprocess.run(
                ["git"] + args_list,
                cwd=resolved,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if p.returncode != 0:
                return None
            s = (p.stdout or "").strip()
            return s if s else None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    out["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    out["commit"] = _run_git(["rev-parse", "HEAD"])
    out["commit_short"] = _run_git(["rev-parse", "--short", "HEAD"])
    return out
