"""
获取指定目录下 Git 仓库当前分支与 commit，用于评估/训练日志复现。
不再调用 subprocess，而是通过直接读取 .git 目录下的 HEAD 和 refs/packed-refs 获取。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _find_git_dir(start_path: str) -> Optional[str]:
    """
    从 start_path 向上查找 .git 目录。
    支持 .git 为目录，或 .git 为文件（如 worktree 或 submodule）。
    """
    curr = os.path.abspath(start_path)
    while True:
        git_path = os.path.join(curr, ".git")
        if os.path.exists(git_path):
            if os.path.isdir(git_path):
                return git_path
            elif os.path.isfile(git_path):
                # 处理 gitdir: 情况（worktree 或 submodule）
                try:
                    with open(git_path, "r", encoding="utf-8") as f:
                        line = f.readline().strip()
                        if line.startswith("gitdir:"):
                            rel_gitdir = line[len("gitdir:") :].strip()
                            # gitdir 可能是绝对路径，也可能是相对路径
                            if os.path.isabs(rel_gitdir):
                                return rel_gitdir
                            else:
                                return os.path.abspath(os.path.join(curr, rel_gitdir))
                except Exception:
                    pass
        
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return None


def _resolve_ref(git_dir: str, ref_name: str, depth: int = 0) -> Optional[str]:
    """
    从 git_dir 解析指定的 ref（如 'refs/heads/master'），返回 40 位 SHA。
    depth 用于防止无限递归。
    """
    if depth > 10:
        return None

    # 1. 尝试读取松散文件 (loose ref)
    loose_path = os.path.join(git_dir, ref_name)
    if os.path.isfile(loose_path):
        try:
            with open(loose_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith("ref:"):
                    return _resolve_ref(git_dir, content[4:].strip(), depth + 1)
                if len(content) >= 40:
                    return content[:40]
        except Exception:
            pass

    # 2. 尝试读取 packed-refs
    packed_path = os.path.join(git_dir, "packed-refs")
    if os.path.isfile(packed_path):
        try:
            with open(packed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == ref_name:
                        return parts[0]
        except Exception:
            pass
            
    return None


def get_git_repo_info(cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    在 cwd 下查找 Git 仓库，并直接读取文件获取当前分支与 commit SHA。
    
    Args:
        cwd: Git 工作区内的任意目录；为 None 时使用当前工作目录。

    Returns:
        dict，包含 branch, commit, commit_short, cwd；失败时对应字符串字段为 None。
    """
    resolved = os.path.abspath(cwd if cwd else os.getcwd())
    out: Dict[str, Any] = {
        "branch": None,
        "commit": None,
        "commit_short": None,
        "cwd": resolved,
    }

    git_dir = _find_git_dir(resolved)
    if not git_dir:
        return out

    head_path = os.path.join(git_dir, "HEAD")
    if not os.path.isfile(head_path):
        return out

    try:
        with open(head_path, "r", encoding="utf-8") as f:
            head_content = f.read().strip()

        if head_content.startswith("ref:"):
            # 在一个分支上
            ref_name = head_content[4:].strip()
            out["branch"] = ref_name.replace("refs/heads/", "")
            out["commit"] = _resolve_ref(git_dir, ref_name)
        else:
            # Detached HEAD (head_content 本身就是 SHA)
            if len(head_content) >= 40:
                out["branch"] = "HEAD (detached)"
                out["commit"] = head_content[:40]
            
        if out["commit"]:
            out["commit_short"] = out["commit"][:7]

    except Exception:
        pass

    return out


if __name__ == "__main__":
    # 简单测试
    import json
    info = get_git_repo_info()
    print(json.dumps(info, indent=4, ensure_ascii=False))
