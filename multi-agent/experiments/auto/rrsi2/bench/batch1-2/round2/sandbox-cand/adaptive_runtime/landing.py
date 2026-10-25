"""落地层：把被接受的候选真正写进生产 harness，并保证可逐条撤销。

为什么非有不可
--------------
probe 的候选改动只打在**沙箱副本**上（``create_sandbox`` 复制一份干净的 multi-agent）。
于是原实现有一个致命缺口：即使某候选被判 ACCEPT、经验库也写了条目，
生产的 ``.env`` / ``prompt.py`` **一个字都没变** —— 用户下一次正常提问，
agent 行为与改进前完全一致。那不是自我改进，那是一次「只写报告的实验」。

真正的 RRSI 必须让被接受的 harness 成为下一轮的起点，所以这里提供两条落地通道：

- **L1/L3 参数** → 写进 ``.env`` 末尾的**机器托管块**。
  为什么不是覆盖用户那行：用户的 ``.env`` 是手工维护的，直接把 ``AGENT_MAX_ITERATIONS=15``
  改成 25 会在下次他手改时把改进冲掉，而且改动不可归属。托管块让「机器的改动」与
  「人的改动」在同一个文件里物理分离：每行自带 ``# landed by base/vN patch=HASH``，
  撤销 = 从块里删掉，用户的行永远不动。（python-dotenv 对同文件重复键取**后者**，
  块在文件末尾，因此托管值生效 —— 这条性质由单测钉住，不靠记忆。）
- **L2 提示** → 在生产 ``prompt.py`` 的锚点前插入补丁正文（与沙箱里同一操作）。
  锚点必须恰好出现一次，否则拒绝落地（不静默）。回滚 = 删除那段逐字相同的文本。

两道保险
--------
1. 绝不落地 Frozen Core 覆盖的 env / 符号：落地前先查，落地后立刻 ``frozen.verify()``，
   一旦越界**自动回滚并抛错**。「自己改判分口径」是最危险的自我修改，必须由代码而不是
   由自觉来禁止。
2. 每次落地写 ``experiments/landing/history.jsonl``，记录版本、patch_hash、原文与插入内容
   —— 回滚不依赖 git，因此不会碰用户其它未提交的改动。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from .io_util import append_jsonl, now_iso, read_jsonl
from .settings import PATHS, ROOT

BLOCK_START = "# >>> adaptive_runtime overlay (machine-managed, do not edit by hand) >>>"
BLOCK_END = "# <<< adaptive_runtime overlay <<<"
#: 块注释里允许出现的说明行前缀
_NOTE = "# 由 adaptive_runtime 自动维护：删除本块即完全回滚，你手工写的行永不被改动。"


def env_path(root: Path | None = None) -> Path:
    return (root or ROOT) / ".env"


# ---------------------------------------------------------------- 托管块解析

@dataclass
class OverlayEntry:
    key: str
    value: str
    version_tag: str = ""
    patch_hash: str = ""
    landed_at: str = ""
    kind: str = "env"
    extra: dict = field(default_factory=dict)

    def line(self) -> str:
        meta = " ".join(x for x in (f"landed by {self.version_tag}" if self.version_tag else "",
                                    f"patch={self.patch_hash}" if self.patch_hash else "") if x)
        return f"{self.key}={self.value}" + (f"   # {meta}" if meta else "")

    def to_dict(self) -> dict:
        return {"key": self.key, "value": self.value, "version_tag": self.version_tag,
                "patch_hash": self.patch_hash, "landed_at": self.landed_at,
                "kind": self.kind, **self.extra}


def _block_span(text: str) -> tuple[int, int] | None:
    lines = text.splitlines()
    start = end = -1
    for i, ln in enumerate(lines):
        if ln.strip() == BLOCK_START:
            start = i
        elif ln.strip() == BLOCK_END:
            end = i
    if start == -1 or end == -1 or end < start:
        return None
    return start, end


def read_block(root: Path | None = None) -> list[OverlayEntry]:
    """解析托管块 → 条目列表。块不存在返回空列表。"""
    p = env_path(root)
    if not p.exists():
        return []
    span = _block_span(p.read_text(encoding="utf-8", errors="replace"))
    if not span:
        return []
    out: list[OverlayEntry] = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()[span[0] + 1:span[1]]:
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        v = v.strip()
        meta = ""
        if "#" in v:
            v, meta = v.split("#", 1)
            v, meta = v.strip(), meta.strip()
        tag = ""
        if "landed by" in meta:
            tag = meta.split("landed by", 1)[1].split("patch=")[0].strip()
        ph = meta.split("patch=")[1].strip() if "patch=" in meta else ""
        out.append(OverlayEntry(key=k.strip(), value=v, version_tag=tag, patch_hash=ph))
    return out


def render_block(entries: list[OverlayEntry]) -> str:
    body = "\n".join(e.line() for e in entries)
    return f"{BLOCK_START}\n{_NOTE}\n{body}\n{BLOCK_END}".rstrip() + "\n"


def _replace_block(text: str, block: str) -> str:
    """只重写托管块区域，块外的任何行（用户手工区）逐字保留。"""
    block = block.rstrip("\n")
    span = _block_span(text)
    if span:
        lines = text.splitlines()
        new = lines[:span[0]] + block.splitlines() + lines[span[1] + 1:]
        return "\n".join(new) + "\n"
    tail = text.rstrip("\n")
    return (tail + "\n\n" + block + "\n") if tail else (block + "\n")


def _write_env(entries: list[OverlayEntry], root: Path | None = None) -> Path:
    p = env_path(root)
    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    p.write_text(_replace_block(text, render_block(entries)), encoding="utf-8")
    return p


def _sync_process_env(key: str, value: str | None, root: Path | None) -> None:
    """让**当前控制平面进程**的环境与新落地值一致（仅限落地到生产目录时）。

    为什么必须做：probe 的子进程环境是 ``os.environ.copy()``，而 ``load_dotenv(override=False)``
    对**已存在**的环境变量不重写。若不刷新父进程环境，当用户 shell 里恰好 export 过这个键时，
    刚落地的值会被旧环境变量遮住 —— 落地在代码层生效、在行为层没生效，而闭环看到的
    恰好是「ΔQ≈0」，于是把一条真正有效的改动记成「无效」。
    """
    if root is not None:
        return                                  # 测试用的临时 root 不污染本进程
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(value)


def upsert_entry(key: str, value: str, *, version_tag: str = "", patch_hash: str = "",
                 root: Path | None = None) -> dict:
    """在托管块内幂等地写入 ``key=value``（块内同键只保留一条，且总在用户行之后）。"""
    entries = read_block(root)
    hit = next((e for e in entries if e.key == key), None)
    if hit:
        hit.value, hit.version_tag, hit.patch_hash = str(value), version_tag, patch_hash
    else:
        entries.append(OverlayEntry(key=key, value=str(value), version_tag=version_tag,
                                    patch_hash=patch_hash, landed_at=now_iso()))
    _write_env(entries, root)
    _sync_process_env(key, str(value), root)
    return {"action": "upsert", "key": key, "value": str(value),
            "version_tag": version_tag, "patch_hash": patch_hash}


def drop_entries(keys: list[str], root: Path | None = None) -> list[str]:
    entries = read_block(root)
    keep = [e for e in entries if e.key not in set(keys)]
    removed = [e.key for e in entries if e.key in set(keys)]
    if removed:
        _write_env(keep, root)
        # 回滚后退回「用户区那个值」：文件已重写，此处 parse 拿到的就是用户行
        from .frozen import parse_env_file
        user_vals = parse_env_file(env_path(root))
        for k in removed:
            _sync_process_env(k, user_vals.get(k), root)
    return removed


def live_value(key: str, root: Path | None = None) -> str | None:
    """当前生效值：进程环境 > 托管块 > .env 用户区。"""
    v = os.environ.get(key)
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    hit = next((e for e in read_block(root) if e.key == key), None)
    if hit:
        return hit.value
    from .frozen import parse_env_file
    return parse_env_file(env_path(root)).get(key)


# ---------------------------------------------------------------- L2 提示落地

def land_prompt(file: str, anchor: str, insert: str, *, version_tag: str = "",
                patch_hash: str = "", root: Path | None = None) -> dict:
    """在生产文件的锚点前插入正文。锚点不唯一 → 直接拒绝，不静默。"""
    r = root or ROOT
    p = r / file
    if not p.exists():
        raise RuntimeError(f"落地目标文件不存在：{p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    n = text.count(anchor)
    if n != 1:
        raise RuntimeError(f"锚点在 {file} 出现 {n} 次（必须恰好 1 次），拒绝落地：{anchor!r}")
    p.write_text(text.replace(anchor, insert + anchor, 1), encoding="utf-8")
    append_jsonl(PATHS.landing_history, [{"ts": now_iso(), "kind": "prompt", "file": file,
                                          "version_tag": version_tag, "patch_hash": patch_hash,
                                          "anchor": anchor, "insert_chars": len(insert)}])
    return {"action": "land_prompt", "file": file, "version_tag": version_tag,
            "patch_hash": patch_hash, "chars": len(insert)}


def revert_prompt(file: str, insert: str, root: Path | None = None) -> bool:
    """逐字删除已插入的正文。返回是否真的删掉了。"""
    p = (root or ROOT) / file
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8", errors="replace")
    if text.count(insert) < 1:
        return False
    p.write_text(text.replace(insert, "", 1), encoding="utf-8")
    return True


# ---------------------------------------------------------------- 候选级 API

def land(cand, *, version_tag: str = "", root: Path | None = None,
         allow_keys: tuple[str, ...] = ()) -> dict:
    """把一个被接受的候选写进生产。Frozen Core 覆盖的键一律拒绝。"""
    from . import frozen

    patch_hash = getattr(cand, "patch_hash", "")
    if getattr(cand, "is_env", False):
        key = cand.target
        if key in frozen.FROZEN_ENV and key not in allow_keys:
            raise RuntimeError(f"拒绝落地冻结 env：{key}")
        res = upsert_entry(key, cand.new, version_tag=version_tag, patch_hash=patch_hash,
                           root=root)
    elif getattr(cand, "is_prompt", False):
        patch = cand.patch or {}
        res = land_prompt(str(patch.get("file") or ""), str(patch.get("anchor") or ""),
                          str(patch.get("insert") or ""), version_tag=version_tag,
                          patch_hash=patch_hash, root=root)
    else:
        raise RuntimeError(f"无法落地的候选类型：patch.kind={getattr(cand, 'patch', {}).get('kind')}")
    res["target"] = getattr(cand, "target", "")
    res["level"] = getattr(cand, "level", 0)
    append_jsonl(PATHS.landing_history, [dict(res, ts=now_iso(),
                                              candidate_id=getattr(cand, "candidate_id", ""))])
    return res


def check_integrity(root: Path | None = None) -> list[str]:
    """落地后的自检：冻结符号/env 是否仍完整 + L2 锚点是否仍唯一。"""
    from . import frozen, proposer
    out = [f"{v.kind}:{v.target}" for v in frozen.verify(root=root)]
    if root is None:
        out += [f"patch_anchor:{x}" for x in proposer._check_prompt_patches()]
    return out


def rollback_to(version_tag: str, root: Path | None = None) -> dict:
    """撤销所有比 ``version_tag`` 新的落地改动（env 块条目 + 已落地提示补丁）。"""
    hist = read_jsonl(PATHS.landing_history)
    entries = read_block(root)
    keep = [e for e in entries if _le(e.version_tag, version_tag)]
    drop_keys = [e.key for e in entries if not _le(e.version_tag, version_tag)]
    removed = drop_entries(drop_keys, root)
    reverted_prompts: list[str] = []
    for h in reversed(hist):
        if h.get("kind") != "prompt":
            continue
        if _le(str(h.get("version_tag") or ""), version_tag):
            break
        f = str(h.get("file") or "")
        # history 不存正文（避免大字段），改用 patch_hash 对应的模板文本回滚
        tmpl = _patch_text_by_hash(h.get("patch_hash"))
        if tmpl and revert_prompt(f, tmpl, root):
            reverted_prompts.append(f)
    append_jsonl(PATHS.landing_history, [{"ts": now_iso(), "kind": "rollback",
                                          "to_version": version_tag,
                                          "removed_env": removed,
                                          "reverted_prompt_files": reverted_prompts}])
    return {"removed_env": removed, "reverted_prompt_files": reverted_prompts,
            "kept_env": [e.key for e in keep]}


def _patch_text_by_hash(patch_hash: str) -> str | None:
    """由 patch_hash 反查提示补丁正文。

    必须与 ``Candidate.diff_text()`` 的格式逐字一致 —— 否则 hash 对不上，
    回滚会「静默找不到正文」，于是生产里留着一段没人能拆的补丁。
    """
    from .io_util import sha256_text
    from .proposer import PROMPT_PATCHES
    for name, spec in PROMPT_PATCHES.items():
        fname = name.split("::", 1)[0]
        head = f"{fname} @ {spec.get('anchor')!r}\n"
        if sha256_text(head + str(spec.get("insert") or ""))[:16] == patch_hash:
            return str(spec.get("insert") or "")
    return None


def _le(a: str, b: str) -> bool:
    """a 是否不晚于 b（按 ``base/vN`` 里的版本号比较）。

    解不出版本号的当作「比 b 新」，于是它会被撤销。**这是故意的**：
    回滚是撤销动作，漏撤销 = 被否决的候选继续在生产里起作用；
    过撤销 = 机器托管块里多删几行（用户手工区的行永远不会被动，而且
    ``history.jsonl`` 记着每一条，重放即可恢复）。两个方向不可逆程度不同，选可恢复那个。
    """
    def num(t: str) -> int | None:
        s = str(t).strip()
        if "v" not in s:
            return None
        tail = s.rsplit("v", 1)[1]
        return int(tail) if tail.isdigit() else None
    na, nb = num(a), num(b)
    if na is None or nb is None:
        return False
    return na <= nb


def content_stamp(root: Path | None = None) -> str:
    """落地产物的内容指纹。

    为什么需要：``BaseVersion.base_commit`` 靠 git HEAD，而自动落地**不提交**
    （提交必须人工授权）。若仍然只记 commit，base/v1 与 base/v2 的 commit 会相同，
    「改进对着哪份代码比的」就失去归属。用内容 hash 补上这个缺口。
    """
    r = root or ROOT
    parts = [render_block(read_block(r))]
    for f in ("prompt.py", "supervisor.py", "agents.py"):
        p = r / f
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()[:16]


def status(root: Path | None = None) -> dict:
    return {"env_block_entries": [e.to_dict() for e in read_block(root)],
            "landing_events": len(read_jsonl(PATHS.landing_history)),
            "content_stamp": content_stamp(root),
            "integrity": check_integrity(root)}
