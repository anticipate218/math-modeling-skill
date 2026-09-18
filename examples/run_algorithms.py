#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""examples/algorithms/ 统一自测入口。

用途
----
跑遍 ``examples/algorithms/`` 下每个模块的 ``_self_test()``，把结果与
``examples/algorithms_golden.json`` 里记录的"黄金值"逐键比对，并验证自测是**确定性的**
（同一进程内跑两次结果必须完全一致）。

为什么要有"黄金值"文件
----------------------
自测函数能跑通、不报异常，**并不代表结果是对的**。比如一个把成本型指标算反的 TOPSIS 实现
照样能返回一组漂亮的 [0,1] 之间的贴近度。把某次人工核验过的输出固化下来，之后任何改动
只要改变了数值就会被立刻发现——这是防止"悄悄改坏"的最低成本手段。

黄金值不是"真理"，而是**已人工核验过的基线**。修改算法后数值发生变化时：
先在 ``--verbose`` 下看清差异，确认新数值确实更正确（最好用独立的对照实现，例如 scipy），
再运行 ``--update-golden`` 更新基线。不要为了"让 CI 绿"而直接更新黄金值。

用法
----
    python examples/run_algorithms.py                  # 全部模块
    python examples/run_algorithms.py --verbose        # 打印每个键的实际值
    python examples/run_algorithms.py --module graphs  # 只跑一个模块
    python examples/run_algorithms.py --list           # 只列出模块与键数
    python examples/run_algorithms.py --update-golden  # 重新生成黄金值文件（谨慎）

退出码
------
0 全部通过；1 有比对失败、缺失模块或非确定性。

依赖
----
numpy（算法模块本身需要）+ 标准库。本脚本不依赖 scipy / sklearn。
"""

import argparse
import importlib
import json
import math
import os
import pkgutil
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "algorithms_golden.json")

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import algorithms  # noqa: E402

# 包内不作为独立模型模块列出的文件
SKIP_MODULES = {"_common"}


def discover_modules() -> List[str]:
    """返回 ``examples/algorithms/`` 下所有应参与自测的模块名（排序后）。

    以 ``_`` 开头的模块（``__init__``、``_common`` 等）是共享基础设施，本身不建模，
    不参与自测。
    """
    names = []
    for info in pkgutil.iter_modules(algorithms.__path__):
        if info.name.startswith("_") or info.name in SKIP_MODULES:
            continue
        names.append(info.name)
    return sorted(names)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(path: str, got: Any, want: Any, rtol: float, atol: float,
            out: List[str]) -> None:
    """递归比对 got 与 want，把差异描述追加到 out。"""
    if isinstance(want, bool) or isinstance(got, bool):
        if not (isinstance(got, bool) and isinstance(want, bool) and got == want):
            out.append(f"{path}: 实际 {got!r} != 期望 {want!r}")
        return

    if want is None or got is None:
        if got is not want:
            out.append(f"{path}: 实际 {got!r} != 期望 {want!r}")
        return

    if isinstance(want, dict):
        if not isinstance(got, dict):
            out.append(f"{path}: 期望 dict，实际 {type(got).__name__}")
            return
        for k in sorted(set(want) | set(got)):
            if k not in got:
                out.append(f"{path}.{k}: 结果里缺少这个键")
            elif k not in want:
                out.append(f"{path}.{k}: 黄金值里没有这个键（新增键？）")
            else:
                compare(f"{path}.{k}", got[k], want[k], rtol, atol, out)
        return

    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)):
            out.append(f"{path}: 期望序列，实际 {type(got).__name__}")
            return
        if len(got) != len(want):
            out.append(f"{path}: 长度 {len(got)} != 期望 {len(want)}")
            return
        for i, (g, w) in enumerate(zip(got, want)):
            compare(f"{path}[{i}]", g, w, rtol, atol, out)
        return

    if _is_number(want) and _is_number(got):
        if math.isnan(float(want)) or math.isnan(float(got)):
            if not (math.isnan(float(want)) and math.isnan(float(got))):
                out.append(f"{path}: 实际 {got} != 期望 {want}")
            return
        if not math.isclose(float(got), float(want), rel_tol=rtol, abs_tol=atol):
            out.append(f"{path}: 实际 {got!r} != 期望 {want!r}")
        return

    if got != want:
        out.append(f"{path}: 实际 {got!r} != 期望 {want!r}")


def run_module(name: str, rtol: float, atol: float, verbose: bool
               ) -> Tuple[bool, List[str], Dict[str, Any], int]:
    """跑一个模块的自测并比对。返回 (是否通过, 差异列表, 实际结果, 键数)。"""
    problems: List[str] = []
    module = importlib.import_module("algorithms." + name)

    if not hasattr(module, "_self_test"):
        return False, [f"{name}: 模块没有 _self_test()"], {}, 0

    first = module._self_test()
    second = module._self_test()

    # 确定性：同一进程内两次调用必须完全一致
    det: List[str] = []
    compare("determinism", second, first, 0.0, 0.0, det)
    problems.extend(det)

    golden = GOLDEN.get(name)
    if golden is None:
        problems.append(f"{name}: algorithms_golden.json 里没有它的基线，"
                        f"请人工核验后运行 --update-golden")
    else:
        compare(name, first, golden, rtol, atol, problems)

    n_keys = len(first) if isinstance(first, dict) else 0
    if verbose:
        print(f"  {name} 实际输出：")
        print("    " + json.dumps(first, ensure_ascii=False, default=str,
                                  indent=2).replace("\n", "\n    "))
    return (not problems), problems, first, n_keys


def main(argv: List[str]) -> int:
    global GOLDEN

    parser = argparse.ArgumentParser(
        description="跑遍 examples/algorithms/ 的自测并与黄金值比对")
    parser.add_argument("--module", action="append", default=None,
                        help="只跑指定模块（可重复）")
    parser.add_argument("--rtol", type=float, default=1e-9,
                        help="相对容差，默认 1e-9")
    parser.add_argument("--atol", type=float, default=1e-9,
                        help="绝对容差，默认 1e-9")
    parser.add_argument("--verbose", action="store_true", help="打印每个键的实际值")
    parser.add_argument("--list", action="store_true", help="只列出模块与键数")
    parser.add_argument("--update-golden", action="store_true",
                        help="用当前结果重写 algorithms_golden.json（改算法后谨慎使用）")
    args = parser.parse_args(argv)

    if os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
            GOLDEN = json.load(fh)
    else:
        GOLDEN = {}

    available = discover_modules()
    selected = args.module if args.module else available

    unknown = [m for m in selected if m not in available]
    if unknown:
        print(f"未知模块：{', '.join(unknown)}")
        print(f"可用模块：{', '.join(available)}")
        return 1

    if args.list:
        for name in selected:
            g = GOLDEN.get(name)
            n = len(g) if isinstance(g, dict) else 0
            print(f"{name:<14} 基线键数 {n}")
        return 0

    print(f"算法自测：{len(selected)} 个模块，容差 rtol={args.rtol:g} atol={args.atol:g}")
    print("=" * 68)

    if args.update_golden:
        fresh: Dict[str, Any] = {}
        for name in available:
            module = importlib.import_module("algorithms." + name)
            fresh[name] = module._self_test()
        with open(GOLDEN_PATH, "w", encoding="utf-8") as fh:
            json.dump(fresh, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"已重写 {os.path.relpath(GOLDEN_PATH, os.path.dirname(HERE))}")
        print("注意：请确认这次 diff 是有意为之，而不是为了掩盖回归。")
        return 0

    failed_modules = 0
    total_keys = 0
    for name in selected:
        ok, problems, _result, n_keys = run_module(name, args.rtol, args.atol,
                                                   args.verbose)
        total_keys += n_keys
        if ok:
            print(f"PASS  {name:<14} {n_keys:>3} 个键")
        else:
            failed_modules += 1
            print(f"FAIL  {name:<14} {n_keys:>3} 个键")
            for p in problems:
                print(f"        - {p}")

    # 有基线但没被本次选中的模块也要提醒
    if not args.module:
        extra = sorted(set(GOLDEN) - set(available))
        if extra:
            print()
            print(f"警告：黄金值里有已不存在的模块：{', '.join(extra)}")

    print("=" * 68)
    print(f"合计 {len(selected)} 个模块 / {total_keys} 个断言键，失败 {failed_modules} 个模块")

    if failed_modules:
        print("自测失败。请修复实现，或在人工核验新数值确实更正确后更新基线。")
        return 1
    print("自测全部通过。")
    return 0


GOLDEN: Dict[str, Any] = {}

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
