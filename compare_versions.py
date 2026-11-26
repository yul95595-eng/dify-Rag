# -*- coding: utf-8 -*-
"""多版本对比 v1.0 / v2.0 / v2.1 / v3.1 / v3.2 评测报告。

用法：python compare_versions.py
自动在 e:\\dify-Rag\\eval_output 下发现各类报告并取最新一份。
"""
import glob
import json
import os
import re

EVAL_DIR = r"e:\dify-Rag\eval_output"

# (标签, 文件名匹配模式)
SPECS = [
    ("v1.0", "eval_full_report_*.json"),
    ("v2.0", "v2_regression_report_*.json"),
    ("v2.1", "v21_regression_report_*.json"),
    ("v3.1", "v31_regression_report_*.json"),
    ("v3.2", "v32_regression_report_*.json"),
]


def latest(pattern):
    hits = sorted(glob.glob(os.path.join(EVAL_DIR, pattern)))
    return hits[-1] if hits else None


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(n, d):
    return "%.2f%%" % (100.0 * n / d) if d else "-"


def main():
    reports = []
    for label, pat in SPECS:
        p = latest(pat)
        if not p:
            print("[WARN] 未找到 %s 报告（模式 %s），跳过" % (label, pat))
            continue
        reports.append((label, os.path.basename(p), load(p)))

    if not reports:
        print("没有任何报告可对比")
        return

    print("=" * 78)
    print("报告文件")
    for label, name, _ in reports:
        print("  %-6s %s" % (label, name))

    # ---------------------------------------------------------- metrics 全字段对比
    all_keys = []
    for _, _, d in reports:
        for k in (d.get("metrics") or {}):
            if k not in all_keys:
                all_keys.append(k)

    print("=" * 78)
    print("metrics 全字段对比")
    header = "  %-34s" % "字段"
    for label, _, _ in reports:
        header += "%16s" % label
    print(header)
    for k in all_keys:
        row = "  %-34s" % k[:34]
        for _, _, d in reports:
            v = (d.get("metrics") or {}).get(k)
            if isinstance(v, float):
                s = "%.4f" % v
            elif isinstance(v, list):
                s = "list(%d)" % len(v)
            elif isinstance(v, dict):
                s = "dict(%d)" % len(v)
            else:
                s = str(v)
            row += "%16s" % s[:16]
        print(row)

    # ---------------------------------------------------------- 派生统计
    print("=" * 78)
    print("派生统计")
    stats = []
    for label, _, d in reports:
        recs = d.get("records") or []
        n = len(recs)
        wrong = [r for r in recs if not r.get("category_correct")]
        api_fail = [r for r in recs if not r.get("api_ok")]
        parse_fail = [r for r in recs if not r.get("parse_ok")]
        fb = [r for r in recs if r.get("is_fallback")]
        human = [r for r in recs if r.get("need_human")]
        kwh = sum(int(r.get("kw_hit") or 0) for r in recs)
        kwt = sum(int(r.get("kw_total") or 0) for r in recs)
        allhit = [r for r in recs if r.get("kw_total") and r.get("kw_hit") == r.get("kw_total")]
        el = [float(r.get("elapsed") or 0) for r in recs]
        # 来源冗余：形如 X#X（文档名与章节名重复拼接）
        dup_src = [r for r in recs
                   if re.match(r"^(.+)#\1$", str(r.get("source") or "").strip())]
        empty_src = [r for r in recs
                     if str(r.get("source") or "").strip() in ("", "none")]
        stats.append(dict(label=label, n=n, wrong=wrong, api_fail=api_fail,
                          parse_fail=parse_fail, fb=fb, human=human,
                          kwh=kwh, kwt=kwt, allhit=allhit,
                          avg=sum(el) / len(el) if el else 0,
                          mx=max(el) if el else 0,
                          dup_src=dup_src, empty_src=empty_src))

    rows = [
        ("题量", lambda s: str(s["n"])),
        ("分类正确", lambda s: "%d/%d" % (s["n"] - len(s["wrong"]), s["n"])),
        ("分类准确率", lambda s: pct(s["n"] - len(s["wrong"]), s["n"])),
        ("API 失败", lambda s: str(len(s["api_fail"]))),
        ("解析失败", lambda s: str(len(s["parse_fail"]))),
        ("兜底条数", lambda s: "%d (%s)" % (len(s["fb"]), pct(len(s["fb"]), s["n"]))),
        ("need_human", lambda s: str(len(s["human"]))),
        ("关键词命中率", lambda s: pct(s["kwh"], s["kwt"])),
        ("整题全命中", lambda s: pct(len(s["allhit"]), s["n"])),
        ("来源重复拼接(X#X)", lambda s: str(len(s["dup_src"]))),
        ("来源为空/none", lambda s: str(len(s["empty_src"]))),
        ("平均时延(s)", lambda s: "%.2f" % s["avg"]),
        ("最大时延(s)", lambda s: "%.2f" % s["mx"]),
    ]
    header = "  %-22s" % "指标"
    for s in stats:
        header += "%18s" % s["label"]
    print(header)
    for name, fn in rows:
        line = "  %-22s" % name
        for s in stats:
            line += "%18s" % fn(s)[:18]
        print(line)

    # ---------------------------------------------------------- 错题清单
    print("=" * 78)
    print("错题清单")
    for s in stats:
        if not s["wrong"]:
            print("  %-6s 无错题" % s["label"])
            continue
        for r in s["wrong"]:
            print("  %-6s %s「%s」期望 %s → 实际 %s" % (
                s["label"], r.get("id"), r.get("question"),
                r.get("expected_category"), r.get("predicted_category") or "(空)"))

    # ---------------------------------------------------------- 版本间差异（逐题）
    if len(reports) >= 2:
        print("=" * 78)
        print("逐题变化（相对首份报告）")
        # 按 records 建索引
        idx = []
        for label, _, d in reports:
            idx.append((label, {r.get("id"): r for r in (d.get("records") or [])}))
        b_label, b_map = idx[0]
        for label, m in idx[1:]:
            fixed, broken = [], []
            for qid, r in m.items():
                b = b_map.get(qid)
                if not b:
                    continue
                if r.get("category_correct") and not b.get("category_correct"):
                    fixed.append(qid)
                if not r.get("category_correct") and b.get("category_correct"):
                    broken.append(qid)
            print("  %s vs %s：修复 %d %s | 回退 %d %s" % (
                label, b_label, len(fixed), fixed, len(broken), broken))

    print("=" * 78)


if __name__ == "__main__":
    main()
