#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业智能问答与工单分类助手 —— Dify 批量测试与评估脚本

功能：
  1. 读取 test_questions.json
  2. 逐条调用 Dify Chat API（/v1/chat-messages）
  3. 打印每个问题的分类与回答
  4. 统计分类准确率（含每类别 P/R/F1 与混淆矩阵）
  5. 统计回答中包含「暂未找到明确制度，建议转人工处理」的比例（兜底率）
  6. 统计关键词命中率、need_human 触发率、平均时延、失败率
  7. 导出评估结果（JSON + CSV）

依赖：仅使用 Python 标准库（无需 pip install）。Python >= 3.8

用法示例：
  # 1) 先本地校验测试集（不调用 API，不需要密钥）
  python run_assistant.py --validate

  # 2) 离线演练（不调用 API，用占位回答跑通评估流程）
  python run_assistant.py --offline

  # 3) 正式批量测试
  python run_assistant.py --api-key app-xxxxxxxxxxxxxxxx --limit 10

  # 4) 私有化部署（例如本地/内网 Dify）
  python run_assistant.py --api-url http://192.168.x.x/v1/chat-messages --api-key app-xxxx

  # 5) 只跑某个类别
  python run_assistant.py --category IT

安全提示：不要把真实 API Key 硬编码进代码或提交到版本库。
推荐通过环境变量传入：  set DIFY_API_KEY=app-xxxx   （PowerShell: $env:DIFY_API_KEY="app-xxxx"）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------

#: Dify Chat API 默认地址（云端 SaaS）。私有化部署请用 --api-url 覆盖。
API_URL_DEFAULT = "https://api.dify.ai/v1/chat-messages"

#: API Key 占位符，请勿硬编码真实密钥；运行时用 --api-key 或环境变量 DIFY_API_KEY 传入。
API_KEY = "app-你的Dify应用API密钥"

#: 兜底话术，必须与 dify_prompts.json 中 fallback_sentence 完全一致
FALLBACK_TEXT = "暂未找到明确制度，建议转人工处理。"

#: 合法类别
VALID_CATEGORIES = ("人事", "行政", "财务", "IT", "后勤", "其他")

#: 最终输出必须包含的字段
REQUIRED_OUTPUT_FIELDS = ("question", "category", "answer", "suggestion", "source", "need_human")

#: 默认测试集文件（与本脚本同目录）
DEFAULT_QUESTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_questions.json")

DEFAULT_TIMEOUT = 60
DEFAULT_RETRY = 2
DEFAULT_SLEEP = 0.5


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _setup_console() -> None:
    """尽量让 Windows 控制台正常输出中文。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def load_questions(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """读取测试问题集，返回 (问题列表, meta)。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到测试问题集文件：{path}")
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, list):
        questions, meta = data, {}
    elif isinstance(data, dict):
        questions = data.get("questions") or data.get("data") or []
        meta = data.get("meta", {})
    else:
        raise ValueError("test_questions.json 结构不合法：应为数组或含 questions 字段的对象")

    cleaned: List[Dict[str, Any]] = []
    for idx, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        if not q:
            continue
        cleaned.append(
            {
                "id": item.get("id", f"Q{idx:03d}"),
                "question": q,
                "expected_category": (item.get("expected_category") or "").strip(),
                "expected_keywords": item.get("expected_keywords") or [],
                "topic": item.get("topic", ""),
            }
        )
    if not cleaned:
        raise ValueError("测试问题集为空")
    return cleaned, meta


def validate_questions(questions: List[Dict[str, Any]]) -> List[str]:
    """校验测试集合法性，返回问题描述列表（空列表表示通过）。"""
    problems: List[str] = []
    for item in questions:
        qid = item["id"]
        if not item["question"]:
            problems.append(f"{qid}: question 为空")
        if item["expected_category"] not in VALID_CATEGORIES:
            problems.append(
                f"{qid}: expected_category 非法（{item['expected_category']!r}），"
                f"必须是 {VALID_CATEGORIES} 之一"
            )
        kws = item["expected_keywords"]
        if not isinstance(kws, list) or not kws:
            problems.append(f"{qid}: expected_keywords 必须是非空数组")
        elif any(not isinstance(k, str) or not k.strip() for k in kws):
            problems.append(f"{qid}: expected_keywords 含非字符串或空字符串元素")
    dup = [q for q, c in Counter(i["question"] for i in questions).items() if c > 1]
    for d in dup:
        problems.append(f"重复问题：{d}")
    return problems


# ---------------------------------------------------------------------------
# Dify API 调用
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    """去掉模型可能输出的 ```json ... ``` 包裹。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_structured_answer(raw_answer: str) -> Dict[str, Any]:
    """
    解析 Dify 返回的 answer 文本，尽量抽取 question/category/answer/suggestion/source/need_human。
    兼容三种情况：纯 JSON、JSON 被代码块包裹、纯自然语言。
    """
    result: Dict[str, Any] = {
        "question": "",
        "category": "",
        "answer": "",
        "suggestion": "",
        "source": "none",
        "need_human": False,
        "parse_ok": False,
    }
    text = _strip_code_fence(raw_answer)
    if not text:
        return result

    obj: Optional[Dict[str, Any]] = None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            obj = loaded
    except json.JSONDecodeError:
        # 尝试从文本中截取第一个 JSON 对象
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                loaded = re_json = json.loads(match.group(0))
                if isinstance(loaded, dict):
                    obj = re_json
            except json.JSONDecodeError:
                obj = None

    if obj is not None:
        result.update(
            {
                "question": str(obj.get("question", "")).strip(),
                "category": str(obj.get("category", "")).strip(),
                "answer": str(obj.get("answer", "")).strip(),
                "suggestion": str(obj.get("suggestion", "")).strip(),
                "source": str(obj.get("source", "none")).strip() or "none",
                "need_human": _to_bool(obj.get("need_human", False)),
                "parse_ok": True,
            }
        )
    else:
        # 非结构化：整体当作 answer，并尝试从文本中猜类别
        result["answer"] = text
        guessed = guess_category_from_text(text)
        if guessed:
            result["category"] = guessed

    if not result["answer"]:
        result["answer"] = text
    return result


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是"}
    return False


def guess_category_from_text(text: str) -> str:
    """从自由文本中提取类别标签（用于兼容非 JSON 输出）。"""
    for cat in VALID_CATEGORIES:
        if re.search(rf"(category|类别|分类)\D{{0,6}}{re.escape(cat)}", text):
            return cat
    hits = [cat for cat in VALID_CATEGORIES if cat in text]
    return hits[0] if len(hits) == 1 else ""


def normalize_category(value: str) -> str:
    """归一化类别文本，容错大小写与空格。"""
    v = (value or "").strip()
    if not v:
        return ""
    if v.upper() == "IT":
        return "IT"
    for cat in VALID_CATEGORIES:
        if v == cat:
            return cat
    for cat in VALID_CATEGORIES:
        if cat.lower() in v.lower():
            return cat
    return v


def call_dify_chat(
    api_url: str,
    api_key: str,
    question: str,
    user: str,
    timeout: int = DEFAULT_TIMEOUT,
    retry: int = DEFAULT_RETRY,
    conversation_id: str = "",
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    调用 Dify Chat API（blocking 模式）。
    返回：{"ok": bool, "answer": str, "message_id": str, "conversation_id": str,
           "elapsed": float, "status": int, "error": str, "raw": dict}
    """
    payload = {
        "inputs": inputs or {},
        "query": question,
        "response_mode": "blocking",
        "conversation_id": conversation_id or "",
        "user": user,
        "auto_generate_name": True,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_err = ""
    for attempt in range(retry + 1):
        started = time.time()
        try:
            req = urlrequest.Request(api_url, data=body, headers=headers, method="POST")
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
                status = getattr(resp, "status", 200)
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            try:
                raw = json.loads(raw_text)
            except json.JSONDecodeError:
                raw = {"answer": raw_text}
            elapsed = time.time() - started
            return {
                "ok": True,
                "answer": raw.get("answer", "") if isinstance(raw, dict) else str(raw),
                "message_id": raw.get("message_id", "") if isinstance(raw, dict) else "",
                "conversation_id": raw.get("conversation_id", "") if isinstance(raw, dict) else "",
                "elapsed": elapsed,
                "status": status,
                "error": "",
                "raw": raw,
            }
        except urlerror.HTTPError as exc:  # 4xx / 5xx
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            last_err = f"HTTPError {exc.code}: {detail or exc.reason}"
            if exc.code in (401, 403):
                break  # 鉴权问题不必重试
        except urlerror.URLError as exc:
            last_err = f"URLError: {exc.reason}"
        except Exception as exc:  # 超时、解析异常等
            last_err = f"{type(exc).__name__}: {exc}"

        if attempt < retry:
            time.sleep(1.5 * (attempt + 1))

    return {
        "ok": False,
        "answer": "",
        "message_id": "",
        "conversation_id": "",
        "elapsed": time.time() - started,
        "status": 0,
        "error": last_err or "unknown error",
        "raw": {},
    }


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def keyword_hit(answer_text: str, keywords: List[str]) -> Tuple[int, int, List[str]]:
    """返回 (命中数, 关键词总数, 未命中列表)。"""
    if not keywords:
        return 0, 0, []
    text = answer_text or ""
    hit, miss = 0, []
    for kw in keywords:
        if kw and kw in text:
            hit += 1
        else:
            miss.append(kw)
    return hit, len(keywords), miss


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总所有单条结果，生成评估指标。"""
    total = len(records)
    ok_records = [r for r in records if r["api_ok"]]
    failed = total - len(ok_records)

    # 分类准确率
    cat_correct = sum(1 for r in ok_records if r["category_correct"])
    cat_accuracy = cat_correct / len(ok_records) if ok_records else 0.0

    # 每类别 P/R/F1 + 混淆矩阵
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for r in ok_records:
        gold = r["expected_category"]
        pred = r["predicted_category"] or "(空)"
        confusion[gold][pred] += 1
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    per_category: Dict[str, Dict[str, float]] = {}
    for cat in VALID_CATEGORIES:
        support = tp[cat] + fn[cat]
        if support == 0 and fp[cat] == 0:
            continue
        p, rc, f1 = prf(tp[cat], fp[cat], fn[cat])
        per_category[cat] = {
            "support": support,
            "correct": tp[cat],
            "precision": round(p, 4),
            "recall": round(rc, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = (
        sum(v["f1"] for v in per_category.values()) / len(per_category) if per_category else 0.0
    )

    # 兜底 / 转人工统计
    fallback_records = [r for r in ok_records if r["is_fallback"]]
    need_human_records = [r for r in ok_records if r["need_human"]]

    # 「其他」类应触发兜底，非「其他」类触发兜底视为知识库覆盖不足
    other_records = [r for r in ok_records if r["expected_category"] == "其他"]
    other_fallback = [r for r in other_records if r["is_fallback"]]
    nonother_fallback = [r for r in ok_records if r["expected_category"] != "其他" and r["is_fallback"]]

    # 关键词命中
    kw_hit_total = sum(r["kw_hit"] for r in ok_records)
    kw_all_total = sum(r["kw_total"] for r in ok_records)
    kw_rate = kw_hit_total / kw_all_total if kw_all_total else 0.0
    full_kw_records = [r for r in ok_records if r["kw_total"] and r["kw_hit"] == r["kw_total"]]

    # 引用来源
    has_source = [r for r in ok_records if r["source"] and r["source"].lower() != "none"]

    # 时延
    latencies = [r["elapsed"] for r in ok_records if r["elapsed"]]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # 结构化输出合规
    parse_ok = [r for r in ok_records if r["parse_ok"]]
    # 记录中的键名与输出字段名的映射（category 存为 predicted_category）
    field_key_map = {"category": "predicted_category"}
    field_missing = Counter()
    for r in ok_records:
        for f in REQUIRED_OUTPUT_FIELDS:
            key = field_key_map.get(f, f)
            if r.get(key) in (None, ""):
                field_missing[f] += 1

    # 错误分类明细
    wrong = [
        {
            "id": r["id"],
            "question": r["question"],
            "expected": r["expected_category"],
            "predicted": r["predicted_category"],
            "topic": r["topic"],
        }
        for r in ok_records
        if not r["category_correct"]
    ]

    return {
        "total": total,
        "api_success": len(ok_records),
        "api_failed": failed,
        "success_rate": round(len(ok_records) / total, 4) if total else 0.0,
        "category_correct": cat_correct,
        "category_accuracy": round(cat_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_category": per_category,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "fallback_count": len(fallback_records),
        "fallback_rate": round(len(fallback_records) / len(ok_records), 4) if ok_records else 0.0,
        "fallback_on_other_count": len(other_fallback),
        "fallback_on_other_rate": round(len(other_fallback) / len(other_records), 4) if other_records else 0.0,
        "unexpected_fallback_count": len(nonother_fallback),
        "unexpected_fallback_questions": [
            {"id": r["id"], "question": r["question"], "expected": r["expected_category"]}
            for r in nonother_fallback
        ],
        "need_human_count": len(need_human_records),
        "need_human_rate": round(len(need_human_records) / len(ok_records), 4) if ok_records else 0.0,
        "keyword_hit": kw_hit_total,
        "keyword_total": kw_all_total,
        "keyword_hit_rate": round(kw_rate, 4),
        "keyword_full_hit_count": len(full_kw_records),
        "keyword_full_hit_rate": round(len(full_kw_records) / len(ok_records), 4) if ok_records else 0.0,
        "has_source_count": len(has_source),
        "has_source_rate": round(len(has_source) / len(ok_records), 4) if ok_records else 0.0,
        "structured_parse_ok_count": len(parse_ok),
        "structured_parse_ok_rate": round(len(parse_ok) / len(ok_records), 4) if ok_records else 0.0,
        "missing_fields": dict(field_missing),
        "avg_latency_sec": round(avg_latency, 3),
        "max_latency_sec": round(max(latencies), 3) if latencies else 0.0,
        "wrong_classifications": wrong,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def print_record(idx: int, total: int, rec: Dict[str, Any]) -> None:
    mark = "✓" if rec["category_correct"] else "✗"
    fb = " [兜底]" if rec["is_fallback"] else ""
    print("-" * 78)
    print(f"[{idx}/{total}] {rec['id']} {mark}{fb}")
    print(f"  问题: {rec['question']}")
    print(f"  期望: {rec['expected_category']}   实际: {rec['predicted_category'] or '(空)'}")
    answer = (rec["answer"] or "").replace("\n", " ")
    if len(answer) > 260:
        answer = answer[:260] + "…"
    print(f"  回答: {answer}")
    print(f"  建议: {(rec['suggestion'] or '')[:120]}")
    print(f"  来源: {rec['source']}   转人工: {rec['need_human']}   时延: {rec['elapsed']:.2f}s")
    if rec["kw_miss"]:
        print(f"  未命中关键词: {', '.join(rec['kw_miss'])}")
    if not rec["api_ok"]:
        print(f"  调用失败: {rec['error']}")


def print_report(metrics: Dict[str, Any], started_at: str) -> None:
    line = "=" * 78
    print()
    print(line)
    print("最终评估结果")
    print(line)
    print(f"开始时间            : {started_at}")
    print(f"问题总数            : {metrics['total']}")
    print(f"API 成功 / 失败     : {metrics['api_success']} / {metrics['api_failed']}"
          f"  （成功率 {metrics['success_rate']:.2%}）")
    print()
    print(f"★ 分类准确率        : {metrics['category_correct']}/{metrics['api_success']}"
          f" = {metrics['category_accuracy']:.2%}")
    print(f"★ 宏平均 F1         : {metrics['macro_f1']:.4f}")
    print(f"★ 兜底命中数        : {metrics['fallback_count']} 条包含"
          f"「{FALLBACK_TEXT}」  （兜底率 {metrics['fallback_rate']:.2%}）")
    print(f"  ├─ 「其他」类正确兜底 : {metrics['fallback_on_other_count']} 条"
          f"（占「其他」类 {metrics['fallback_on_other_rate']:.2%}）")
    print(f"  └─ 非「其他」类意外兜底: {metrics['unexpected_fallback_count']} 条（知识库覆盖不足信号）")
    print(f"★ need_human=true   : {metrics['need_human_count']} 条"
          f"（{metrics['need_human_rate']:.2%}）")
    print(f"★ 关键词命中率      : {metrics['keyword_hit']}/{metrics['keyword_total']}"
          f" = {metrics['keyword_hit_rate']:.2%}   （整题全命中 {metrics['keyword_full_hit_rate']:.2%}）")
    print(f"★ 引用来源填写率    : {metrics['has_source_rate']:.2%}")
    print(f"★ 结构化输出合规率  : {metrics['structured_parse_ok_rate']:.2%}")
    print(f"★ 平均时延          : {metrics['avg_latency_sec']}s   最大 {metrics['max_latency_sec']}s")
    if metrics["missing_fields"]:
        print(f"  输出字段缺失统计  : {metrics['missing_fields']}")

    if metrics["per_category"]:
        print()
        print(f"{'类别':<6}{'样本':>6}{'正确':>6}{'精确率':>10}{'召回率':>10}{'F1':>10}")
        print("-" * 48)
        for cat, v in metrics["per_category"].items():
            print(f"{cat:<6}{v['support']:>6}{v['correct']:>6}"
                  f"{v['precision']:>10.2%}{v['recall']:>10.2%}{v['f1']:>10.4f}")

    if metrics["confusion_matrix"]:
        print()
        print("混淆矩阵（行=期望，列=预测）")
        preds = sorted({p for row in metrics["confusion_matrix"].values() for p in row})
        header = "  期望\\预测  " + "".join(f"{p:>8}" for p in preds)
        print(header)
        for gold, row in metrics["confusion_matrix"].items():
            print(f"  {gold:<10}" + "".join(f"{row.get(p, 0):>8}" for p in preds))

    if metrics["wrong_classifications"]:
        print()
        print(f"分类错误明细（{len(metrics['wrong_classifications'])} 条）")
        for w in metrics["wrong_classifications"][:20]:
            print(f"  - {w['id']} 期望={w['expected']} 实际={w['predicted']} | {w['question']}")
        if len(metrics["wrong_classifications"]) > 20:
            print(f"  …其余 {len(metrics['wrong_classifications']) - 20} 条见导出文件")

    if metrics["unexpected_fallback_questions"]:
        print()
        print("非预期兜底明细（建议补充知识库内容）")
        for w in metrics["unexpected_fallback_questions"][:15]:
            print(f"  - {w['id']} [{w['expected']}] {w['question']}")

    print(line)


def save_outputs(records: List[Dict[str, Any]], metrics: Dict[str, Any], out_dir: str, prefix: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"{prefix}_report_{stamp}.json")
    csv_path = os.path.join(out_dir, f"{prefix}_details_{stamp}.csv")

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "generated_at": stamp,
                "fallback_sentence": FALLBACK_TEXT,
                "metrics": metrics,
                "records": [
                    {k: v for k, v in r.items() if k not in ("raw",)} for r in records
                ],
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = [
        "id", "topic", "question", "expected_category", "predicted_category",
        "category_correct", "answer", "suggestion", "source", "need_human",
        "is_fallback", "kw_hit", "kw_total", "kw_miss", "api_ok", "elapsed", "error",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["kw_miss"] = ";".join(r.get("kw_miss") or [])
            writer.writerow(row)

    return {"json": json_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# 离线模式
# ---------------------------------------------------------------------------

def offline_answer(item: Dict[str, Any]) -> str:
    """离线演练用：构造一个符合输出格式的占位回答，不调用任何 API。"""
    if item["expected_category"] == "其他":
        obj = {
            "question": item["question"],
            "category": "其他",
            "answer": FALLBACK_TEXT,
            "suggestion": "该问题暂未被知识库覆盖，建议提交人工工单由归口部门跟进。",
            "source": "none",
            "need_human": True,
        }
    else:
        obj = {
            "question": item["question"],
            "category": item["expected_category"],
            "answer": "【离线演练占位回答】" + "、".join(item["expected_keywords"]) + "（示例内容，未调用 Dify API）",
            "suggestion": "请通过对应内部系统入口办理（示例渠道）。",
            "source": "offline_demo.md#占位",
            "need_human": False,
        }
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="企业智能问答与工单分类助手 —— Dify 批量测试与评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--questions", default=DEFAULT_QUESTION_FILE, help="测试问题集 JSON 路径")
    p.add_argument("--api-url", default=os.environ.get("DIFY_API_URL", API_URL_DEFAULT),
                   help=f"Dify Chat API 地址，默认 {API_URL_DEFAULT}")
    p.add_argument("--api-key", default=None,
                   help="Dify 应用 API Key（留空则读取环境变量 DIFY_API_KEY）")
    p.add_argument("--user", default="eval-script", help="Dify 要求的 user 标识，默认 eval-script")
    p.add_argument("--limit", type=int, default=0, help="仅测试前 N 条，0 表示全部")
    p.add_argument("--category", default="", help=f"仅测试指定类别，可选 {VALID_CATEGORIES}")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单次请求超时（秒）")
    p.add_argument("--retry", type=int, default=DEFAULT_RETRY, help="失败重试次数")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="两次请求之间的间隔（秒），用于限流")
    p.add_argument("--quiet", action="store_true", help="不逐条打印，只输出汇总")
    p.add_argument("--out-dir", default=None, help="评估结果导出目录，默认脚本所在目录下 eval_output")
    p.add_argument("--prefix", default="eval", help="导出文件名前缀")
    p.add_argument("--validate", action="store_true", help="只校验测试集格式，不调用 API")
    p.add_argument("--offline", action="store_true", help="离线演练：不调用 API，用占位回答跑通评估流程")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _setup_console()
    args = build_parser().parse_args(argv)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) 读取测试集
    try:
        questions, meta = load_questions(args.questions)
    except Exception as exc:
        print(f"[错误] 读取测试集失败：{exc}")
        return 2

    problems = validate_questions(questions)
    if problems:
        print(f"[警告] 测试集存在 {len(problems)} 处问题：")
        for p in problems[:20]:
            print(f"  - {p}")

    if args.category:
        questions = [q for q in questions if q["expected_category"] == args.category]
        if not questions:
            print(f"[错误] 没有类别为 {args.category} 的测试问题")
            return 2
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    print("=" * 78)
    print("企业智能问答与工单分类助手 —— 批量测试")
    print("=" * 78)
    print(f"测试集        : {args.questions}")
    print(f"待测问题数    : {len(questions)}")
    dist = Counter(q["expected_category"] for q in questions)
    print(f"类别分布      : {dict(dist)}")
    print(f"API 地址      : {args.api_url}")
    print(f"运行模式      : {'离线演练（不调用 API）' if args.offline else '在线调用 Dify API'}")
    if meta:
        print(f"测试集说明    : {meta.get('description', '')}")

    if args.validate:
        print()
        if problems:
            print(f"校验结果：不通过，共 {len(problems)} 处问题（见上方列表）")
            return 1
        print("校验结果：通过。测试集格式合法，类别取值全部在允许范围内。")
        return 0

    # 2) 校验 API Key
    api_key = args.api_key or os.environ.get("DIFY_API_KEY") or API_KEY
    if not args.offline and (not api_key or api_key == API_KEY):
        print()
        print("[错误] 未提供有效的 API Key。")
        print("       方式一： python run_assistant.py --api-key app-你的密钥")
        print("       方式二： 设置环境变量 DIFY_API_KEY 后再运行")
        print("       方式三： 先用 --offline 演练流程，或用 --validate 校验测试集")
        print("       注意：不要把真实密钥硬编码进脚本或提交到代码仓库。")
        return 2

    # 3) 逐条执行
    records: List[Dict[str, Any]] = []
    total = len(questions)
    for idx, item in enumerate(questions, start=1):
        if args.offline:
            api_result: Dict[str, Any] = {
                "ok": True,
                "answer": offline_answer(item),
                "elapsed": 0.0,
                "error": "",
                "message_id": "",
                "conversation_id": "",
            }
        else:
            api_result = call_dify_chat(
                api_url=args.api_url,
                api_key=api_key,
                question=item["question"],
                user=args.user,
                timeout=args.timeout,
                retry=args.retry,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

        parsed = parse_structured_answer(api_result.get("answer", ""))
        predicted = normalize_category(parsed.get("category", ""))
        answer_text = parsed.get("answer", "") or api_result.get("answer", "")
        kw_hit, kw_total, kw_miss = keyword_hit(answer_text, item["expected_keywords"])

        rec: Dict[str, Any] = {
            "id": item["id"],
            "topic": item["topic"],
            "question": item["question"],
            "expected_category": item["expected_category"],
            "predicted_category": predicted,
            "category_correct": bool(api_result["ok"]) and predicted == item["expected_category"],
            "answer": answer_text,
            "suggestion": parsed.get("suggestion", ""),
            "source": parsed.get("source", "none"),
            "need_human": bool(parsed.get("need_human", False)),
            "is_fallback": FALLBACK_TEXT in (answer_text or ""),
            "kw_hit": kw_hit,
            "kw_total": kw_total,
            "kw_miss": kw_miss,
            "parse_ok": bool(parsed.get("parse_ok")),
            "api_ok": bool(api_result["ok"]),
            "elapsed": float(api_result.get("elapsed") or 0.0),
            "error": api_result.get("error", ""),
            "message_id": api_result.get("message_id", ""),
        }
        records.append(rec)
        if not args.quiet:
            print_record(idx, total, rec)

    # 4) 汇总评估
    metrics = evaluate(records)
    print_report(metrics, started_at)

    # 5) 导出
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output")
    try:
        paths = save_outputs(records, metrics, out_dir, args.prefix)
        print(f"评估报告已导出：{paths['json']}")
        print(f"明细数据已导出：{paths['csv']}")
    except Exception as exc:
        print(f"[警告] 导出结果失败：{exc}")

    # 6) 退出码：准确率达标（示例阈值 0.8）且无 API 失败视为成功
    if metrics["api_failed"] > 0:
        return 1
    return 0 if metrics["category_accuracy"] >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
