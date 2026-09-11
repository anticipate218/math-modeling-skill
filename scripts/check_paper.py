#!/usr/bin/env python3
"""数学建模论文自检工具（国赛 / 研赛 / 美赛）。

对论文草稿（Markdown / 纯文本 / LaTeX）做结构与合规性检查，输出 FAIL/WARN/INFO
清单。仅使用 Python 标准库，非交互，可直接在 CI 或 agent 流程里调用。

用法示例：
    python check_paper.py paper.md --contest cumcm
    python check_paper.py paper.md --contest mcm --json
    python check_paper.py --self-test

退出码：
    0  无 FAIL（可能有 WARN/INFO）
    1  存在 FAIL
    2  用法或读取错误

规则依据（部分）：
    全国大学生数学建模竞赛论文格式规范（2026 年修订稿）：
      摘要页第三页起、正文不超过 30 页、不得出现身份/学校/赛区信息、
      附录须含可运行源程序或声明"本论文没有用到程序"、参考文献须规范标注。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- 竞赛档案

# 各竞赛要求的正文章节（同义词用 | 分隔，命中任一即算通过）
SECTION_RULES: dict[str, dict[str, list[str]]] = {
    "cumcm": {
        "摘要": [r"摘\s*要", r"abstract"],
        "关键词": [r"关键词", r"关键字", r"key\s*words?"],
        "问题重述": [r"问题重述", r"问题提出", r"问题背景"],
        "问题分析": [r"问题分析"],
        "模型假设": [r"模型假设", r"假设与符号", r"基本假设"],
        "符号说明": [r"符号说明", r"符号约定", r"变量说明", r"记号说明"],
        "模型建立与求解": [r"模型(的)?(建立|构建)", r"建模", r"模型求解"],
        "结果分析与检验": [r"结果(的)?(分析|检验|验证)", r"模型检验", r"误差分析"],
        # 注：「灵敏度分析」「模型评价」不列为必备章节。实证（2021–2025 年 64 篇国赛获奖论文）
        # 显示二者独立成章的比例仅约 19% 与 28%，多数并入"建模与求解"各问之后；
        # 因此改为内容级 WARN 检查（见 check_sensitivity_and_evaluation）。
        "参考文献": [r"参考文献", r"references"],
        "附录": [r"附\s*录", r"appendix"],
    },
    "yjs": {  # 研究生数学建模（华为杯）：结构与国赛接近，摘要/检验要求同样高
        "摘要": [r"摘\s*要", r"abstract"],
        "关键词": [r"关键词", r"关键字", r"key\s*words?"],
        "问题重述": [r"问题重述", r"问题提出", r"问题背景"],
        "问题分析": [r"问题分析"],
        "模型假设": [r"模型假设", r"基本假设"],
        "符号说明": [r"符号说明", r"符号约定", r"变量说明"],
        "模型建立与求解": [r"模型(的)?(建立|构建)", r"建模", r"模型求解"],
        "结果分析与检验": [r"结果(的)?(分析|检验|验证)", r"模型检验", r"误差分析"],
        "参考文献": [r"参考文献", r"references"],
        "附录": [r"附\s*录", r"appendix"],
    },
    "mcm": {  # MCM/ICM：Summary Sheet + 英文写作 + Strengths/Weaknesses
        "Summary": [r"summary", r"摘\s*要"],
        "Introduction/Background": [r"introduction", r"background", r"问题重述", r"restatement"],
        "Assumptions": [r"assumption", r"假设"],
        "Notations": [r"notation", r"符号说明", r"variables"],
        "Model Development": [r"model(ing)?\s+(development|construction|building)", r"模型建立", r"the model"],
        "Results": [r"results?", r"结果"],
        "Sensitivity Analysis": [r"sensitivit", r"robustness", r"灵敏"],
        "Strengths and Weaknesses": [r"strengths?\s*(and|&)\s*weakness", r"优缺点", r"模型评价"],
        "References": [r"references", r"参考文献"],
        "Appendix/Memo": [r"appendix", r"memo", r"附\s*录", r"letter"],
    },
    "generic": {
        "摘要": [r"摘\s*要", r"summary", r"abstract"],
        "关键词": [r"关键词", r"key\s*words?"],
        "问题分析": [r"问题(重述|分析|提出)"],
        "模型假设": [r"假设"],
        "符号说明": [r"符号(说明|约定)", r"notation"],
        "模型建立与求解": [r"模型(建立|构建|求解)"],
        "结果与分析": [r"结果", r"results?"],
        "检验/灵敏度": [r"检验|验证|灵敏|稳健|sensitivit", ],
        "参考文献": [r"参考文献", r"references"],
        "附录": [r"附\s*录", r"appendix"],
    },
}

# 匿名合规：出现这些词就要人工确认（国赛/研赛严禁身份信息）
IDENTITY_PATTERNS = [
    r"[\u4e00-\u9fa5]{2,15}大学",
    r"[\u4e00-\u9fa5]{2,15}学院",
    r"[\u4e00-\u9fa5]{2,10}职业技术学院",
    r"(指导)?教师[：:]\s*\S",
    r"队\s*号[：:]\s*\S",
    r"参赛队[号]?[：:]\s*\S",
    r"学\s*号[：:]\s*\S",
    r"赛\s*区[：:]\s*\S",
    r"@[a-zA-Z0-9.-]+\.(com|cn|edu|org|net)",
    r"(?<!\d)1[3-9]\d{9}(?!\d)",
]

ABSTRACT_INGREDIENTS = {
    "问题/目标": [r"问题[一二三1-9]?", r"针对", r"本文(研究|解决|讨论)", r"要求"],
    "方法/模型": [r"模型", r"算法", r"方法", r"优化", r"回归", r"微分方程", r"网络", r"仿真"],
    "结果/结论": [r"结果(表明|显示)", r"求得", r"得到", r"结论", r"最优(解|值)", r"误差", r"精度", r"提升|提高"],
    "关键词": [r"关键词", r"关键字"],
}

PROGRAM_DECLARATIONS = [
    r"本论文没有用到程序",
    r"没有用到程序",
    r"未使用程序",
]
SUPPORT_DECLARATIONS = [r"本论文没有支撑材料", r"无支撑材料"]

CODE_HINTS = [r"\bdef\s+\w+\s*\(", r"\bimport\s+\w+", r"\bfunction\s+\w+\s*\(",
              r"\bfor\s*\(", r"\bwhile\s*\(", r"#include", r"disp\(", r"\bmodel\b\s*:", r"<<", r"```"]

# AI 工具使用声明（国赛 2026 年试行规定第 3 条：置于参考文献之前）
AI_DECL_HEADING = r"AI\s*工具使用声明|人工智能工具使用声明|AI\s*Usage\s*Statement"
AI_DECL_NONE = r"未使用任何\s*AI\s*工具|未使用任何人工智能"
AI_DECL_USED = r"使用了\s*AI\s*工具|使用了人工智能"
# 美赛：Report on Use of AI（附于 25 页正文之后，不计页数）
MCM_AI_REPORT = r"report\s+on\s+use\s+of\s+ai|use\s+of\s+ai"

FIGURE_RE = re.compile(r"(?:图|figure|fig\.?)\s*([0-9]+)", re.IGNORECASE)
TABLE_RE = re.compile(r"(?:表|table)\s*([0-9]+)", re.IGNORECASE)
CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[-,，]\s*\d+)*\s*\]")


@dataclass
class Finding:
    level: str  # FAIL | WARN | INFO
    code: str
    message: str
    hint: str = ""


def read_text(path: Path) -> str:
    """读取论文草稿。支持 .pdf（需 pypdf，可选依赖）。"""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise SystemExit(
                "读取 PDF 需要可选依赖：pip install pypdf（或改用 Markdown/文本草稿）"
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def check_sections(text: str, contest: str) -> list[Finding]:
    findings: list[Finding] = []
    rules = SECTION_RULES[contest]
    plain = strip_code_blocks(text)
    missing = []
    for name, patterns in rules.items():
        if not any(re.search(p, plain, re.IGNORECASE) for p in patterns):
            missing.append(name)
    if missing:
        findings.append(
            Finding("FAIL", "structure",
                    "缺少必备章节：" + "、".join(missing),
                    "按竞赛规范补齐章节标题；缺失章节是最常见的直接失分点。")
        )
    else:
        findings.append(Finding("INFO", "structure", f"必备章节齐全（{len(rules)} 项）"))
    return findings


def check_abstract(text: str, contest: str) -> list[Finding]:
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    # 摘要区块：从"摘要/Summary"到下一个标题（Markdown 标题或中文序号标题）
    m = re.search(
        r"(?:摘\s*要|summary|abstract)\s*[:：]?\s*\n?(.*?)(?=\n\s*#{1,4}\s|\n\s*[一二三四五六七八九十]+\s*、|\Z)",
        plain, re.IGNORECASE | re.DOTALL)
    if not m:
        # 摘要标题缺失由 structure 检查负责，这里只做提示
        return findings
    body = m.group(1)
    if contest == "mcm":
        words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", body))
        if words and words < 120:
            findings.append(Finding("WARN", "abstract-length",
                                    f"Summary Sheet 偏短（约 {words} 英文词）",
                                    "MCM/ICM 的 Summary 往往需 250-400 词，且是评审第一入口。"))
        else:
            findings.append(Finding("INFO", "abstract-length", f"Summary 约 {words} 英文词"))
    else:
        chars = len(re.sub(r"\s", "", body))
        if chars > 1400:
            findings.append(Finding("WARN", "abstract-length",
                                    f"摘要约 {chars} 字，可能超出一页",
                                    "国赛规范要求摘要原则上不超过一页（含标题与关键词）。"))
        else:
            findings.append(Finding("INFO", "abstract-length", f"摘要约 {chars} 字"))
    missing = [k for k, pats in ABSTRACT_INGREDIENTS.items()
               if not any(re.search(p, body, re.IGNORECASE) for p in pats)]
    if missing:
        findings.append(Finding("FAIL", "abstract-ingredients",
                                "摘要缺少要素：" + "、".join(missing),
                                "摘要必须让评委在 1 分钟内看到：问题→方法→结果→结论。"))
    else:
        findings.append(Finding("INFO", "abstract-ingredients", "摘要四要素齐备"))
    return findings


def check_anonymity(text: str, contest: str) -> list[Finding]:
    if contest == "mcm":
        return [Finding("INFO", "anonymity", "MCM/ICM 无匿名要求，跳过身份检查")]
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    hits = []
    for pat in IDENTITY_PATTERNS:
        for mm in re.finditer(pat, plain):
            snippet = mm.group(0).strip()
            if snippet not in hits:
                hits.append(snippet)
    if hits:
        findings.append(Finding("FAIL", "anonymity",
                                "疑似出现身份/学校/赛区信息：" + "、".join(hits[:8]),
                                "国赛与研赛严禁在摘要页、正文、附录出现身份信息，可能被取消评奖资格。"))
    else:
        findings.append(Finding("INFO", "anonymity", "未检出疑似身份信息"))
    return findings


def check_numbering(text: str, contest: str) -> list[Finding]:
    findings: list[Finding] = []
    for label, rx in (("图", FIGURE_RE), ("表", TABLE_RE)):
        nums = [int(n) for n in rx.findall(text)]
        if not nums:
            findings.append(Finding("WARN", "numbering", f"未检出{label}编号",
                                    f"建模论文通常需要{label}来承载结果，且编号需连续。"))
            continue
        uniq = sorted(set(nums))
        expected = list(range(1, len(uniq) + 1))
        if uniq != expected:
            findings.append(Finding("WARN", "numbering",
                                    f"{label}编号不连续：{uniq}",
                                    f"检查是否有缺号、重号或未编号的{label}。"))
        else:
            findings.append(Finding("INFO", "numbering", f"{label}编号连续（共 {len(uniq)} 个）"))
    return findings


def check_citations(text: str, contest: str) -> list[Finding]:
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    has_ref_section = bool(re.search(r"参考文献|references", plain, re.IGNORECASE))
    inline = CITATION_RE.findall(plain)
    if has_ref_section and not inline:
        findings.append(Finding("FAIL", "citation",
                                "有参考文献列表，但正文未见 [n] 形式的引用标注",
                                "规范要求引用他人成果须在正文引用处标注。"))
    elif inline:
        findings.append(Finding("INFO", "citation", f"检出正文引用标注 {len(inline)} 处"))
    elif not has_ref_section:
        findings.append(Finding("WARN", "citation", "未检出参考文献部分",
                                "即使主要自建模型，也应列出数据来源与方法出处。"))
    return findings


def check_appendix(text: str, contest: str) -> list[Finding]:
    findings: list[Finding] = []
    appendix = ""
    m = re.search(r"(附\s*录|appendix|memo)(.*)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        appendix = m.group(2)
    declared_none = any(re.search(p, text) for p in PROGRAM_DECLARATIONS)
    has_code = any(re.search(p, appendix) for p in CODE_HINTS)
    if contest == "mcm":
        if not (has_code or re.search(r"appendix|memo", text, re.IGNORECASE)):
            findings.append(Finding("WARN", "appendix", "未见附录/程序",
                                    "MCM/ICM 建议附代码或说明可复现性。"))
        else:
            findings.append(Finding("INFO", "appendix", "附录/程序信息存在"))
        return findings
    if declared_none:
        findings.append(Finding("INFO", "appendix", "已声明未使用程序"))
    elif has_code:
        findings.append(Finding("INFO", "appendix", "附录疑似包含源程序"))
    else:
        findings.append(Finding("FAIL", "appendix",
                                "附录未检出源程序，也未声明“本论文没有用到程序”",
                                "国赛规范：缺少必要源程序或程序不能运行，可能被取消评奖资格。"))
    if not any(re.search(p, text) for p in SUPPORT_DECLARATIONS):
        if not re.search(r"支撑材料", text):
            findings.append(Finding("WARN", "support-material",
                                    "未提及支撑材料文件列表",
                                    "附录应包含支撑材料文件列表；确无则注明“本论文没有支撑材料”。"))
        else:
            findings.append(Finding("INFO", "support-material", "提及支撑材料"))
    return findings


def check_ai_disclosure(text: str, contest: str) -> list[Finding]:
    """AI 使用声明检查。

    国赛《人工智能工具使用规定（2026 年试行）》第 3 条：须在参考文献之前设置
    「AI 工具使用声明」，二者择一；第 5 条：隐瞒或虚假声明 → 取消评奖资格。
    研赛《人工智能工具及输出使用规定（2025）》要求标注 AI 参与内容与工具信息。
    美赛 COMAP 政策：须在报告中说明并在参考文献列出，另附 Report on Use of AI。
    """
    findings: list[Finding] = []
    plain = strip_code_blocks(text)

    if contest in {"cumcm", "yjs"}:
        decl = re.search(AI_DECL_HEADING, plain, re.IGNORECASE)
        if not decl:
            findings.append(Finding(
                "FAIL", "ai-disclosure", "未检出「AI工具使用声明」",
                "国赛 2026 年试行规定第 3 条：须在参考文献之前设置该声明；缺失或不实声明"
                "可能被取消评奖资格。未使用也要声明。"))
        elif not re.search(AI_DECL_NONE, plain) and not re.search(AI_DECL_USED, plain):
            findings.append(Finding(
                "FAIL", "ai-disclosure", "AI 工具使用声明缺少规定表述",
                "二选一：未使用→「本参赛队在竞赛过程中未使用任何AI工具。」；"
                "已使用→「本参赛队在竞赛过程中使用了AI工具，主要用于【简要用途】，"
                "详细使用情况见支撑材料。」"))
        else:
            used = bool(re.search(AI_DECL_USED, plain))
            findings.append(Finding("INFO", "ai-disclosure",
                                    f"AI 工具使用声明存在（声明为：{'已使用' if used else '未使用'}）"))
            if used:
                findings.append(Finding(
                    "WARN", "ai-detail",
                    "声明使用了 AI：支撑材料中需包含「AI工具使用详情.pdf」",
                    "须含：工具名称/版本或型号、使用目的与环节、主要提示方式与使用过程、"
                    "对输出的采纳与人工核验情况（语言润色除外）。"))
        # 位置要求：声明须在参考文献之前
        ref = re.search(r"参考文献|references", plain, re.IGNORECASE)
        if decl and ref and decl.start() > ref.start():
            findings.append(Finding("FAIL", "ai-disclosure-position",
                                    "「AI工具使用声明」位于参考文献之后",
                                    "规定要求设置在参考文献之前。"))
    elif contest == "mcm":
        if not re.search(MCM_AI_REPORT, plain, re.IGNORECASE):
            findings.append(Finding(
                "WARN", "ai-disclosure", "未检出 Report on Use of AI",
                "COMAP 政策：使用 AI 须在报告中明确说明并在参考文献中列出所用工具，"
                "并在 25 页正文之后附「Report on Use of AI」（该部分不计页数）。"
                "未使用 AI 也建议在参考文献中说明。"))
        else:
            findings.append(Finding("INFO", "ai-disclosure", "存在 AI 使用报告"))
    return findings


def check_model_validation(text: str, contest: str) -> list[Finding]:
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    validation = [r"误差", r"残差", r"交叉验证", r"拟合优度", r"R\^?2", r"置信区间",
                  r"显著性", r"对比(实验|分析)", r"验证", r"检验"]
    if not any(re.search(p, plain, re.IGNORECASE) for p in validation):
        findings.append(Finding("WARN", "validation", "未检出结果验证/误差分析",
                                "只给结果不做检验是常见失分点，至少给误差或对比基线。"))
    else:
        findings.append(Finding("INFO", "validation", "存在结果验证/误差分析"))
    return findings


def check_length(text: str, contest: str) -> list[Finding]:
    chars = len(re.sub(r"\s", "", strip_code_blocks(text)))
    findings = [Finding("INFO", "length", f"正文约 {chars} 字（未计图片与公式排版）")]
    if contest in {"cumcm", "yjs"} and chars < 6000:
        findings.append(Finding("WARN", "length", "正文篇幅偏短",
                                "国赛/研赛正文上限 30 页；过短通常意味着建模与检验不充分。"))
    return findings


def check_sensitivity_and_evaluation(text: str, contest: str) -> list[Finding]:
    """灵敏度/稳健性与模型评价：内容级检查（不要求独立成章）。

    实证依据：2021–2025 年 64 篇国赛获奖论文中，灵敏度分析独立成章仅约 19%（含该内容约 36%）、
    模型评价独立成章仅约 28%，多数并入「建模与求解」各问之后。因此对国赛/研赛只做 WARN，
    避免把结构规范的获奖论文误判为不合格；但官方《章程》把「结果的分析和检验」「模型的改进」
    列为答卷必备内容，所以**内容**仍须存在。美赛官方明确要求 sensitivity 与 strengths/weaknesses，
    故对 mcm 按 FAIL 处理。
    """
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    has_sens = bool(re.search(r"灵敏(度|性)|稳健性|敏感性|sensitivity|robustness", plain, re.IGNORECASE))
    has_eval = bool(re.search(r"模型(的)?(评价|改进|推广)|优缺点|局限|strengths?\s*(and|&)?\s*weakness",
                              plain, re.IGNORECASE))
    level_sens = "FAIL" if contest == "mcm" else "WARN"
    level_eval = "FAIL" if contest == "mcm" else "WARN"

    if has_sens:
        findings.append(Finding("INFO", "sensitivity", "存在灵敏度/稳健性相关论述"))
    else:
        findings.append(Finding(
            level_sens, "sensitivity", "全文未检出灵敏度/稳健性分析",
            "官方《章程》把「结果的分析和检验」列为必备内容；国赛获奖论文中约 36% 含此内容"
            "（多为并入「建模与求解」各问之后，**不必独立成章**）。美赛则被官方明确点名要求。"))
    if has_eval:
        findings.append(Finding("INFO", "model-evaluation", "存在模型评价/改进相关论述"))
    else:
        findings.append(Finding(
            level_eval, "model-evaluation", "未检出模型评价/改进/局限的相关论述",
            "官方《章程》要求答卷包含「模型的改进」；美赛官方点名要求 strengths and weaknesses。"))
    return findings


# 单位混用检测：同一篇论文里同时出现互斥单位，通常意味着单位不统一
UNIT_CONFLICT_PAIRS = [
    ("时间", r"\d+\s*(?:小时|h\b|hrs?\b|hours?\b)", r"\d+\s*(?:分钟|min(?:ute)?s?\b)"),
    ("金额", r"\d+\s*万元", r"\d+\s*元(?!胞)"),
    ("长度", r"\d+\s*(?:千米|公里|km\b)",
     r"\d+\s*(?<!千)(?<!厘)(?<!毫)(?<!微)(?:米|m\b)"),
    ("质量", r"\d+\s*(?:千克|公斤|kg\b)", r"\d+\s*(?<!千)(?<!毫)(?:克|g\b)"),
]

REF_ENTRY_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+)$", re.MULTILINE)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def check_units(text: str, contest: str) -> list[Finding]:
    """单位混用：同一量纲下同时出现两种单位，提示人工确认（不判失败）。"""
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    conflicts = []
    for name, pat_a, pat_b in UNIT_CONFLICT_PAIRS:
        if re.search(pat_a, plain) and re.search(pat_b, plain):
            conflicts.append(name)
    if conflicts:
        findings.append(Finding(
            "WARN", "units", "疑似单位混用：" + "、".join(conflicts),
            "同一量纲出现两种单位时务必统一（或显式做换算），单位混用是低级但致命的错误。"))
    else:
        findings.append(Finding("INFO", "units", "未检出明显单位混用"))
    return findings


def check_figure_citation(text: str, contest: str) -> list[Finding]:
    """图表是否在正文被引用：编号只出现一次，通常意味着只有图题、正文未引用。"""
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    uncited = []
    for label, rx in (("图", FIGURE_RE), ("表", TABLE_RE)):
        counts: dict[int, int] = {}
        for n in rx.findall(plain):
            counts[int(n)] = counts.get(int(n), 0) + 1
        for num, cnt in sorted(counts.items()):
            if cnt < 2:
                uncited.append(f"{label}{num}")
    if uncited:
        findings.append(Finding(
            "WARN", "figure-citation", "以下图表编号只出现一次，可能未在正文引用：" + "、".join(uncited),
            "每个图表都应在正文被引用并解释其含义（规范与评阅都关注这一点）。"))
    else:
        findings.append(Finding("INFO", "figure-citation", "图表编号均被重复提及（疑似已引用）"))
    return findings


def check_references(text: str, contest: str) -> list[Finding]:
    """参考文献数量与条目完整性（作者. 题名. 出处, 年）粗检。"""
    findings: list[Finding] = []
    plain = strip_code_blocks(text)
    m = re.search(r"(参考文献|references)(.*)$", plain, re.IGNORECASE | re.DOTALL)
    if not m:
        return findings  # 缺少参考文献由 check_citations 负责
    block = m.group(2)
    entries = REF_ENTRY_RE.findall(block)
    if not entries:
        findings.append(Finding("WARN", "references", "参考文献部分未检出 [n] 形式的条目",
                                "正文引用与文献列表要一一对应。"))
        return findings
    findings.append(Finding("INFO", "references", f"检出参考文献 {len(entries)} 条"))
    if len(entries) < 3:
        findings.append(Finding("WARN", "references",
                                f"参考文献仅 {len(entries)} 条，偏少",
                                "即使是自建模型，也应列出数据来源、方法出处与背景资料。"))
    missing_year = [n for n, body in entries if not YEAR_RE.search(body)]
    if missing_year and len(missing_year) >= max(1, len(entries) // 2):
        findings.append(Finding(
            "WARN", "references", f"{len(missing_year)}/{len(entries)} 条文献未检出出版年份",
            "国赛/研赛要求按科技论文规范著录（研赛还要求引书标页码）；美赛须含规范引用。"))
    return findings


def run_checks(text: str, contest: str) -> list[Finding]:
    findings: list[Finding] = []
    findings += check_sections(text, contest)
    findings += check_sensitivity_and_evaluation(text, contest)
    findings += check_abstract(text, contest)
    findings += check_anonymity(text, contest)
    findings += check_numbering(text, contest)
    findings += check_figure_citation(text, contest)
    findings += check_citations(text, contest)
    findings += check_references(text, contest)
    findings += check_appendix(text, contest)
    findings += check_ai_disclosure(text, contest)
    findings += check_model_validation(text, contest)
    findings += check_units(text, contest)
    findings += check_length(text, contest)
    return findings


# ---------------------------------------------------------------- 自检固件

SELF_TEST_CASES: list[tuple[str, str, str]] = [
    ("好稿", "cumcm", """# 标题
摘要
针对问题一，本文建立多目标优化模型，采用遗传算法求解，结果误差为 1.2%，表明模型精度高。
关键词：优化；遗传算法

## 一、问题重述
## 二、问题分析
## 三、模型假设
## 四、符号说明
| 符号 | 含义 |
## 五、模型的建立与求解
我们建立规划模型并求解，得到最优解。见图 1 与表 1。
## 六、结果分析与检验
误差分析、残差检验、交叉验证均通过。
## 七、灵敏度分析
## 八、模型的评价与推广
## AI工具使用声明
本参赛队在竞赛过程中未使用任何AI工具。
## 参考文献
[1] 姜启源. 数学模型.
## 附录
支撑材料文件列表。以下为源程序：
```
def solve():
    import numpy
    return 1
```
"""),
    ("坏稿", "cumcm", """# 标题
摘要
本文研究了这个问题。
## 问题重述
## 模型的建立
结果见图 3 与图 1、表 2。
"""),
    ("美赛坏稿", "mcm", """# Summary
We model the problem and solve it.

## Introduction
## Assumptions
## Model Development
We use 30 minutes per step and 2 小时 in total.
## Results
## References
[1] Someone. A paper.
## Appendix
```
def solve():
    return 1
```
"""),
]


def self_test() -> int:
    failures = 0
    for name, contest, text in SELF_TEST_CASES:
        findings = run_checks(text, contest)
        fails = [f for f in findings if f.level == "FAIL"]
        print(f"[self-test] {name}: FAIL={len(fails)} WARN={sum(1 for f in findings if f.level == 'WARN')}")
        for f in findings:
            print(f"           - {f.level:4} {f.code}: {f.message}")
        expected_fail = "坏" in name
        if bool(fails) != expected_fail:
            print(f"[self-test] 预期不符：{name} 期望 FAIL={expected_fail}，实际 {bool(fails)}",
                  file=sys.stderr)
            failures += 1
    if failures:
        print(f"[self-test] 失败 {failures} 项", file=sys.stderr)
        return 1
    print("[self-test] 全部通过")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认非 UTF-8，会输出乱码；显式切到 UTF-8（失败则忽略）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(
        prog="check_paper.py",
        description="数学建模论文自检：结构完整性、摘要要素、匿名合规、编号连续性、引用、附录程序等。",
        epilog="示例：python check_paper.py paper.md --contest cumcm --json",
    )
    parser.add_argument("paper", nargs="?", help="论文草稿路径（.md/.txt/.tex/.pdf）")
    parser.add_argument("--contest", choices=sorted(SECTION_RULES), default="cumcm",
                        help="竞赛类型：cumcm=国赛，yjs=研赛，mcm=美赛，generic=通用（默认 cumcm）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    parser.add_argument("--self-test", action="store_true", help="运行内置固件自检并退出")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.paper:
        parser.print_help(sys.stderr)
        return 2

    path = Path(args.paper)
    if not path.is_file():
        print(f"错误：找不到文件 {path}", file=sys.stderr)
        return 2

    text = read_text(path)
    findings = run_checks(text, args.contest)
    fails = [f for f in findings if f.level == "FAIL"]
    warns = [f for f in findings if f.level == "WARN"]

    if args.json:
        print(json.dumps({
            "paper": str(path),
            "contest": args.contest,
            "summary": {"fail": len(fails), "warn": len(warns),
                        "info": len(findings) - len(fails) - len(warns)},
            "findings": [f.__dict__ for f in findings],
        }, ensure_ascii=False, indent=2))
    else:
        print(f"论文自检报告 — {path}（{args.contest}）")
        print(f"结果：FAIL {len(fails)} 项，WARN {len(warns)} 项\n")
        for f in findings:
            print(f"[{f.level:4}] {f.code}: {f.message}")
            if f.hint:
                print(f"        → {f.hint}")
        print("\n提示：本工具只做可机械校验的结构与合规检查，"
              "模型合理性、创新性与结果正确性必须人工复核。")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
