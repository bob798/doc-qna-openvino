"""
实体一致性守卫（Entity-Consistency Gate）

reranker 负责拦"语义主体串台"（火星探测器的额定功率 → 低分）；
本模块负责拦 reranker 不一定拦得住的**词面 ID 近似串台**：

  - "GB/T 2423.5 的实施日期"  —— 语料里只有 GB/T 2423.1，2423.5 不存在，
     但两者字面高度相似，cross-encoder 有可能给中高分；
  - "A500 型号的功率"        —— 语料里只有 A100/A200/A300，A500 不存在。

做法（保守、确定性、只在有正向证据时才拒答，绝不误伤域内问题）：
  1. 建索引时从所有 chunk 文本里抽出"域内已知实体"：产品型号码 + 标准号；
  2. 提问时用同样的规则从 query 抽实体；
  3. **只有当 query 明确提到某个"域内形态"的实体，而它不在已知集合里**，才判定
     为文档未覆盖并拒答；query 没提到任何可识别实体时不拦（交给 reranker）。

这样"火星探测器的额定功率"（没有型号/标准码）不会被本模块处理——那是 reranker
的活；而"GB/T 2423.5""A500"这类查得出的伪实体会被确定性地挡住。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── 实体抽取规则 ─────────────────────────────────────────────────────────────
# 每个家族：(名称, 正则)。正则第 0 组为整体匹配，用于归一化。
_STANDARD_RE = re.compile(
    r"(?:GB/T|GB|IEC|ISO|IEEE|JIS|EN)\s*\d[\d]*(?:[.\-][\d]+)*",
    re.IGNORECASE,
)
# 产品型号码：1~3 个字母 + 2~4 位数字，可带 -/空格 后缀（H100 PCIe / H100-NVL）。
# 要求首字母大写，避免误抓中文里夹的英文小词。
_MODEL_RE = re.compile(r"\b[A-Z]{1,3}\d{2,4}(?:[-\s][A-Za-z]{2,5})?\b")


def _norm_standard(s: str) -> str:
    """标准号归一化：大写、去空格、各种破折号统一成 '-'。"""
    s = s.upper()
    s = re.sub(r"[—–−~]", "-", s)
    s = re.sub(r"\s+", "", s)
    return s


def _norm_model(s: str) -> str:
    """型号归一化：大写、破折号→空格后压成单空格，便于 'H100 PCIe' 稳定匹配。"""
    s = s.upper().replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract(text: str) -> Dict[str, Set[str]]:
    """从一段文本抽出 {family: {归一化实体}}。"""
    out: Dict[str, Set[str]] = {"standard": set(), "model": set()}
    for m in _STANDARD_RE.findall(text):
        out["standard"].add(_norm_standard(m))
    for m in _MODEL_RE.findall(text):
        out["model"].add(_norm_model(m))
    return out


@dataclass
class EntityGate:
    """域内已知实体集合 + query 一致性检查。"""

    known: Dict[str, Set[str]] = field(
        default_factory=lambda: {"standard": set(), "model": set()}
    )
    # 标准号常带年份后缀（GB/T 2423.1—2008），query 可能只写 GB/T 2423.1，
    # 用前缀匹配容忍年份有无。min_prefix 防止过短前缀误判。
    min_prefix: int = 4

    @classmethod
    def from_chunks(cls, chunks: List[dict]) -> "EntityGate":
        known: Dict[str, Set[str]] = {"standard": set(), "model": set()}
        for c in chunks:
            e = _extract(c.get("text", "") or "")
            for fam in known:
                known[fam] |= e[fam]
        logger.info(
            f"EntityGate: 已知标准 {sorted(known['standard'])}, "
            f"已知型号 {sorted(known['model'])}"
        )
        return cls(known=known)

    def _standard_known(self, q_std: str) -> bool:
        """前缀双向匹配：query 标准号是某已知号前缀，或反之。"""
        for k in self.known["standard"]:
            lo = min(len(k), len(q_std))
            if lo >= self.min_prefix and (k.startswith(q_std) or q_std.startswith(k)):
                return True
        return False

    def check(self, query: str) -> Optional[Tuple[str, str]]:
        """
        返回 None 表示放行；否则返回 (family, offending_entity) 表示 query 提到了
        一个"域内形态但语料没有"的伪实体，应当拒答。
        """
        q = _extract(query)

        for std in q["standard"]:
            if self.known["standard"] and not self._standard_known(std):
                return ("standard", std)

        for mdl in q["model"]:
            # 型号要求精确命中；语料没有任何型号时不启用该家族（避免误伤）
            if self.known["model"] and mdl not in self.known["model"]:
                return ("model", mdl)

        return None

    def refusal_text(self, family: str, entity: str) -> str:
        fam_cn = {"standard": "标准号", "model": "型号", "subject": "主体"}.get(family, "实体")
        return (
            f"文档中未提及。（问题涉及的{fam_cn} “{entity}” 不在本文档覆盖的实体范围内，"
            f"判定为文档未覆盖的问题）"
        )


# ── 主体接地守卫（subject grounding）─────────────────────────────────────────
# reranker 拦不住的最后一类串台：query 的字段（工作温度/额定功率）与某张表**极强**
# 匹配，cross-encoder 就算主体是"特斯拉 Model 3"也会给中高分（实测 0.77，甚至高过
# 某些弱召回的域内题 0.64，单一重排阈值出现反转、拦不干净）。
#
# 原理：一个问题若真被文档覆盖，它的**主体词**必然出现在检索到的证据里。于是把 query
# 里的"主体候选词"抽出来（去掉疑问词/字段词），只要**没有任何一个**出现在 top-K 证据
# 文本中，就判定"问的是文档里没有的东西"→ 拒答。只在"全都没接地"时才拒，极保守，
# 域内题的真实主体（A100 / GB/T 2423.1 / E01）总会命中自己的证据段，不会误伤。

# 疑问词 / 字段词 / 功能词停用表：这些不是"主体"，接地判断时忽略
_STOP_PHRASES = [
    "工作温度", "额定功率", "国际标准", "标准编号", "实施日期", "故障代码",
    "年径流量", "径流量", "如何处理", "怎么处理", "是多少", "多少瓦", "哪一天",
    "对应", "编号", "型号", "标准", "范围", "多少", "如何", "怎么", "处理",
    "出现", "应该", "这份", "什么", "高度", "功率", "温度", "日期", "代码",
    "米", "瓦", "立方米", "是", "的", "或", "与", "和", "为", "在", "请问",
    "这", "那", "本", "份", "个", "条", "种", "及", "等", "就", "都", "会",
    "要", "能", "可", "有", "了", "吗", "呢", "时", "应",
]
_LATIN_STOP = {"the", "of", "is", "a", "an", "to", "for", "and", "or", "what", "how"}
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.]{1,}")
_CJK_RE = re.compile(r"[一-鿿]{2,}")


def subject_terms(query: str) -> List[str]:
    """从 query 抽"主体候选词"：长度≥2 的拉丁词 + 去掉字段/疑问词后剩下的中文名词块。"""
    terms: List[str] = []
    for m in _LATIN_RE.findall(query):
        if m.lower() not in _LATIN_STOP:
            terms.append(m)
    # 中文：先把已知字段/疑问短语抠成分隔符，剩下的连续中文块即候选主体
    masked = query
    for ph in _STOP_PHRASES:
        masked = masked.replace(ph, " ")
    for m in _CJK_RE.findall(masked):
        terms.append(m)
    # 去重保序
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def check_grounding(query: str, passages: List[str]) -> Optional[str]:
    """
    返回 None 表示主体已接地（放行）；否则返回未接地的主体词（应拒答）。
    只有当 query 的**全部**主体候选词都没出现在证据里，才判未接地——极保守。
    """
    terms = subject_terms(query)
    if not terms:
        return None  # 抽不出主体，不做判断（交给别的守卫）
    evidence = "\n".join(passages).lower()
    for t in terms:
        if t.lower() in evidence:
            return None  # 有任意主体词接地 → 放行
    return terms[0]  # 全部未接地 → 用第一个主体词说明拒答理由
