"""
Advanced Sentence Splitting Utility.
Priority: pysbd (Best accuracy) -> Robust Regex (Fallback)
"""

import re
from typing import List
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 全局缓存 pysbd 对象
_PYSBD_SEGMENTERS = {}

def _get_pysbd_segmenter(lang: str):
    """
    获取或加载 pysbd 分句器 (Lazy loading)
    """
    global _PYSBD_SEGMENTERS

    # pysbd 使用 'en' 和 'zh' 标准代码
    if lang not in _PYSBD_SEGMENTERS:
        try:
            import pysbd
            # clean=False 表示保留原始文本格式（如空格），这对后续处理很重要
            _PYSBD_SEGMENTERS[lang] = pysbd.Segmenter(language=lang, clean=False)
            logger.info(f"Loaded pysbd segmenter for language: {lang}")
        except ImportError:
            return None

    return _PYSBD_SEGMENTERS.get(lang)


def clean_text(text: str) -> str:
    """
    关键步骤：清洗原始字符串中的转义字符。
    将 \" 转换为 "，将 \\' 转换为 '
    """
    if not text:
        return ""

    # 1. 类似 Python 的 codecs.decode('unicode_escape')，但更安全的手动替换
    # 处理双反斜杠转义的引号 (常见于 JSON dump 或 SQL dump)
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")

    # 2. 去除多余的换行符（如果不需要保留段落格式）
    # text = text.replace('\n', ' ')

    # 3. 压缩多个空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def _split_by_regex(text: str, lang: str) -> List[str]:
    """
    增强版正则分句，处理了标点后跟随引号的情况。
    """
    if lang == 'zh':
        # 中文优化正则
        # 逻辑：匹配 [。！？] 后面可能跟随的 [”’] (右引号)，再后面是字符串结尾或非标点
        # 这是一个 "Split but keep delimiter" 的变体技巧
        pattern = r'([。！？?][”’"\']?)'
    else:
        # 英文优化正则 (简单的 fallback，处理不了 Mr. 等复杂情况，复杂情况请安装 pysbd)
        pattern = r'([.!?][”’"\']?)'

    # 使用捕获组 () 进行 split，会保留分隔符
    parts = re.split(pattern, text)

    sentences = []
    current_sent = ""

    # re.split 结果形式为: [文本, 分隔符, 文本, 分隔符, ...]
    for i in range(0, len(parts) - 1, 2):
        content = parts[i]
        separator = parts[i+1]

        # 拼接 文本 + 标点
        sent = content + separator
        if sent.strip():
            sentences.append(sent.strip())

    # 处理最后一段（通常是没有标点结尾的文本）
    if len(parts) % 2 != 0:
        last_part = parts[-1].strip()
        if last_part:
            sentences.append(last_part)

    return sentences

def split_sentences(text: str, lang: str = "auto") -> List[str]:
    """
    最准确的分句函数。

    Args:
        text: 输入文本
        lang: 'zh', 'en', 或 'auto'

    Returns:
        分句后的列表
    """

    if not text or not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    # 1. 自动语言检测
    if lang == "auto":
        # 只要包含中文字符，就优先按中文逻辑处理
        if re.search(r'[\u4e00-\u9fff]', text):
            lang = "zh"
        else:
            lang = "en"

    # 2. 尝试使用 pysbd (业界公认最准确的规则分句)
    segmenter = _get_pysbd_segmenter(lang)

    if segmenter:
        try:
            return segmenter.segment(text)
        except Exception as e:
            logger.warning(f"pysbd failed: {e}, falling back to regex.")

    # 3. 降级使用增强版正则
    # 当 pysbd 未安装或报错时使用
    return _split_by_regex(text, lang)