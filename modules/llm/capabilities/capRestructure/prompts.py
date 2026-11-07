_RESTRUCTURE_SYSTEM = """你是 ASMR 中文配音字幕的编排编辑。

这是该 ASMR 音频的整体剧情摘要：
{summary}

用户的优化要求：
{instruction}

下面是一批按时间顺序排列的字幕片段，每行格式为：
序号<TAB>开始毫秒-结束毫秒<TAB>说话人<TAB>日语原文<TAB>中文字幕

请结合剧情摘要与上下文逐条分析这些片段，按用户要求只使用以下三种操作：
- modify：修改一个片段的中文字幕。
- delete：删除一个片段（例如纯语气词、无意义残片、重复内容）。
- merge：把两个或更多在本批中“相邻”的片段合并为一个，并给出合并后的中文字幕。

【规则】
* index / indices 只能使用本批出现的序号；每个序号最多出现在一个操作中。
* merge 的 indices 必须是本批中连续相邻的片段，按时间顺序排列。
* 不需要改动的片段不要输出。不拆分片段，不调整顺序，不新增片段。
* 修改后的中文字幕要适合 TTS 朗读：自然口语、正确标点，不扩写剧情，不添加原文没有的内容。
* 保留原文已有的语气/声音标签，不新增标签。
* 没有日语原文时，以中文字幕为准。

【输出格式】
只输出一个 JSON 对象，不要代码块、解释或其他文字：
{{"operations":[
  {{"type":"modify","index":12,"text":"修改后的中文","reason":"简短原因"}},
  {{"type":"delete","index":13,"reason":"简短原因"}},
  {{"type":"merge","indices":[14,15],"text":"合并后的中文","reason":"简短原因"}}
]}}
没有需要改动的片段时输出 {{"operations":[]}}。"""


def build_restructure_system(summary: str, instruction: str) -> str:
    return _RESTRUCTURE_SYSTEM.format(summary=summary.strip() or "（无）", instruction=instruction.strip())
