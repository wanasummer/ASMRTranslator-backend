"""Response normalization for supported provider protocols."""
from ...error import LLMError

def parse_response(provider: str, resp_json) -> str:
    if not isinstance(resp_json, dict):
        raise LLMError(f"LLM 返回了非 JSON 对象响应: {str(resp_json)[:200]}")
    if resp_json.get("error"):
        raise LLMError(f"LLM API 返回错误: {str(resp_json['error'])[:300]}")
    try:
        if provider == "anthropic":
            content = resp_json.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                # DeepSeek's Anthropic-compatible layer may put a thinking/
                # reasoning block first; scan for the first text-bearing block.
                for block in content:
                    if isinstance(block, dict):
                        if isinstance(block.get("text"), str) and block["text"].strip():
                            return block["text"]
                        if isinstance(block.get("content"), str) and block["content"].strip():
                            return block["content"]
                    elif isinstance(block, str) and block.strip():
                        return block
            # Some proxies return an OpenAI-shaped response even when asked
            # in Anthropic format — accept it as a fallback.
            choices = resp_json.get("choices") if isinstance(resp_json, dict) else None
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {})
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
                if isinstance(c, list) and c and isinstance(c[0], dict):
                    return c[0].get("text", "")
            raise LLMError(f"Anthropic 响应中没有可解析的文本: {str(resp_json)[:300]}")
        choices = resp_json["choices"]
        content = choices[0]["message"].get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
                    return block["text"]
        finish_reason = choices[0].get("finish_reason")
        detail = "（输出 token 上限可能过低）" if finish_reason == "length" else ""
        raise LLMError(f"大模型未返回文本，finish_reason={finish_reason or '未知'}{detail}")
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise LLMError(f"unexpected LLM response shape: {e}") from e
