"""LLM client wrapper for Qwen API"""
import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from json_repair import loads as repair_json_loads
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for Qwen API calls through DashScope OpenAI-compatible mode"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3.5-35b-a3b",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking: bool = False
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_finish_reason = None
        self.last_response_chars = 0
        self.last_raw_content = ""
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        response_format: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate completion from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Expected format ('json' or 'text')
            
        Returns:
            Parsed response or None on error
        """
        content = ""
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_finish_reason = None
        self.last_response_chars = 0
        self.last_raw_content = ""
        try:
            request_options = {}
            if response_format == "json":
                request_options["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": self.enable_thinking},
                **request_options,
            )
            
            content = response.choices[0].message.content
            self.last_raw_content = content or ""
            usage = getattr(response, "usage", None)
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
            finish_reason = response.choices[0].finish_reason
            self.last_finish_reason = finish_reason
            self.last_response_chars = len(content or "")
            
            if response_format == "json":
                parsed = self._parse_json_content(content)
            else:
                parsed = {"content": content}
            if parsed is None:
                return None
            parsed["_usage"] = dict(self.last_usage)
            parsed["_finish_reason"] = finish_reason
            parsed["_response_chars"] = len(content or "")
            parsed["_raw_content"] = content or ""
            return parsed
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error("Raw response could not be parsed (%s chars)", len(content or ""))
            return None
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None

    @staticmethod
    def _extract_json_text(content: str) -> str:
        """Extract raw JSON text from model output, including fenced code blocks."""
        if not content:
            return content

        text = content.strip()
        fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            text = fenced_match.group(1).strip()

        return text

    @classmethod
    def _parse_json_content(cls, content: str) -> Dict[str, Any]:
        """Parse JSON content with fallbacks for fenced JSON and LaTeX backslashes."""
        text = cls._extract_json_text(content)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_object_text = cls._extract_first_json_object(text)
        if json_object_text != text:
            try:
                return json.loads(json_object_text)
            except json.JSONDecodeError:
                text = json_object_text

        escaped_text = cls._escape_invalid_json_backslashes(text)
        try:
            return json.loads(escaped_text)
        except json.JSONDecodeError as decode_error:
            try:
                repaired = repair_json_loads(text)
            except Exception:
                repaired = None
            if isinstance(repaired, dict):
                repaired["_json_repaired"] = True
                return repaired

            recovered_answer = cls._recover_final_answer(text)
            if recovered_answer is None:
                raise decode_error
            return {
                "final_answer": recovered_answer,
                "reasoning": "Response tail was malformed; final_answer was recovered from the leading field.",
                "used_bullet_ids": [],
                "used_skill_ids": [],
                "_json_repaired": True,
                "_final_answer_only_recovery": True,
            }

    @staticmethod
    def _recover_final_answer(text: str) -> Optional[str]:
        """Recover a leading final_answer when only the response tail is malformed."""
        quoted = re.search(
            r'"final_answer"\s*:\s*"(.*?)"\s*(?:,|\})',
            text,
            flags=re.DOTALL,
        )
        if quoted:
            return quoted.group(1).replace(r'\"', '"').replace(r'\\', '\\').strip()

        scalar = re.search(r'"final_answer"\s*:\s*([^,\r\n\}]+)', text)
        if scalar:
            return scalar.group(1).strip().strip('"')
        return None

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """Extract the first balanced JSON object from a response."""
        start = text.find("{")
        if start == -1:
            return text

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return text[start:]

    @staticmethod
    def _escape_invalid_json_backslashes(text: str) -> str:
        r"""Escape invalid JSON backslashes, commonly caused by LaTeX like \mathbf."""
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    @staticmethod
    def _safe_preview(content: str, limit: int = 300) -> str:
        """Create logging-safe preview text for model output."""
        if not content:
            return "<empty response>"

        preview = content[:limit]
        safe_preview = preview.encode("ascii", errors="backslashreplace").decode("ascii")
        if len(content) > limit:
            safe_preview += "..."
        return safe_preview
    
    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Generate with automatic retry on failure"""
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for attempt in range(max_retries):
            result = self.generate(prompt, **kwargs)
            for key in total_usage:
                total_usage[key] += int(self.last_usage.get(key, 0) or 0)
            if result is not None:
                self.last_usage = total_usage
                result["_usage"] = dict(total_usage)
                return result
            logger.warning(f"Retry {attempt + 1}/{max_retries}")
        self.last_usage = total_usage
        logger.error("All retry attempts failed")
        return None
