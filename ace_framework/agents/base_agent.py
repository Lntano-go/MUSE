"""Base agent class for all ACE agents"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..utils.llm_client import LLMClient
from ..config import ModelConfig


class BaseAgent(ABC):
    """Abstract base class for ACE agents"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm_client = LLMClient(
            api_key=config.api_key,
            model_name=config.name,
            base_url=config.base_url,
            enable_thinking=config.enable_thinking
        )
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_finish_reason = None
    
    @abstractmethod
    def _build_prompt(self, **kwargs) -> str:
        """Build prompt for the agent"""
        pass
    
    @abstractmethod
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        pass
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute agent logic
        
        Returns:
            Parsed response or error dict
        """
        prompt = self._build_prompt(**kwargs)
        
        response = self.llm_client.generate_with_retry(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        if response is None:
            self.last_usage = dict(self.llm_client.last_usage)
            self.last_finish_reason = self.llm_client.last_finish_reason or "request_failed_or_unparseable"
            parsed = self._get_error_response()
            parsed["_usage"] = self.last_usage
            parsed["_finish_reason"] = self.last_finish_reason
            parsed["_response_chars"] = self.llm_client.last_response_chars
            parsed["_raw_content"] = self.llm_client.last_raw_content
            return parsed

        self.last_usage = response.get("_usage", dict(self.llm_client.last_usage))
        self.last_finish_reason = response.get("_finish_reason")
        parsed = self._parse_response(response)
        parsed["_usage"] = self.last_usage
        parsed["_finish_reason"] = self.last_finish_reason
        parsed["_response_chars"] = response.get("_response_chars", 0)
        parsed["_raw_content"] = response.get("_raw_content", "")
        self.last_usage = parsed["_usage"]
        return parsed
    
    @abstractmethod
    def _get_error_response(self) -> Dict[str, Any]:
        """Get default error response"""
        pass
