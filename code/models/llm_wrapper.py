"""
LLM wrapper for trading agents with support for multiple backends.
Implements prompt templates and context management.
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing LLM libraries
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available")

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""

    backend: str = "local"  # "openai", "anthropic", "local", "mock"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    use_cache: bool = True


class LLMWrapper:
    """Unified wrapper for multiple LLM backends."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.backend = config.backend

        if self.backend == "openai":
            self._init_openai()
        elif self.backend == "anthropic":
            self._init_anthropic()
        elif self.backend == "local":
            self._init_local()
        elif self.backend == "mock":
            self._init_mock()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, falling back to mock")
            self.backend = "mock"
            self._init_mock()
            return

        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI backend with model {self.config.model_name}")

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, falling back to mock")
            self.backend = "mock"
            self._init_mock()
            return

        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(
            f"Initialized Anthropic backend with model {self.config.model_name}"
        )

    def _init_local(self):
        """Initialize local HuggingFace model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, falling back to mock")
            self.backend = "mock"
            self._init_mock()
            return

        try:
            model_name = self.config.model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )
            logger.info(f"Initialized local model {model_name}")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            logger.warning("Falling back to mock backend")
            self.backend = "mock"
            self._init_mock()

    def _init_mock(self):
        """Initialize mock backend for testing."""
        logger.info("Initialized mock LLM backend")
        self.mock_responses = {
            "analyze": "Based on the technical indicators, RSI shows oversold conditions at 28, MACD crossover signals bullish momentum, and strong volume confirms accumulation. Sentiment is moderately positive.",
            "decide": "BUY 0.3",
            "risk": "Current position size: 0.3. Portfolio volatility: 12%. Recommendation: Maintain position with stop-loss at -3%.",
            "execute": "EXECUTED: BUY 100 shares at $150.25. Order ID: ORD-12345",
            "explain": "The BUY decision is based on three converging signals: (1) Technical oversold bounce, (2) positive earnings sentiment, (3) macro support from declining rates.",
        }

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.backend == "openai":
            return self._generate_openai(prompt, system_prompt, **kwargs)
        elif self.backend == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, **kwargs)
        elif self.backend == "local":
            return self._generate_local(prompt, system_prompt, **kwargs)
        elif self.backend == "mock":
            return self._generate_mock(prompt, system_prompt, **kwargs)

    def _generate_openai(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )

        return response.choices[0].message.content

    def _generate_anthropic(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate using Anthropic API."""
        system = (
            system_prompt or "You are a helpful AI assistant for financial trading."
        )

        message = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def _generate_local(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate using local HuggingFace model."""
        # Format prompt with system message
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\n\nAssistant:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            do_sample=True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("Assistant:")[-1].strip()

        return response

    def _generate_mock(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate mock response for testing."""
        # Simple keyword matching for mock responses
        prompt_lower = prompt.lower()

        if "analyze" in prompt_lower or "technical" in prompt_lower:
            return self.mock_responses["analyze"]
        elif "decide" in prompt_lower or "action" in prompt_lower:
            return self.mock_responses["decide"]
        elif "risk" in prompt_lower or "position" in prompt_lower:
            return self.mock_responses["risk"]
        elif "execute" in prompt_lower or "order" in prompt_lower:
            return self.mock_responses["execute"]
        elif "explain" in prompt_lower or "rationale" in prompt_lower:
            return self.mock_responses["explain"]
        else:
            return "I understand your request. Based on current market conditions, the recommended action is to HOLD with careful monitoring of key support levels."


class PromptTemplate:
    """Template manager for agent prompts."""

    # System prompts for different agents
    SYSTEM_PROMPTS = {
        "analyst": """You are an expert financial analyst specializing in technical and fundamental analysis.
Your role is to analyze market data, indicators, news sentiment, and macroeconomic factors to provide
comprehensive market insights. Be precise, data-driven, and cite specific indicators in your analysis.""",
        "decision": """You are a quantitative portfolio manager making trading decisions.
Based on analysis from multiple sources, you must decide on trading actions (BUY/SELL/HOLD) and position sizes.
Your decisions must be rational, risk-aware, and explicitly justified. Output format: ACTION POSITION_SIZE""",
        "risk": """You are a risk management specialist ensuring portfolio safety and compliance.
Monitor position sizes, exposure limits, drawdown levels, and volatility. Flag any concerns and recommend
adjustments to maintain risk within acceptable bounds.""",
        "execution": """You are a trade execution specialist responsible for optimal order placement.
Consider market impact, slippage, timing, and execution algorithms. Report execution details clearly.""",
        "explainability": """You are an AI transparency officer explaining trading decisions to human stakeholders.
Provide clear, jargon-free explanations linking decisions to specific evidence. Highlight key factors and
confidence levels. Structure: Decision → Evidence → Reasoning → Risks.""",
    }

    @staticmethod
    def format_analysis_prompt(
        ticker: str,
        current_price: float,
        indicators: Dict[str, float],
        news_sentiment: Dict[str, Any],
        macro_context: Dict[str, float],
    ) -> str:
        """Format prompt for analyst agent."""
        prompt = f"""Analyze {ticker} for trading decision:

Current Price: ${current_price:.2f}

Technical Indicators:
{json.dumps(indicators, indent=2)}

Recent News Sentiment:
{json.dumps(news_sentiment, indent=2)}

Macroeconomic Context:
{json.dumps(macro_context, indent=2)}

Provide comprehensive analysis covering:
1. Technical signal strength and direction
2. Sentiment implications
3. Macro tailwinds/headwinds
4. Overall outlook (bullish/bearish/neutral)
"""
        return prompt

    @staticmethod
    def format_decision_prompt(
        ticker: str,
        analysis: str,
        current_position: float,
        portfolio_value: float,
        risk_limits: Dict[str, float],
    ) -> str:
        """Format prompt for decision agent."""
        prompt = f"""Make trading decision for {ticker}:

Analysis Summary:
{analysis}

Current Position: {current_position:.2f} shares
Portfolio Value: ${portfolio_value:.2f}
Risk Limits: {json.dumps(risk_limits, indent=2)}

Decide on action (BUY/SELL/HOLD) and position size adjustment.
Output format: ACTION SIZE
Example: BUY 0.25 (increase allocation by 0.25 of portfolio)
"""
        return prompt

    @staticmethod
    def format_explanation_prompt(
        decision: str, evidence: Dict[str, Any], context: str
    ) -> str:
        """Format prompt for explainability agent."""
        prompt = f"""Explain the following trading decision to stakeholders:

Decision: {decision}

Supporting Evidence:
{json.dumps(evidence, indent=2)}

Market Context:
{context}

Provide clear explanation covering:
1. What decision was made and why
2. Key evidence supporting the decision
3. Reasoning chain from evidence to decision
4. Potential risks and uncertainties
5. Confidence level

Use clear language suitable for non-technical audiences.
"""
        return prompt


if __name__ == "__main__":
    # Test LLM wrapper with mock backend
    config = LLMConfig(backend="mock")
    llm = LLMWrapper(config)

    # Test analyst prompt
    template = PromptTemplate()
    prompt = template.format_analysis_prompt(
        ticker="AAPL",
        current_price=150.25,
        indicators={"rsi": 28, "macd": 1.2, "sma_20": 148.5},
        news_sentiment={"mean": 0.6, "count": 5},
        macro_context={"fed_funds": 5.25, "vix": 15.2},
    )

    response = llm.generate(
        prompt=prompt, system_prompt=template.SYSTEM_PROMPTS["analyst"]
    )

    print("Prompt:")
    print(prompt)
    print("\nResponse:")
    print(response)
