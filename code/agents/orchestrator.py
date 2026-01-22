"""
Multi-agent orchestration system for LLM-powered trading.
Implements hierarchical agent communication and decision flow.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
import numpy as np

from models.llm_wrapper import LLMWrapper, LLMConfig, PromptTemplate
from risk.risk_manager import RiskManager, RiskConfig
from prompts.prompt_registry import PromptRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enumeration of agent roles."""

    ANALYST = "analyst"
    DECISION = "decision"
    RISK = "risk"
    EXECUTION = "execution"
    EXPLAINABILITY = "explainability"


@dataclass
class AgentMessage:
    """Message passed between agents."""

    sender: AgentRole
    receiver: AgentRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class TradingContext:
    """Shared context for all agents."""

    ticker: str
    timestamp: pd.Timestamp
    current_price: float
    position: float  # Current position size
    portfolio_value: float
    features: Dict[str, float]  # Technical indicators, sentiment, etc.
    news: List[Dict[str, Any]]  # Recent news items
    history: List[Dict[str, Any]] = field(default_factory=list)  # Trade history


class BaseAgent:
    """Base class for all trading agents."""

    def __init__(
        self, role: AgentRole, llm: LLMWrapper, config: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.llm = llm
        self.config = config or {}
        self.message_history: List[AgentMessage] = []

    def receive_message(self, message: AgentMessage):
        """Receive and log message."""
        self.message_history.append(message)
        logger.debug(f"{self.role.value} received message from {message.sender.value}")

    def send_message(
        self, receiver: AgentRole, content: str, metadata: Optional[Dict] = None
    ) -> AgentMessage:
        """Create and log outgoing message."""
        message = AgentMessage(
            sender=self.role,
            receiver=receiver,
            content=content,
            metadata=metadata or {},
            timestamp=pd.Timestamp.now(),
        )
        logger.debug(f"{self.role.value} sending message to {receiver.value}")
        return message

    def process(self, context: TradingContext) -> AgentMessage:
        """Process context and generate output. To be implemented by subclasses."""
        raise NotImplementedError


class AnalystAgent(BaseAgent):
    """Analyst agent for market analysis."""

    def __init__(self, llm: LLMWrapper, config: Optional[Dict] = None):
        super().__init__(AgentRole.ANALYST, llm, config)
        self.template = PromptTemplate()

    def process(self, context: TradingContext) -> AgentMessage:
        """Analyze market conditions and generate insights."""

        # Extract relevant features
        indicators = {
            k: v
            for k, v in context.features.items()
            if k in ["rsi", "macd", "sma_20", "sma_50", "atr", "bb_width"]
        }

        # Extract sentiment
        sentiment_features = {
            k: v for k, v in context.features.items() if "sentiment" in k or "news" in k
        }

        # Extract macro features
        macro_features = {
            k: v
            for k, v in context.features.items()
            if k in ["DFF", "VIXCLS", "T10Y2Y"] or "change" in k
        }

        # Format prompt
        prompt = self.template.format_analysis_prompt(
            ticker=context.ticker,
            current_price=context.current_price,
            indicators=indicators,
            news_sentiment=sentiment_features,
            macro_context=macro_features,
        )

        # Generate analysis
        analysis = self.llm.generate(
            prompt=prompt, system_prompt=self.template.SYSTEM_PROMPTS["analyst"]
        )

        return self.send_message(
            receiver=AgentRole.DECISION,
            content=analysis,
            metadata={
                "indicators": indicators,
                "sentiment": sentiment_features,
                "macro": macro_features,
            },
        )


class DecisionAgent(BaseAgent):
    """Decision agent for trading actions."""

    def __init__(self, llm: LLMWrapper, config: Optional[Dict] = None):
        super().__init__(AgentRole.DECISION, llm, config)
        self.template = PromptTemplate()
        self.max_position = config.get("max_position", 0.2) if config else 0.2

    def process(
        self, context: TradingContext, analysis_message: AgentMessage
    ) -> Tuple[AgentMessage, Dict[str, Any]]:
        """Make trading decision based on analysis."""

        # Get risk limits
        risk_limits = {
            "max_position": self.max_position,
            "max_drawdown": self.config.get("max_drawdown", 0.15),
            "position_limit": self.config.get("position_limit", 1000),
        }

        # Format decision prompt
        prompt = self.template.format_decision_prompt(
            ticker=context.ticker,
            analysis=analysis_message.content,
            current_position=context.position,
            portfolio_value=context.portfolio_value,
            risk_limits=risk_limits,
        )

        # Generate decision
        decision_text = self.llm.generate(
            prompt=prompt, system_prompt=self.template.SYSTEM_PROMPTS["decision"]
        )

        # Parse decision
        decision = self._parse_decision(decision_text)

        return (
            self.send_message(
                receiver=AgentRole.RISK, content=decision_text, metadata=decision
            ),
            decision,
        )

    def _parse_decision(self, decision_text: str) -> Dict[str, Any]:
        import json

        try:
            j = json.loads(decision_text)
            if isinstance(j, dict) and "action" in j:
                return {**j, "raw_text": decision_text}
        except Exception:
            pass
        """Parse decision text into structured format."""
        # Simple parsing of "ACTION SIZE" format
        parts = decision_text.strip().split()

        action = "HOLD"
        size = 0.0

        for part in parts:
            part_upper = part.upper()
            if part_upper in ["BUY", "SELL", "HOLD"]:
                action = part_upper
            else:
                try:
                    size = float(part)
                except ValueError:
                    continue

        return {"action": action, "size": size, "raw_text": decision_text}


class RiskAgent(BaseAgent):
    """Risk management agent."""

    def __init__(self, llm: LLMWrapper, config: Optional[Dict] = None):
        super().__init__(AgentRole.RISK, llm, config)
        self.template = PromptTemplate()
        self.max_position = config.get("max_position", 0.2) if config else 0.2
        self.max_drawdown = config.get("max_drawdown", 0.15) if config else 0.15

    def process(
        self, context: TradingContext, decision_message: AgentMessage
    ) -> Tuple[AgentMessage, bool]:
        """
        Validate decision against risk constraints.

        Returns:
            (message, approved) tuple
        """
        decision = decision_message.metadata

        # Check position limits
        new_position = context.position
        if decision["action"] == "BUY":
            new_position += (
                decision["size"] * context.portfolio_value / context.current_price
            )
        elif decision["action"] == "SELL":
            new_position -= (
                decision["size"] * context.portfolio_value / context.current_price
            )

        position_pct = abs(
            new_position * context.current_price / context.portfolio_value
        )

        # Risk checks
        checks = {
            "position_limit": position_pct <= self.max_position,
            "position_positive": new_position >= 0,
        }

        approved = all(checks.values())

        # Generate risk assessment
        risk_prompt = f"""Assess risk for trading decision:

Decision: {decision['action']} {decision['size']}
Current Position: {context.position}
New Position: {new_position}
Position %: {position_pct*100:.1f}%
Portfolio Value: ${context.portfolio_value:.2f}

Risk Checks:
{json.dumps(checks, indent=2)}

Approved: {approved}

Provide risk assessment and recommendations.
"""

        risk_assessment = self.llm.generate(
            prompt=risk_prompt, system_prompt=self.template.SYSTEM_PROMPTS["risk"]
        )

        return (
            self.send_message(
                receiver=AgentRole.EXECUTION if approved else AgentRole.DECISION,
                content=risk_assessment,
                metadata={
                    "approved": approved,
                    "checks": checks,
                    "new_position": new_position,
                },
            ),
            approved,
        )


class ExecutionAgent(BaseAgent):
    """Trade execution agent."""

    def __init__(self, llm: LLMWrapper, config: Optional[Dict] = None):
        super().__init__(AgentRole.EXECUTION, llm, config)
        self.template = PromptTemplate()

    def process(
        self,
        context: TradingContext,
        decision_message: AgentMessage,
        risk_message: AgentMessage,
    ) -> Tuple[AgentMessage, Dict[str, Any]]:
        """Execute approved trade."""

        decision = decision_message.metadata
        new_position = risk_message.metadata["new_position"]

        # Calculate order details
        shares_to_trade = abs(new_position - context.position)

        # Simulate execution with slippage
        slippage = np.random.normal(0, 0.001)  # 10 bps average slippage
        execution_price = context.current_price * (1 + slippage)

        execution_details = {
            "action": decision["action"],
            "shares": shares_to_trade,
            "price": execution_price,
            "slippage": slippage,
            "timestamp": context.timestamp,
            "order_id": f"ORD-{context.timestamp.strftime('%Y%m%d%H%M%S')}",
        }

        # Generate execution report
        exec_prompt = f"""Report trade execution:

Action: {decision['action']}
Shares: {shares_to_trade:.0f}
Execution Price: ${execution_price:.2f}
Slippage: {slippage*100:.3f}%
Order ID: {execution_details['order_id']}

Provide brief execution summary.
"""

        exec_report = self.llm.generate(
            prompt=exec_prompt, system_prompt=self.template.SYSTEM_PROMPTS["execution"]
        )

        return (
            self.send_message(
                receiver=AgentRole.EXPLAINABILITY,
                content=exec_report,
                metadata=execution_details,
            ),
            execution_details,
        )


class ExplainabilityAgent(BaseAgent):
    """Explainability agent for decision transparency."""

    def __init__(self, llm: LLMWrapper, config: Optional[Dict] = None):
        super().__init__(AgentRole.EXPLAINABILITY, llm, config)
        self.template = PromptTemplate()

    def process(
        self,
        context: TradingContext,
        analysis_message: AgentMessage,
        decision_message: AgentMessage,
        execution_message: AgentMessage,
    ) -> AgentMessage:
        """Generate explanation for complete decision chain."""

        # Gather evidence
        evidence = {
            "technical_signals": analysis_message.metadata.get("indicators", {}),
            "sentiment": analysis_message.metadata.get("sentiment", {}),
            "macro": analysis_message.metadata.get("macro", {}),
            "decision": decision_message.metadata,
            "execution": execution_message.metadata,
        }

        # Format explanation prompt
        prompt = self.template.format_explanation_prompt(
            decision=f"{decision_message.metadata['action']} {decision_message.metadata['size']}",
            evidence=evidence,
            context=f"Trading {context.ticker} at ${context.current_price:.2f}",
        )

        # Generate explanation
        explanation = self.llm.generate(
            prompt=prompt, system_prompt=self.template.SYSTEM_PROMPTS["explainability"]
        )

        return self.send_message(
            receiver=AgentRole.ANALYST,  # Complete the cycle
            content=explanation,
            metadata=evidence,
        )


class MultiAgentOrchestrator:
    """Orchestrates multi-agent trading system."""

    def __init__(self, llm_config: LLMConfig, agent_config: Optional[Dict] = None):
        self.llm = LLMWrapper(llm_config)
        self.prompt_registry = PromptRegistry()
        self.risk_manager = RiskManager(RiskConfig())
        self.agent_config = agent_config or {}

        # Initialize agents
        self.analyst = AnalystAgent(self.llm, self.agent_config.get("analyst"))
        self.decision_maker = DecisionAgent(self.llm, self.agent_config.get("decision"))
        self.risk_manager = RiskAgent(self.llm, self.agent_config.get("risk"))
        self.executor = ExecutionAgent(self.llm, self.agent_config.get("execution"))
        self.explainer = ExplainabilityAgent(
            self.llm, self.agent_config.get("explainability")
        )

        self.message_log: List[AgentMessage] = []

    def run_cycle(self, context: TradingContext) -> Dict[str, Any]:
        """
        Run one complete agent cycle.

        Returns:
            Dictionary with decision, execution, and explanation
        """
        logger.info(f"Running agent cycle for {context.ticker} at {context.timestamp}")

        # Step 1: Analysis
        analysis_msg = self.analyst.process(context)
        self.message_log.append(analysis_msg)

        # Step 2: Decision
        decision_msg, decision = self.decision_maker.process(context, analysis_msg)
        self.message_log.append(decision_msg)

        # Step 3: Risk check
        risk_msg, approved = self.risk_manager.process(context, decision_msg)
        self.message_log.append(risk_msg)

        if not approved:
            logger.warning(f"Decision rejected by risk manager: {decision}")
            return {
                "decision": decision,
                "approved": False,
                "execution": None,
                "explanation": "Decision rejected by risk management",
                "messages": self.message_log[-3:],
            }

        # Step 4: Execution
        exec_msg, execution = self.executor.process(context, decision_msg, risk_msg)
        self.message_log.append(exec_msg)

        # Step 5: Explanation
        explain_msg = self.explainer.process(
            context, analysis_msg, decision_msg, exec_msg
        )
        self.message_log.append(explain_msg)

        return {
            "decision": decision,
            "approved": True,
            "execution": execution,
            "explanation": explain_msg.content,
            "messages": self.message_log[-5:],
        }

    def get_message_log(self) -> List[AgentMessage]:
        """Return full message log."""
        return self.message_log


if __name__ == "__main__":
    # Test orchestrator
    llm_config = LLMConfig(backend="mock")
    orchestrator = MultiAgentOrchestrator(llm_config)

    # Create test context
    context = TradingContext(
        ticker="AAPL",
        timestamp=pd.Timestamp.now(),
        current_price=150.25,
        position=0,
        portfolio_value=100000,
        features={
            "rsi": 28.5,
            "macd": 1.2,
            "sma_20": 148.5,
            "sentiment_mean": 0.6,
            "news_count": 5,
            "DFF": 5.25,
            "VIXCLS": 15.2,
        },
        news=[],
    )

    # Run cycle
    result = orchestrator.run_cycle(context)

    print("\n=== Agent Cycle Result ===")
    print(f"Decision: {result['decision']}")
    print(f"Approved: {result['approved']}")
    print(f"Execution: {result['execution']}")
    print(f"\nExplanation:\n{result['explanation']}")
