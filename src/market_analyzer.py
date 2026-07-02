# -*- coding: utf-8 -*-
"""
===================================
å¤§çå¤çåææ¨¡å
===================================

èè´£ï¼
1. è·åå¤§çææ°æ°æ®ï¼ä¸è¯ãæ·±è¯ãåä¸æ¿ï¼
2. æç´¢å¸åºæ°é»å½¢æå¤çææ¥
3. ä½¿ç¨å¤§æ¨¡åçææ¯æ¥å¤§çå¤çæ¥å
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from inspect import getattr_static
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from src.llm.backend_registry import (
    resolve_generation_backend_id,
    resolve_generation_fallback_backend_id,
)
from src.llm.generation_backend import GenerationError
from src.schemas.market_light import MarketLightSnapshot
from src.services.run_diagnostics import record_llm_run, record_llm_run_started
from src.services.intelligence_service import IntelligenceService
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*ä¸ã(?:çé¢æ»è§|å¸åºæ»ç»)",
    "index_commentary": r"###\s*äºã(?:ææ°ç»æ|ææ°ç¹è¯|ä¸»è¦ææ°)",
    "sector_highlights": r"###\s*ä¸ã(?:æ¿åä¸»çº¿|ç­ç¹è§£è¯»|æ¿åè¡¨ç°)",
    "funds_sentiment": r"###\s*åã(?:èµéä¸æç»ª|èµéå¨å)",
    "news_catalysts": r"###\s*äºã(?:æ¶æ¯å¬å|åå¸å±æ)",
}


@dataclass
class MarketIndex:
    """å¤§çææ°æ°æ®"""
    code: str                    # ææ°ä»£ç 
    name: str                    # ææ°åç§°
    current: float = 0.0         # å½åç¹ä½
    change: float = 0.0          # æ¶¨è·ç¹æ°
    change_pct: float = 0.0      # æ¶¨è·å¹(%)
    open: float = 0.0            # å¼çç¹ä½
    high: float = 0.0            # æé«ç¹ä½
    low: float = 0.0             # æä½ç¹ä½
    prev_close: float = 0.0      # æ¨æ¶ç¹ä½
    volume: float = 0.0          # æäº¤éï¼æï¼
    amount: float = 0.0          # æäº¤é¢ï¼åï¼
    amplitude: float = 0.0       # æ¯å¹(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """å¸åºæ¦è§æ°æ®"""
    date: str                           # æ¥æ
    indices: List[MarketIndex] = field(default_factory=list)  # ä¸»è¦ææ°
    up_count: int = 0                   # ä¸æ¶¨å®¶æ°
    down_count: int = 0                 # ä¸è·å®¶æ°
    flat_count: int = 0                 # å¹³çå®¶æ°
    limit_up_count: int = 0             # æ¶¨åå®¶æ°
    limit_down_count: int = 0           # è·åå®¶æ°
    total_amount: float = 0.0           # ä¸¤å¸æäº¤é¢ï¼äº¿åï¼
    # north_flow: float = 0.0           # ååèµéåæµå¥ï¼äº¿åï¼- å·²åºå¼ï¼æ¥å£ä¸å¯ç¨
    
    # æ¿åæ¶¨å¹æ¦
    top_sectors: List[Dict] = field(default_factory=list)     # æ¶¨å¹å5æ¿å
    bottom_sectors: List[Dict] = field(default_factory=list)  # è·å¹å5æ¿å
    top_concepts: List[Dict] = field(default_factory=list)    # æ¶¨å¹å5æ¦å¿µ
    bottom_concepts: List[Dict] = field(default_factory=list) # è·å¹å5æ¦å¿µ


@dataclass
class MarketLightReviewResult:
    """Internal market-review parts built from one overview fetch."""

    overview: MarketOverview
    report: str
    market_light_snapshot: Dict[str, Any]
    structured_payload: Dict[str, Any] = field(default_factory=dict)


class MarketAnalyzer:
    """
    å¤§çå¤çåæå¨
    
    åè½ï¼
    1. è·åå¤§çææ°å®æ¶è¡æ
    2. è·åå¸åºæ¶¨è·ç»è®¡
    3. è·åæ¿åæ¶¨è·æ¦
    4. æç´¢å¸åºæ°é»
    5. çæå¤§çå¤çæ¥å
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
        config: Optional[Any] = None,
    ):
        """
        åå§åå¤§çåæå¨

        Args:
            search_service: æç´¢æå¡å®ä¾
            analyzer: AIåæå¨å®ä¾ï¼ç¨äºè°ç¨LLMï¼
            region: å¸åºåºå cn=Aè¡ us=ç¾è¡
            config: æ¬æ¬¡å¤çä½¿ç¨çéç½®ï¼æªä¼ æ¶è¯»åå¨å±éç½®
        """
        self.config = config or get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def _log_context(self) -> str:
        return f"component=market_review region={self.region}"

    def _get_review_language(self) -> str:
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_template_review_language(self) -> str:
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market" if review_language == "en" else "ç¾è¡å¸åº"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "æ¸¯è¡å¸åº"
        if review_language == "en":
            return "A-share market"
        return "Aè¡å¸åº"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "åäº¿ç¾å"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "åäº¿æ¸¯å"
        return "CNY 100m" if self._get_review_language() == "en" else "äº¿"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    def _get_index_change_arrow(self, change_pct: float) -> str:
        if change_pct == 0:
            return "âª"
        color_scheme = getattr(getattr(self, "config", None), "market_review_color_scheme", "green_up")
        if color_scheme == "red_up":
            return "ð´" if change_pct > 0 else "ð¢"
        return "ð¢" if change_pct > 0 else "ð´"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {"us": "US Market Recap", "hk": "HK Market Recap"}
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} å¤§çå¤ç"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return "Analyze the key moves in the S&P 500, Nasdaq, Dow, and other major indices."
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if self.region == "us" and self._get_review_language() == "zh":
            return """## ç¾è¡å¸åºä¸æ®µå¼å¤çç­ç¥
èç¦ææ°è¶å¿ãå®è§åäºä¸æ¿åè½®å¨ï¼ç»åºæ¬¡æ¥é£æ§ä¸ä»ä½æ¡æ¶ã

### ç­ç¥åå
- åçæ æ®500ãçº³æ¯è¾¾åãéç¼æ¯æ¯å¦ååï¼ç¡®è®¤ä¸»çº¿æ¯å¦ä¸è´ã
- ç»åå®è§ä¸æµå¨æ§ææ ï¼è¯å«é£é©åå¥½æ¯ä¿®å¤è¿æ¯è½¬å¼±ã
- å°å¤çè¾åºæ å°ä¸ºâè¿æ»/åè¡¡/é²å®âå¨ä½å»ºè®®ï¼å¹¶ç»åºæç¡®è§¦åå¤±ææ¡ä»¶ã

### åæç»´åº¦
- è¶å¿ç»æï¼æç¡®å¸åºå¤äºä¸å²ãéè¡è¿æ¯é²å®è½¬åï¼å¤æ­æ¯å¦å­å¨å³é®æ¯æä½èç¦»ã
- èµéä¸æç»ªï¼åºåå®è§æ¿ç­ãè´§å¸é¢ä¸æ³¢å¨çå¯¹æçé£é©çå½±åã
- ä¸»é¢çº¿ç´¢ï¼è¯å«æç»­æ§æå¼ºçä¸»é¢ä¸æ¿åè½®å¨æ¯å¦å½¢æå¯äº¤æä¸»çº¿ã

### è¡å¨æ¡æ¶
- è¿æ»ï¼ä¸»æ¿åèå¨ä¸è¡ä¸éè½/é£é©ä½åæ­¥æ¹åã
- åè¡¡ï¼ææ°ååæéè½æªææ¾æ¾å¤§ï¼ä»ä½ä¿å®æ§è¡ã
- é²å®ï¼çªç ´å¤±å®ä¸æ³¢å¨çæ¬åæ¶ï¼ä¼ååç å¹¶ä¿çåå¼¹å¯äº¤ææ§ã"""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if self.region == "us" and review_language == "zh":
            return """### å­ãç­ç¥æ¡æ¶
- **è¶å¿ç»æ**ï¼å¤æ­å¸åºå¨è¿æ»ãéè¡ä¸é²å®ä¸­çç¶ææ¯å¦ä¸è´ã
- **èµéä¸æç»ª**ï¼ç»åæ³¢å¨çãå®½åº¦åä¸»é¢è½®å¨è¯ä¼°é£é©åå¥½ã
- **ä¸»é¢ä¸»çº¿**ï¼è¯å«å¯å»¶ç»­åå¯æ¾å¤§çè¡ä¸ä¸»çº¿ä¸é²å®çº¿ç´¢ã
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "å¼ºå¿ä¸æ¶¨",
                "mild_up": "å°å¹ä¸æ¶¨",
                "mild_down": "å°å¹ä¸è·",
                "strong_down": "ææ¾ä¸è·",
                "range": "éè¡æ´ç",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        è·åå¸åºæ¦è§æ°æ®
        
        Returns:
            MarketOverview: å¸åºæ¦è§æ°æ®å¯¹è±¡
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. è·åä¸»è¦ææ°è¡æï¼æ region åæ¢ A è¡/ç¾è¡ï¼
        overview.indices = self._get_main_indices()

        # 2. è·åæ¶¨è·ç»è®¡ï¼A è¡æï¼ç¾è¡æ ç­ææ°æ®ï¼
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. è·åæ¿åæ¶¨è·æ¦ï¼A è¡æï¼ç¾è¡ææ ï¼
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
            self._get_concept_rankings(overview)
        
        # 4. è·åååèµéï¼å¯éï¼
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """è·åä¸»è¦ææ°å®æ¶è¡æ"""
        indices = []

        try:
            logger.info("[å¤§ç] %s action=get_main_indices status=start", self._log_context())

            # ä½¿ç¨ DataFetcherManager è·åææ°è¡æï¼æ region åæ¢ï¼
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[å¤§ç] %s action=get_main_indices status=empty", self._log_context())
            else:
                logger.info(
                    "[å¤§ç] %s action=get_main_indices status=success count=%d",
                    self._log_context(),
                    len(indices),
                )

        except Exception as e:
            logger.error("[å¤§ç] %s action=get_main_indices status=failed error=%s", self._log_context(), e)

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """è·åå¸åºæ¶¨è·ç»è®¡"""
        try:
            logger.info("[å¤§ç] %s action=get_market_stats status=start", self._log_context())

            stats = self.data_manager.get_market_stats(purpose=f"market_review:{self.region}")

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(
                    "[å¤§ç] %s action=get_market_stats status=success up=%s down=%s flat=%s "
                    "limit_up=%s limit_down=%s amount=%.0fäº¿",
                    self._log_context(),
                    overview.up_count,
                    overview.down_count,
                    overview.flat_count,
                    overview.limit_up_count,
                    overview.limit_down_count,
                    overview.total_amount,
                )
            else:
                logger.warning("[å¤§ç] %s action=get_market_stats status=empty", self._log_context())

        except Exception as e:
            logger.error("[å¤§ç] %s action=get_market_stats status=failed error=%s", self._log_context(), e)

    def _get_sector_rankings(self, overview: MarketOverview):
        """è·åæ¿åæ¶¨è·æ¦"""
        try:
            logger.info("[å¤§ç] %s action=get_sector_rankings status=start", self._log_context())

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(
                    "[å¤§ç] %s action=get_sector_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s['name'] for s in overview.top_sectors],
                    [s['name'] for s in overview.bottom_sectors],
                )
            else:
                logger.warning("[å¤§ç] %s action=get_sector_rankings status=empty", self._log_context())

        except Exception as e:
            logger.error("[å¤§ç] %s action=get_sector_rankings status=failed error=%s", self._log_context(), e)

    def _get_concept_rankings(self, overview: MarketOverview):
        """è·åæ¦å¿µ/é¢ææ¶¨è·æ¦ï¼fail-openï¼ã"""
        try:
            logger.info("[å¤§ç] %s action=get_concept_rankings status=start", self._log_context())

            top_concepts, bottom_concepts = self.data_manager.get_concept_rankings(5)

            if top_concepts or bottom_concepts:
                overview.top_concepts = top_concepts
                overview.bottom_concepts = bottom_concepts

                logger.info(
                    "[å¤§ç] %s action=get_concept_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s.get('name') for s in overview.top_concepts],
                    [s.get('name') for s in overview.bottom_concepts],
                )
            else:
                logger.warning("[å¤§ç] %s action=get_concept_rankings status=empty", self._log_context())

        except Exception as e:
            logger.warning("[å¤§ç] %s action=get_concept_rankings status=failed error=%s", self._log_context(), e)
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """è·åååèµéæµå¥"""
    #     try:
    #         logger.info("[å¤§ç] è·åååèµé...")
    #         
    #         # è·åååèµéæ°æ®
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="åä¸")
    #         
    #         if df is not None and not df.empty:
    #             # åææ°ä¸æ¡æ°æ®
    #             latest = df.iloc[-1]
    #             if 'å½æ¥åæµå¥' in df.columns:
    #                 overview.north_flow = float(latest['å½æ¥åæµå¥']) / 1e8  # è½¬ä¸ºäº¿å
    #             elif 'åæµå¥' in df.columns:
    #                 overview.north_flow = float(latest['åæµå¥']) / 1e8
    #                 
    #             logger.info(f"[å¤§ç] ååèµéåæµå¥: {overview.north_flow:.2f}äº¿")
    #             
    #     except Exception as e:
    #         logger.warning(f"[å¤§ç] è·åååèµéå¤±è´¥: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        æç´¢å¸åºæ°é»
        
        Returns:
            æ°é»åè¡¨
        """
        if not self.search_service:
            logger.warning(
                "[å¤§ç] %s action=search_market_news status=skipped reason=no_search_service",
                self._log_context(),
            )
            return []
        
        all_news = []

        # æ region ä½¿ç¨ä¸åçæ°é»æç´¢è¯
        search_queries = self.profile.news_queries
        review_language = self._get_review_language()
        market_names = {
            "cn": "å¤§ç" if review_language == "zh" else "A-share market",
            "us": "ç¾è¡å¸åº" if review_language == "zh" else "US market",
            "hk": "æ¸¯è¡å¸åº" if review_language == "zh" else "HK market",
        }
        
        try:
            logger.info("[å¤§ç] %s action=search_market_news status=start", self._log_context())
            
            # æ ¹æ® region è®¾ç½®æç´¢ä¸ä¸æåç§°ï¼é¿åç¾è¡æç´¢è¢«è§£è¯»ä¸º A è¡è¯­å¢
            market_name = market_names.get(self.region, "å¤§ç")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(
                        "[å¤§ç] %s action=search_market_news status=query_success count=%d",
                        self._log_context(),
                        len(response.results),
                    )
            
            logger.info(
                "[å¤§ç] %s action=search_market_news status=success count=%d",
                self._log_context(),
                len(all_news),
            )
            
        except Exception as e:
            logger.error("[å¤§ç] %s action=search_market_news status=failed error=%s", self._log_context(), e)
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        ä½¿ç¨å¤§æ¨¡åçæå¤§çå¤çæ¥å
        
        Args:
            overview: å¸åºæ¦è§æ°æ®
            news: å¸åºæ°é»åè¡¨ (SearchResult å¯¹è±¡åè¡¨)
            
        Returns:
            å¤§çå¤çæ¥åææ¬
        """
        backend_error = self._get_analyzer_generation_backend_config_error()
        if backend_error is not None:
            logger.error(
                "[å¤§ç] %s action=generate_review status=failed error_type=%s error=%s",
                self._log_context(),
                type(backend_error).__name__,
                backend_error,
            )
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                error_type=type(backend_error).__name__,
                error_message=backend_error,
            )
            raise backend_error

        if not self.analyzer or not self.analyzer.is_available():
            logger.warning(
                "[å¤§ç] %s action=generate_review status=fallback_template reason=no_analyzer",
                self._log_context(),
            )
            return self._generate_template_review(overview, news)

        # æå»º Prompt
        prompt = self._build_review_prompt(overview, news)

        logger.info("[å¤§ç] %s action=generate_review status=start", self._log_context())
        # Use the public generate_text() entry point - never access private analyzer attributes.
        llm_started_at = time.perf_counter()
        try:
            record_llm_run_started(
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
            )
            review = self.analyzer.generate_text(prompt, max_tokens=8192, temperature=0.7)
        except Exception as exc:
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
                error_type=type(exc).__name__,
                error_message=exc,
            )
            raise

        record_llm_run(
            success=bool(review),
            provider="litellm",
            model=getattr(self.config, "litellm_model", None),
            call_type="market_review",
            duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
            error_type=None if review else "EmptyResponse",
            error_message=None if review else "empty market review response",
        )

        if review:
            logger.info(
                "[å¤§ç] %s action=generate_review status=success length=%d",
                self._log_context(),
                len(review),
            )
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview, news)

        logger.warning(
            "[å¤§ç] %s action=generate_review status=fallback_template reason=empty_llm_response",
            self._log_context(),
        )
        return self._generate_template_review(overview, news)

    def _get_analyzer_generation_backend_config_error(self) -> Optional[GenerationError]:
        """Return analyzer backend config errors without relying on dynamic mock attributes."""
        if self.analyzer is None:
            try:
                resolve_generation_backend_id(self.config)
                resolve_generation_fallback_backend_id(self.config)
            except GenerationError as exc:
                return exc
            return None
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_config_error", missing) is missing:
            return None
        method = getattr(self.analyzer, "get_generation_backend_config_error", None)
        if not callable(method):
            return None
        error = method()
        return error if isinstance(error, GenerationError) else None

    def build_market_review_payload(
        self,
        overview: MarketOverview,
        news: List,
        report: str,
        market_light_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the structured market-review contract consumed by API, Web, and notifications."""
        language = self._get_review_language()
        sections = self._split_report_sections(report)
        title = self._extract_report_title(report) or self._get_review_title(overview.date).lstrip("# ").strip()
        light = market_light_snapshot or self.build_market_light_snapshot(overview)
        breadth_dimensions = None
        if isinstance(light, dict):
            dimensions = light.get("dimensions")
            if isinstance(dimensions, dict):
                breadth_dimensions = dimensions.get("breadth")

        breadth_supported = bool(self.profile.has_market_stats)
        if breadth_supported and isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
            breadth_supported = bool(breadth_dimensions.get("available"))

        has_breadth_data = False
        if breadth_supported:
            if isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
                has_breadth_data = bool(breadth_dimensions.get("available"))
            else:
                breadth_available = overview.up_count + overview.down_count + overview.flat_count > 0
                limit_available = overview.limit_up_count + overview.limit_down_count > 0
                has_breadth_data = bool(breadth_available or limit_available)

        payload = {
            "version": 1,
            "kind": "market_review",
            "region": self.region,
            "language": language,
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "date": overview.date,
            "market_scope": self._get_market_scope_name(language),
            "market_light": light,
            "indices": [idx.to_dict() for idx in overview.indices],
            "sectors": {
                "top": list(overview.top_sectors or []),
                "bottom": list(overview.bottom_sectors or []),
            },
            "concepts": {
                "top": list(overview.top_concepts or []),
                "bottom": list(overview.bottom_concepts or []),
            },
            "news": [self._normalize_news_item(item) for item in (news or [])[:8]],
            "sections": sections,
            "markdown_report": report,
        }

        if has_breadth_data:
            payload["breadth"] = {
                "up_count": overview.up_count,
                "down_count": overview.down_count,
                "flat_count": overview.flat_count,
                "limit_up_count": overview.limit_up_count,
                "limit_down_count": overview.limit_down_count,
                "total_amount": overview.total_amount,
                "turnover_unit": self._get_turnover_unit_label(),
            }

        return payload

    @staticmethod
    def _extract_report_title(report: str) -> str:
        for line in (report or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    @classmethod
    def _split_report_sections(cls, report: str) -> List[Dict[str, str]]:
        text = (report or "").strip()
        if not text:
            return []
        matches = list(re.finditer(r"^(#{2,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
        if not matches:
            return [{"key": "full_review", "title": "Review", "markdown": text}]

        sections: List[Dict[str, str]] = []
        first_match = matches[0]
        starts_with_report_title = first_match.start() == 0 and first_match.group(1) == "##"
        content_start_index = 1 if starts_with_report_title else 0
        intro_start = first_match.end() if starts_with_report_title else 0
        intro_end = (
            matches[1].start()
            if starts_with_report_title and len(matches) > 1
            else (len(text) if starts_with_report_title else matches[0].start())
        )
        intro = text[intro_start:intro_end].strip()
        if intro:
            sections.append({"key": "overview", "title": "Overview", "markdown": intro})

        for index, match in enumerate(matches[content_start_index:], start=content_start_index):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            markdown = text[start:end].strip()
            if not markdown:
                continue
            key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", title).strip("_").lower()
            sections.append({
                "key": key or f"section_{index + 1}",
                "title": title,
                "markdown": markdown,
            })
        return sections

    @classmethod
    def _normalize_news_item(cls, item: Any) -> Dict[str, str]:
        return {
            "title": cls._compact_news_text(cls._get_news_field(item, "title"), limit=120),
            "snippet": cls._compact_news_text(cls._get_news_field(item, "snippet"), limit=260),
            "source": cls._compact_news_text(cls._get_news_field(item, "source"), limit=80),
            "published_date": cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=40),
            "url": cls._compact_news_text(cls._get_news_field(item, "url"), limit=240),
        }
    
    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        if stats_block:
            review = self._insert_after_section(
                review,
                patterns["market_summary"],
                stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review,
                patterns["index_commentary"],
                indices_block,
            )

        if sector_block:
            review = self._insert_after_section(
                review,
                patterns["sector_highlights"],
                sector_block,
            )

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading â append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        if self._get_review_language() == "en":
            light = self.build_market_light_snapshot(overview)
            return "\n".join(
                [
                    f"- **Market Signal**: {light['score']}/100 "
                    f"({light['temperature_label']}, {light['label']})",
                    f"- **Drivers**: {'; '.join(light['reasons'])}",
                    f"- **Guidance**: {light['guidance']}",
                    "",
                    f"- **Breadth**: Advancers {overview.up_count} / Decliners {overview.down_count} / "
                    f"Flat {overview.flat_count}; "
                    f"Limit-up {overview.limit_up_count} / Limit-down {overview.limit_down_count}; "
                    f"Turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})",
                ]
            )
        light = self.build_market_light_snapshot(overview)
        score, label = light["score"], light["temperature_label"]
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        lines = [
            f"- **çé¢ä¿¡å·**ï¼{score}/100ï¼{label}ï¼{light['label']}ï¼",
            f"- **ä¿¡å·ä¾æ®**ï¼{'ï¼'.join(light['reasons'])}",
            f"- **æä½å»ºè®®**ï¼{light['guidance']}",
            "",
            "| ææ  | æ°å¼ | è§å¯ |",
            "|------|------|------|",
            f"| ä¸æ¶¨/ä¸è·/å¹³ç | {overview.up_count} / {overview.down_count} / {overview.flat_count} | ä¸æ¶¨å æ¯(ä¸å«å¹³ç) {up_ratio:.1%} |",
            f"| æ¶¨å/è·å | {overview.limit_up_count} / {overview.limit_down_count} | æ¶¨è·åå·® {limit_spread:+d} |",
            f"| ä¸¤å¸æäº¤é¢ | {overview.total_amount:.0f} äº¿ | {self._describe_turnover(overview.total_amount)} |",
        ]
        return "\n".join(lines)

    def build_market_light_snapshot(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build a deterministic market-light snapshot from structured breadth data."""
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        temperature_label = str(scores["temperature_label"])
        if score >= 60:
            status = "green"
        elif score >= 40:
            status = "yellow"
        else:
            status = "red"

        if self._get_review_language() == "en":
            label_map = {
                "green": "risk-on",
                "yellow": "balanced",
                "red": "risk-off",
            }
            guidance_map = {
                "green": "Risk appetite is acceptable; focus on leading themes and position discipline.",
                "yellow": "Signals are mixed; keep position sizing moderate and wait for confirmation.",
                "red": "Risk is elevated; prioritize drawdown control and avoid chasing weak rebounds.",
            }
            reasons = self._build_market_light_reasons_en(overview, score)
        else:
            label_map = {
                "green": "å¯è¿æ»",
                "yellow": "éè§å¯",
                "red": "åé²å®",
            }
            guidance_map = {
                "green": "é£é©åå¥½å°å¯ï¼å³æ³¨ä¸»çº¿å»¶ç»­ä¸ä»ä½çºªå¾ã",
                "yellow": "ä¿¡å·ååï¼æ§å¶ä»ä½å¹¶ç­å¾éä»·ç¡®è®¤ã",
                "red": "é£é©åé«ï¼ä¼åæ§å¶åæ¤ï¼é¿åè¿½é«å¼±åå¼¹ã",
            }
            reasons = self._build_market_light_reasons_zh(overview, score)

        snapshot = MarketLightSnapshot(
            region=self.region,
            trade_date=overview.date,
            status=status,
            label=label_map[status],
            score=score,
            temperature_label=temperature_label,
            reasons=reasons,
            guidance=guidance_map[status],
            dimensions=scores["dimensions"],
            data_quality=str(scores["data_quality"]),
        )
        return snapshot.model_dump()

    def _build_market_light_reasons_zh(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"ä¸æ¶¨å®¶æ°å æ¯ {up_ratio:.0%}ï¼èµé±æåºæ©æ£")
            elif up_ratio <= 0.4:
                reasons.append(f"ä¸æ¶¨å®¶æ°å æ¯ {up_ratio:.0%}ï¼äºé±æåºè¾å¼º")
            else:
                reasons.append(f"ä¸æ¶¨å®¶æ°å æ¯ {up_ratio:.0%}ï¼å¸åºåå")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"ä¸»è¦ææ°å¹³åæ¶¨è·å¹ {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"æ¶¨è·åå·® {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"æäº¤é¢ {overview.total_amount:.0f} äº¿ï¼{self._describe_turnover(overview.total_amount)}")
        if not reasons:
            reasons.append("ç»æåæ¶¨è·æ°æ®æéï¼æå¯ç¨è¡æç»¼åå¤æ­")
        return reasons[:4]

    def _build_market_light_reasons_en(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is expanding")
            elif up_ratio <= 0.4:
                reasons.append(f"advancers ratio {up_ratio:.0%}, downside pressure dominates")
            else:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is mixed")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"average major-index change {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"limit-up/down spread {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})")
        if not reasons:
            reasons.append("limited structured breadth data; using available market inputs")
        return reasons[:4]

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """æå»ºææ°è¡æè¡¨æ ¼"""
        if not overview.indices:
            return ""
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Last | Change % | Open | High | Low | Amplitude | Turnover ({self._get_turnover_unit_label()}) |",
                "|-------|------|----------|------|------|-----|-----------|-----------------|",
            ]
        else:
            lines = [
                "| ææ° | ææ° | æ¶¨è·å¹ | å¼ç | æé« | æä½ | æ¯å¹ | æäº¤é¢(äº¿) |",
                "|------|------|--------|------|------|------|------|-----------|",
            ]
        for idx in overview.indices:
            arrow = self._get_index_change_arrow(idx.change_pct)
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | "
                f"{self._format_optional_number(idx.open)} | {self._format_optional_number(idx.high)} | "
                f"{self._format_optional_number(idx.low)} | {self._format_optional_pct(idx.amplitude)} | {amount_str} |"
            )
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build industry and concept ranking blocks."""
        if (
            not overview.top_sectors
            and not overview.bottom_sectors
            and not overview.top_concepts
            and not overview.bottom_concepts
        ):
            return ""
        lines = []
        language = self._get_review_language()

        def append_ranking(title: str, name_label: str, rows: List[Dict]) -> None:
            if not rows:
                return
            if lines:
                lines.append("")
            lines.extend([
                title,
                f"| {'Rank' if language == 'en' else 'æå'} | {name_label} | {'Change' if language == 'en' else 'æ¶¨è·å¹'} |",
                "|------|------|--------|",
            ])
            for rank, item in enumerate(rows[:5], 1):
                lines.append(
                    f"| {rank} | {item.get('name', '-')} | {self._format_signed_pct(item.get('change_pct'))} |"
                )

        if language == "en":
            append_ranking("#### Leading Industry Sectors", "Sector", overview.top_sectors)
            append_ranking("#### Lagging Industry Sectors", "Sector", overview.bottom_sectors)
            append_ranking("#### Leading Concept Themes", "Concept", overview.top_concepts)
            append_ranking("#### Lagging Concept Themes", "Concept", overview.bottom_concepts)
        else:
            append_ranking("#### è¡ä¸æ¿åé¢æ¶¨ Top 5", "è¡ä¸æ¿å", overview.top_sectors)
            append_ranking("#### è¡ä¸æ¿åé¢è· Top 5", "è¡ä¸æ¿å", overview.bottom_sectors)
            append_ranking("#### æ¦å¿µæ¿åé¢æ¶¨ Top 5", "æ¦å¿µæ¿å", overview.top_concepts)
            append_ranking("#### æ¦å¿µæ¿åé¢è· Top 5", "æ¦å¿µæ¿å", overview.bottom_concepts)
        return "\n".join(lines)

    def _build_news_block(self, news: List) -> str:
        """Build a compact source-aware news catalyst list for the rendered report."""
        if not news:
            return ""
        language = self._get_review_language()
        if language == "en":
            lines = [
                "#### News Catalysts",
            ]
        else:
            lines = [
                "#### è¿ä¸æ¥å¸åºçº¿ç´¢",
            ]

        for idx, item in enumerate(news[:5], 1):
            lines.append(self._format_news_catalyst_line(idx, item, language=language))
        return "\n".join(lines)

    @staticmethod
    def _get_news_field(item: Any, field: str) -> str:
        if hasattr(item, field):
            value = getattr(item, field, "") or ""
        elif isinstance(item, dict):
            value = item.get(field, "") or ""
        else:
            value = ""
        return str(value).strip()

    @classmethod
    def _format_news_catalyst_line(cls, idx: int, item: Any, *, language: str = "zh") -> str:
        fallback_title = "Untitled catalyst" if language == "en" else "æªå½åçº¿ç´¢"
        title = cls._compact_news_text(cls._get_news_field(item, "title"), limit=90) or fallback_title
        source = cls._compact_news_text(cls._get_news_field(item, "source"), limit=40)
        date_text = cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=24)
        url = cls._compact_news_text(cls._get_news_field(item, "url"), limit=0)
        title_text = cls._escape_markdown_link_label(title)
        if url:
            title_text = f"[{title_text}]({url})"
        meta_parts = [part for part in (source, date_text) if part]
        if language == "en":
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
        else:
            meta = f"ï¼{' / '.join(meta_parts)}ï¼" if meta_parts else ""
        return f"- {idx}. {title_text}{meta}"

    @staticmethod
    def _compact_news_text(value: str, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @classmethod
    def _format_ranking_summary(cls, rows: List[Dict], limit: int = 3) -> str:
        parts = []
        for item in (rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            parts.append(f"{name}({cls._format_signed_pct(item.get('change_pct'))})")
        return ", ".join(parts)

    @staticmethod
    def _escape_markdown_link_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "é«æ´»è·åº¦"
        if total_amount >= 9000:
            return "ä¸­ç­æ´»è·"
        if total_amount > 0:
            return "ç¼©éè§æ"
        return "ææ æ°æ®"

    def _build_market_light_scores(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build the canonical Market Light scores used by reports and alerts."""

        participants = overview.up_count + overview.down_count
        breadth_available = bool(self.profile.has_market_stats and participants > 0)
        breadth_score = 50
        if breadth_available:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        index_available = bool(overview.indices and index_changes)
        index_score = 50
        if index_available:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_available = bool(self.profile.has_market_stats and limit_total > 0)
        limit_score = 50
        if limit_available:
            limit_score = int(overview.limit_up_count / limit_total * 100)

        dimensions = {
            "breadth": {"score": breadth_score, "available": breadth_available},
            "index": {"score": index_score, "available": index_available},
            "limit": {"score": limit_score, "available": limit_available},
        }

        if not index_available:
            data_quality = "unavailable"
        elif all(dimension["available"] for dimension in dimensions.values()):
            data_quality = "ok"
        else:
            data_quality = "partial"

        score = int(round(breadth_score * 0.45 + index_score * 0.35 + limit_score * 0.20))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "å¼ºå¿"
            elif score >= 55:
                label = "åæ"
            elif score >= 40:
                label = "éè¡"
            else:
                label = "åå¼±"
        return {
            "score": score,
            "temperature_label": label,
            "dimensions": dimensions,
            "data_quality": data_quality,
        }

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        label = str(scores["temperature_label"])
        return score, label

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """æå»ºå¤çæ¥å Prompt"""
        review_language = self._get_review_language()

        # ææ°è¡æä¿¡æ¯ï¼ç®æ´æ ¼å¼ï¼ä¸ç¨emojiï¼
        indices_text = ""
        for idx in overview.indices:
            direction = "â" if idx.change_pct > 0 else "â" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # æ¿åä¿¡æ¯
        top_sectors_text = self._format_ranking_summary(overview.top_sectors)
        bottom_sectors_text = self._format_ranking_summary(overview.bottom_sectors)
        top_concepts_text = self._format_ranking_summary(overview.top_concepts)
        bottom_concepts_text = self._format_ranking_summary(overview.bottom_concepts)
        
        # æ°é»ä¿¡æ¯ - æ¯æ SearchResult å¯¹è±¡æå­å¸
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # å¼å®¹ SearchResult å¯¹è±¡åå­å¸
            title = self._compact_news_text(self._get_news_field(n, "title"), limit=90)
            snippet = self._compact_news_text(self._get_news_field(n, "snippet"), limit=220)
            source = self._compact_news_text(self._get_news_field(n, "source"), limit=60)
            published_date = self._compact_news_text(self._get_news_field(n, "published_date"), limit=30)
            url = self._compact_news_text(self._get_news_field(n, "url"), limit=180)
            meta_parts = [part for part in (source, published_date) if part]
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
            url_line = f"\n   URL: {url}" if url else ""
            news_text += f"{i}. {title}{meta}\n   {snippet or '-'}{url_line}\n"
        
        # æ region ç»è£å¸åºæ¦åµä¸æ¿ååºåï¼ç¾è¡æ æ¶¨è·å®¶æ°ãæ¿åæè¡æ°æ®ï¼
        stats_block = ""
        sector_block = ""
        if review_language == "en":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""
            else:
                stats_block = (
                    "## Market Breadth\n"
                    "(Advance/decline breadth is not connected for this market yet. "
                    "This is expected, not a data error. Explain market tone from indices and news.)"
                )

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector / Theme Performance
Industry leading: {top_sectors_text if top_sectors_text else "N/A"}
Industry lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}
Concept leading: {top_concepts_text if top_concepts_text else "N/A"}
Concept lagging: {bottom_concepts_text if bottom_concepts_text else "N/A"}"""
            else:
                sector_block = (
                    "## Sector / Theme Performance\n"
                    "(Sector ranking data is not connected for this market yet. "
                    "This is expected, not a data error. Use index moves and news to explain likely themes, "
                    "and say clearly when a theme is an inference.)"
                )
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## å¸åºæ¦åµ
- ä¸æ¶¨: {overview.up_count} å®¶ | ä¸è·: {overview.down_count} å®¶ | å¹³ç: {overview.flat_count} å®¶
- æ¶¨å: {overview.limit_up_count} å®¶ | è·å: {overview.limit_down_count} å®¶
- ä¸¤å¸æäº¤é¢: {overview.total_amount:.0f} äº¿å"""
            else:
                stats_block = (
                    "## å¸åºæ¦åµ\n"
                    "ï¼è¯¥å¸åºçæ¶¨è·å®¶æ°ç»è®¡ææªæ¥å¥ï¼ä¸æ¯æ°æ®éè¯¯ï¼è¯·ä¸»è¦æ ¹æ®ææ°æ¶¨è·åæ°é»çº¿ç´¢å¤æ­ä»å¤©å¸åºå·æãï¼"
                )

            if self.profile.has_sector_rankings:
                sector_block = f"""## æ¿åè¡¨ç°
è¡ä¸é¢æ¶¨: {top_sectors_text if top_sectors_text else "ææ æ°æ®"}
è¡ä¸é¢è·: {bottom_sectors_text if bottom_sectors_text else "ææ æ°æ®"}
æ¦å¿µé¢æ¶¨: {top_concepts_text if top_concepts_text else "ææ æ°æ®"}
æ¦å¿µé¢è·: {bottom_concepts_text if bottom_concepts_text else "ææ æ°æ®"}"""
            else:
                sector_block = (
                    "## æ¿åè¡¨ç°\n"
                    "ï¼ç¾è¡æ¿åæè¡ææªæ¥å¥ï¼ä¸æ¯æ°æ®éè¯¯ï¼è¯·æ ¹æ®ææ°å¼ºå¼±ãæ°é»çº¿ç´¢åå¸¸è§ä¸»é¢æ¨æ­ä¸»çº¿ï¼å¹¶æç¡®åªäºå¤æ­æ¯æ¨æ­ãï¼"
                )

        data_no_indices_hint = (
            "æ³¨æï¼ç±äºè¡ææ°æ®è·åå¤±è´¥ï¼è¯·ä¸»è¦æ ¹æ®ãå¸åºæ°é»ãè¿è¡å®æ§åæåæ»ç»ï¼ä¸è¦ç¼é å·ä½çææ°ç¹ä½ã"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            news_placeholder = news_text if news_text else "No relevant news"
        else:
            indices_placeholder = indices_text if indices_text else "ææ ææ°æ°æ®ï¼æ¥å£å¼å¸¸ï¼"
            news_placeholder = news_text if news_text else "ææ ç¸å³æ°é»"

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You write plain-language daily market briefs. Please produce a concise, easy-to-read recap based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in English
- Write for readers who want to understand the day quickly, not for professional traders
- Use simple words; when a market term is useful, explain it briefly
- Start with the answer, then explain why, what could go wrong, and what to watch next
- Avoid over-precise conclusions when the source data says a field is not connected yet

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

### 1. Market Summary
(2-3 simple sentences: was the market strong, weak, or mixed; who led; why it mattered.)

### 2. Index Commentary
({self._get_index_hint()} Explain the moves in everyday language.)

### 3. Fund Flows
(Explain what the available activity data suggests. If breadth/turnover is not connected, say so naturally and avoid treating it as a problem.)

### 4. Sector Highlights
(Explain the main themes in simple language. If sector rankings are not connected, use index moves and news as clues and label the conclusion as an inference.)

### 5. Outlook
(State what to watch next in plain, practical terms.)

### 6. Risk Alerts
(List the main things that could make the view wrong.)

### 7. Strategy Plan
(Provide a plain stance, a simple position-sizing guideline, one invalidation trigger, and end with âFor reference only, not investment advice.â)

---

Output the report content directly, no extra commentary.
"""

        return f"""ä½ æ¯ä¸ä½ä¼æè¡å¸ä¿¡æ¯è®²æ¸æ¥çæ¯æ¥ç®æ¥å©æï¼è¯·æ ¹æ®ä»¥ä¸æ°æ®çæä¸ä»½éä¿ææç{self._get_market_scope_name('zh')}å¤§çå¤çã

ãéè¦ãè¾åºè¦æ±ï¼
- å¿é¡»è¾åºçº¯ Markdown ææ¬æ ¼å¼
- ç¦æ­¢è¾åº JSON æ ¼å¼
- ç¦æ­¢è¾åºä»£ç å
- emoji ä»å¨æ é¢å¤å°éä½¿ç¨ï¼æ¯ä¸ªæ é¢æå¤1ä¸ªï¼
- æ¥åè¦åæ¯æ¥ç®æ¥ï¼åè¯´ç»è®ºï¼åè¯´åå ãé£é©ãæå¤©éç¹çä»ä¹
- å°ç¨ä¸ä¸æ¯è¯­ï¼å¿é¡»ä½¿ç¨æ¶ï¼ç¨ä¸å¥è¯è§£éææ®éäººè½æçææ
- æ¯æ®µå°½éç­ï¼ä¼ååç­âä»å¤©åçäºä»ä¹ãä¸ºä»ä¹ãææå¤©è¯¥çåªéâ
- ä¸è¦éå¤ååºå·²ç±ç³»ç»æ³¨å¥çè¡¨æ ¼æ°æ®ï¼æ­£æè´è´£è§£éè¡¨æ ¼èåçå«ä¹
- æ°æ®ææªæ¥å¥æ¶ï¼ä¸è¦åå¾åæ¥éï¼è¦è¯´æè¿æ¯ç³»ç»ææªæ¥å¥çæ­£å¸¸éå¶ï¼å¹¶æ¢ç¨å¯ç¨çº¿ç´¢åæ

---

# ä»æ¥å¸åºæ°æ®

## æ¥æ
{overview.date}

## ä¸»è¦ææ°
{indices_placeholder}

{stats_block}

{sector_block}

## å¸åºæ°é»
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# è¾åºæ ¼å¼æ¨¡æ¿ï¼è¯·ä¸¥æ ¼ææ­¤æ ¼å¼è¾åºï¼

## {overview.date} å¤§çå¤ç

> ä¸å¥è¯ç»åºä»æ¥å¸åºç¶æãæ ¸å¿çç¾åææ¥ä¼åè§å¯æ¹åã

### ä¸ãçé¢æ»è§
ï¼2-3å¥è¯åè¯´ç»è®ºï¼ä»å¤©å¸åºåå¼ºãåå¼±è¿æ¯ååï¼è°å¸¦å¨ï¼æ®éäººè¯¥å¦ä½çè§£ï¼

### äºãææ°ç»æ
ï¼{self._get_index_hint()}ï¼ç¨éä¿è¯­è¨è¯´æåªä¸ªææ°æ´å¼ºãåªä¸ªæåè¿ï¼ä»¥åå³é®è§å¯ä½ç½®ï¼

### ä¸ãæ¿åä¸»çº¿
ï¼è¯´æä»å¤©å¤§å®¶ä¸»è¦å¨ä¹°ä»ä¹æ¹åãä¸ºä»ä¹ä¹°ï¼è¥æ²¡ææ¿åæè¡ï¼å°±æ ¹æ®ææ°åæ°é»æ¨æ­ï¼å¹¶æç¡®è¿æ¯æ¨æ­ï¼

### åãèµéä¸æç»ª
ï¼ç¨âé±æ´æ¿æè¿æ»è¿æ¯è§æâçæ¹å¼è§£éï¼è¥å¸åºå®½åº¦ææäº¤é¢ææªæ¥å¥ï¼è¦èªç¶è¯´æï¼ä¸è¦å½æå©ç©ºï¼

### äºãæ¶æ¯å¬å
ï¼æè¿ä¸æ¥æ°é»ç¿»è¯ææ®éè¯ï¼åªæ¡æ¶æ¯å¯è½æ¨å¨å¸åºï¼åªæ¡å¯è½å¸¦æ¥æ°å¨ï¼

### å­ãææ¥äº¤æè®¡å
ï¼ç´æ¥åæå¤©éç¹çä»ä¹ãåªäºæ¹åå¯å³æ³¨ãåªäºæåµè¦å°å¿ï¼ä»¥åä¸ä¸ªçéçä¿¡å·ï¼

### ä¸ãé£é©æç¤º
ï¼ååºæå®¹æè®©å¤æ­å¤±æçé£é©ï¼æåè¡¥åâå»ºè®®ä»ä¾åèï¼ä¸æææèµå»ºè®®âãï¼

---

è¯·ç´æ¥è¾åºå¤çæ¥ååå®¹ï¼ä¸è¦è¾åºå¶ä»è¯´ææå­ã
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """ä½¿ç¨æ¨¡æ¿çæå¤çæ¥åï¼æ å¤§æ¨¡åæ¶çå¤éæ¹æ¡ï¼"""
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # æ ¹æ® mood_index_code æ¥æ¾å¯¹åºææ°
        # cn: mood_code="000001"ï¼idx.code å¯è½ä¸º "sh000001"ï¼ä»¥ mood_code ç»å°¾ï¼
        # us: mood_code="SPX"ï¼idx.code ç´æ¥ä¸º "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)
        
        # ææ°è¡æï¼ç®æ´æ ¼å¼ï¼
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "â" if idx.change_pct > 0 else "â" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # æ¿åä¿¡æ¯
        separator = ", " if template_language == "en" else "ã"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])
        top_concept_text = separator.join([s['name'] for s in overview.top_concepts[:3]])
        bottom_concept_text = separator.join([s['name'] for s in overview.bottom_concepts[:3]])

        if template_language == "en":
            stats_section = ""
            if self.profile.has_market_stats:
                stats_section = f"""
### 3. Breadth & Liquidity
| Metric | Value |
|--------|-------|
| Advancers | {overview.up_count} |
| Decliners | {overview.down_count} |
| Limit-up | {overview.limit_up_count} |
| Limit-down | {overview.limit_down_count} |
| Turnover ({self._get_turnover_unit_label()}) | {overview.total_amount:.0f} |
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text or top_concept_text or bottom_concept_text):
                sector_section = f"""
### 4. Sector / Theme Highlights
- **Industry Leaders**: {top_text or "N/A"}
- **Industry Laggards**: {bottom_text or "N/A"}
- **Concept Leaders**: {top_concept_text or "N/A"}
- **Concept Laggards**: {bottom_concept_text or "N/A"}
"""
            market_names = {"us": "US Market Recap", "hk": "HK Market Recap"}
            market_name = market_names.get(self.region, "A-share Market Recap")
            report = f"""## {overview.date} {market_name}

### 1. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 2. Major Indices
{indices_text or "- No index data available"}
{stats_section}
{sector_section}
### 5. Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

{self._get_strategy_markdown_block(template_language)}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return report

        market_labels = {"cn": "Aè¡", "us": "ç¾è¡", "hk": "æ¸¯è¡"}
        market_label = market_labels.get(self.region, "Aè¡")
        dashboard_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        if not dashboard_block:
            dashboard_block = (
                f"- {market_label}æ¶¨è·å®¶æ°ç»è®¡ææªæ¥å¥ï¼ä¸æ¯æ°æ®éè¯¯ï¼"
                "ä¸é¢æ ¹æ®ææ°è¡¨ç°åæ°é»çº¿ç´¢å¤æ­å¸åºå·æã"
            )
        if not sector_block:
            if self.region == "us":
                sector_block = (
                    "- ç¾è¡æ¿åæè¡ææªæ¥å¥ï¼ä¸æ¯æ°æ®éè¯¯ï¼"
                    "ä¸»çº¿å¤æ­ä¼æ ¹æ®çº³æãæ æ®ãéæå¼ºå¼±åæ°é»çº¿ç´¢åæ¨æ­ã"
                )
            else:
                sector_block = (
                    f"- {market_label}æ¿åæè¡ææªæ¥å¥ï¼ä¸æ¯æ°æ®éè¯¯ï¼"
                    "ä¸»çº¿å¤æ­ä¼æ ¹æ®ææ°åæ°é»çº¿ç´¢åæ¨æ­ã"
                )
        return f"""## {overview.date} å¤§çå¤ç

> ä»æ¥{market_label}å¸åºæ´ä½åç°**{market_mood}**ï¼åçä¸»è¦ææ°æ¯å¦è¿è½ç¨³ä½ï¼åçç­ç¹æ¯å¦ç»§ç»­æ©æ£ã

### ä¸ãçé¢æ»è§
{dashboard_block}

### äºãææ°ç»æ
{indices_block or indices_text or "ææ ææ°æ°æ®ã"}

### ä¸ãæ¿åä¸»çº¿
{sector_block}

### åãèµéä¸æç»ª
- å¦ææ¶¨å¿ä¸»è¦éä¸­å¨å°æ°å¤§çè¡ï¼è¯´æå¸åºè¿ä¸ç®å¨é¢è½¬å¼ºï¼å¦ææ´å¤è¡ç¥¨è·æ¶¨ï¼æç»ªä¼æ´å¥åº·ã

### äºãæ¶æ¯å¬å
- ææç¡®æ°é»æ¨å¨çæ¹åæ´å®¹æå»¶ç»­ï¼å¦æåªæ¯åæ¥ä¸æ¶¨ï¼æå¤©è¦çæ¯å¦è¿è½æ¾éè·è¿ã

{self._get_strategy_markdown_block(template_language)}

### å­ãææ¥éç¹çä»ä¹
- ä¸»è¦ææ°è½å¦å®ä½ä»å¤©çå³é®ä½ç½®ã
- ç­ç¹æ¯å¦ä»å°æ°é¾å¤´æ©æ£å°æ´å¤è¡ç¥¨ã
- è¥ææ°å²é«åè½ï¼ä¼åéä½å¯¹ç­çº¿è¿½æ¶¨çä¿¡å¿ã

### ä¸ãé£é©æç¤º
- å¸åºæé£é©ï¼æèµéè°¨æãä»¥ä¸æ°æ®ä»ä¾åèï¼ä¸æææèµå»ºè®®ã

---
*å¤çæ¶é´: {datetime.now().strftime('%H:%M')}*
"""
    
    def _run_daily_review_parts(self) -> MarketLightReviewResult:
        """Run market review once and keep report/snapshot on the same overview."""
        logger.info("========== å¼å§å¤§çå¤çåæ ==========")

        # 1. è·åå¸åºæ¦è§
        overview = self.get_market_overview()

        # 2. æç´¢å¸åºæ°é»
        news = self.search_market_news()
        news = self._merge_persisted_market_intelligence(news)

        # 3. çæå¤çæ¥å
        report = self.generate_market_review(overview, news)
        snapshot = self.build_market_light_snapshot(overview)
        structured_payload = self.build_market_review_payload(
            overview,
            news,
            report,
            snapshot,
        )

        logger.info("========== å¤§çå¤çåæå®æ ==========")

        return MarketLightReviewResult(
            overview=overview,
            report=report,
            market_light_snapshot=snapshot,
            structured_payload=structured_payload,
        )

    def _merge_persisted_market_intelligence(self, news: List) -> List:
        """Merge local persisted market intelligence and search news with bounded prompt/payload slot preservation."""
        search_news = list(news or [])
        merged_local = []
        seen_urls = {
            self._get_news_field(item, "url")
            for item in search_news
            if self._get_news_field(item, "url")
        }
        try:
            service = IntelligenceService()
            payload = service.list_items(
                scope_type="market",
                market=self.region,
                published_days=max(1, int(self.config.get_effective_news_window_days() or 1)),
                page=1,
                page_size=6,
            )
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_local.append({
                    "title": item.get("title") or "æªå½åèµè®¯",
                    "snippet": item.get("summary") or "",
                    "source": item.get("source") or item.get("source_name") or "local-intel",
                    "published_date": item.get("published_at") or "",
                    "url": "" if url.startswith("no-url:intel:") else url,
                })
        except Exception as exc:
            logger.debug("[å¤§ç] %s action=load_local_intelligence status=failed error=%s", self._log_context(), exc)
        merged_news = []
        merged_local_index = 0
        merged_search_index = 0
        while merged_local_index < len(merged_local) or merged_search_index < len(search_news):
            if merged_local_index < len(merged_local):
                merged_news.append(merged_local[merged_local_index])
                merged_local_index += 1
            if merged_search_index < len(search_news):
                merged_news.append(search_news[merged_search_index])
                merged_search_index += 1
        return merged_news

    def run_daily_review(self) -> str:
        """
        æ§è¡æ¯æ¥å¤§çå¤çæµç¨

        Returns:
            å¤çæ¥åææ¬
        """
        return self.run_daily_review_with_snapshot().report

    def run_daily_review_with_snapshot(self) -> MarketLightReviewResult:
        """Run daily review and return the report plus its structured Market Light snapshot."""
        return self._run_daily_review_parts()


# æµè¯å¥å£
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # æµè¯è·åå¸åºæ¦è§
    overview = analyzer.get_market_overview()
    print(f"\n=== å¸åºæ¦è§ ===")
    print(f"æ¥æ: {overview.date}")
    print(f"ææ°æ°é: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"ä¸æ¶¨: {overview.up_count} | ä¸è·: {overview.down_count}")
    print(f"æäº¤é¢: {overview.total_amount:.0f}äº¿")
    
    # æµè¯çææ¨¡æ¿æ¥å
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== å¤çæ¥å ===")
    print(report)
