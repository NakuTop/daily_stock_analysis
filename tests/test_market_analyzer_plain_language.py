# -*- coding: utf-8 -*-
"""Plain-language market review prompt tests."""

import sys
from types import ModuleType
from types import SimpleNamespace

from src.core.market_profile import get_profile
from src.core.market_strategy import get_market_strategy_blueprint


search_service_stub = ModuleType("src.search_service")
search_service_stub.SearchService = object
sys.modules.setdefault("src.search_service", search_service_stub)

data_provider_stub = ModuleType("data_provider")
data_provider_stub.__path__ = []
sys.modules.setdefault("data_provider", data_provider_stub)

data_provider_base_stub = ModuleType("data_provider.base")
data_provider_base_stub.DataFetcherManager = object
sys.modules.setdefault("data_provider.base", data_provider_base_stub)

intelligence_service_stub = ModuleType("src.services.intelligence_service")
intelligence_service_stub.IntelligenceService = object
sys.modules.setdefault("src.services.intelligence_service", intelligence_service_stub)

from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview


def _make_us_analyzer() -> MarketAnalyzer:
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.config = SimpleNamespace(
        report_language="zh",
        market_review_color_scheme="green_up",
    )
    analyzer.region = "us"
    analyzer.profile = get_profile("us")
    analyzer.strategy = get_market_strategy_blueprint("us")
    analyzer.search_service = None
    analyzer.analyzer = None
    return analyzer


def _make_us_overview() -> MarketOverview:
    return MarketOverview(
        date="2026-07-01",
        indices=[
            MarketIndex(
                code="SPX",
                name="æ æ®500ææ°",
                current=7499.36,
                change_pct=0.79,
            ),
            MarketIndex(
                code="IXIC",
                name="çº³æ¯è¾¾åç»¼åææ°",
                current=26213.72,
                change_pct=1.52,
            ),
        ],
    )


def test_us_market_prompt_explains_sector_limitation_without_error_wording() -> None:
    analyzer = _make_us_analyzer()

    prompt = analyzer._build_review_prompt(_make_us_overview(), [])

    assert "ç¾è¡æ¿åæè¡ææªæ¥å¥" in prompt
    assert "ä¸æ¯æ°æ®éè¯¯" in prompt
    assert "ææ æ¿åæ¶¨è·æ°æ®" not in prompt


def test_zh_market_prompt_requests_plain_language_daily_brief() -> None:
    analyzer = _make_us_analyzer()

    prompt = analyzer._build_review_prompt(_make_us_overview(), [])

    assert "æ¯æ¥ç®æ¥" in prompt
    assert "å°ç¨ä¸ä¸æ¯è¯­" in prompt
    assert "åè¯´ç»è®º" in prompt
    assert "æå¤©éç¹çä»ä¹" in prompt
    assert "äº¤æåçåå·¥ä½å°" not in prompt


def test_us_template_review_uses_plain_sector_limitation_copy() -> None:
    analyzer = _make_us_analyzer()

    report = analyzer._generate_template_review(_make_us_overview(), [])

    assert "ç¾è¡æ¿åæè¡ææªæ¥å¥" in report
    assert "ä¸æ¯æ°æ®éè¯¯" in report
    assert "ææ æ¿åæ¶¨è·æ¦æ°æ®" not in report
