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
                name="标普500指数",
                current=7499.36,
                change_pct=0.79,
            ),
            MarketIndex(
                code="IXIC",
                name="纳斯达克综合指数",
                current=26213.72,
                change_pct=1.52,
            ),
        ],
    )


def test_us_market_prompt_explains_sector_limitation_without_error_wording() -> None:
    analyzer = _make_us_analyzer()

    prompt = analyzer._build_review_prompt(_make_us_overview(), [])

    assert "美股板块排行暂未接入" in prompt
    assert "不是数据错误" in prompt
    assert "暂无板块涨跌数据" not in prompt


def test_zh_market_prompt_requests_plain_language_daily_brief() -> None:
    analyzer = _make_us_analyzer()

    prompt = analyzer._build_review_prompt(_make_us_overview(), [])

    assert "每日简报" in prompt
    assert "少用专业术语" in prompt
    assert "先说结论" in prompt
    assert "明天重点看什么" in prompt
    assert "交易员盘后工作台" not in prompt


def test_us_template_review_uses_plain_sector_limitation_copy() -> None:
    analyzer = _make_us_analyzer()

    report = analyzer._generate_template_review(_make_us_overview(), [])

    assert "美股板块排行暂未接入" in report
    assert "不是数据错误" in report
    assert "暂无板块涨跌榜数据" not in report
