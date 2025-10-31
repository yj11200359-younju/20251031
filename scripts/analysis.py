#!/usr/bin/env python3
"""데이터 요약 및 효율적 투자 프런티어(Efficient Frontier) 계산 스크립트.

외부 패키지에 의존하지 않고 Python 표준 라이브러리만으로 CSV를 파싱하고
통계량과 효율적 투자 프런티어를 계산한 뒤, 분석 결과와 SVG 그래프를 생성한다.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATA_PATH = Path("temp.csv")
OUTPUT_DIR = Path("outputs")
SITE_ASSETS_DIR = Path("site/assets")


@dataclass
class TickerSeries:
    ticker: str
    dates: List[datetime]
    prices: List[float]

    def returns(self) -> List[float]:
        """단순 일간 수익률 (전일 대비) 계산."""
        values: List[float] = []
        for prev, curr in zip(self.prices, self.prices[1:]):
            if prev == 0:
                values.append(0.0)
            else:
                values.append((curr / prev) - 1.0)
        return values


@dataclass
class PortfolioPoint:
    weights: Dict[str, float]
    exp_return: float
    risk: float  # 표준편차


def load_close_prices(path: Path) -> Dict[str, TickerSeries]:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        try:
            level1 = next(reader)
            level2 = next(reader)
            next(reader)  # "Date" 행
        except StopIteration as exc:  # pragma: no cover - 방어 로직
            raise ValueError("CSV 파일에 필요한 헤더 행이 부족합니다.") from exc

        # Close 열 위치 식별
        close_indices: Dict[str, int] = {}
        for idx in range(1, len(level1)):
            header_type = (level1[idx] or "").strip().lower()
            ticker = (level2[idx] or "").strip()
            if header_type == "close" and ticker:
                close_indices[ticker] = idx

        if not close_indices:
            raise ValueError("'Close' 열을 찾을 수 없습니다.")

        rows: Dict[str, List[Tuple[datetime, float]]] = {
            ticker: [] for ticker in close_indices
        }
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                date = datetime.strptime(row[0].strip(), "%Y-%m-%d")
            except ValueError:
                # 날짜가 비어 있거나 잘못된 행은 건너뛴다.
                continue
            for ticker, idx in close_indices.items():
                value = row[idx].strip() if idx < len(row) else ""
                if value:
                    try:
                        price = float(value)
                    except ValueError:
                        continue
                    rows[ticker].append((date, price))

    series: Dict[str, TickerSeries] = {}
    for ticker, entries in rows.items():
        entries.sort(key=lambda item: item[0])
        dates = [dt for dt, _ in entries]
        prices = [price for _, price in entries]
        if len(prices) < 2:
            continue
        series[ticker] = TickerSeries(ticker=ticker, dates=dates, prices=prices)

    if not series:
        raise ValueError("분석할 시계열 데이터를 찾을 수 없습니다.")
    return series


def basic_statistics(series: Dict[str, TickerSeries]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for ticker, data in series.items():
        price_stats = {
            "count": len(data.prices),
            "mean": statistics.mean(data.prices),
            "median": statistics.median(data.prices),
            "min": min(data.prices),
            "max": max(data.prices),
            "stdev": statistics.stdev(data.prices) if len(data.prices) > 1 else 0.0,
        }
        returns = data.returns()
        if returns:
            return_stats = {
                "mean": statistics.mean(returns),
                "stdev": statistics.stdev(returns) if len(returns) > 1 else 0.0,
                "min": min(returns),
                "max": max(returns),
            }
        else:
            return_stats = {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
        stats[ticker] = {
            "price": price_stats,
            "return": return_stats,
        }
    return stats


def covariance_matrix(series: Dict[str, TickerSeries]) -> Dict[str, Dict[str, float]]:
    tickers = list(series.keys())
    returns = {ticker: series[ticker].returns() for ticker in tickers}
    # 공통 기간을 맞추기 위해 가장 짧은 길이를 사용
    min_len = min(len(values) for values in returns.values())
    trimmed = {
        ticker: values[-min_len:]
        for ticker, values in returns.items()
    }
    cov_matrix: Dict[str, Dict[str, float]] = {
        t1: {t2: 0.0 for t2 in tickers}
        for t1 in tickers
    }
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers[i:], start=i):
            values1 = trimmed[t1]
            values2 = trimmed[t2]
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)
            covariance = sum(
                (a - mean1) * (b - mean2) for a, b in zip(values1, values2)
            ) / (min_len - 1)
            cov_matrix[t1][t2] = covariance
            cov_matrix[t2][t1] = covariance
    return cov_matrix


def expected_returns(series: Dict[str, TickerSeries]) -> Dict[str, float]:
    exp: Dict[str, float] = {}
    for ticker, data in series.items():
        rets = data.returns()
        exp[ticker] = statistics.mean(rets) if rets else 0.0
    return exp


def generate_portfolios(
    tickers: List[str],
    exp_returns: Dict[str, float],
    cov_matrix: Dict[str, Dict[str, float]],
    step: float = 0.02,
) -> List[PortfolioPoint]:
    points: List[PortfolioPoint] = []
    steps = int(1 / step)
    for i in range(steps + 1):
        w1 = i * step
        for j in range(steps + 1 - i):
            w2 = j * step
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            weights = {tickers[0]: w1, tickers[1]: w2, tickers[2]: max(w3, 0.0)}
            exp_return = sum(weights[t] * exp_returns[t] for t in tickers)
            variance = 0.0
            for a in tickers:
                for b in tickers:
                    variance += weights[a] * weights[b] * cov_matrix[a][b]
            risk = math.sqrt(max(variance, 0.0))
            points.append(PortfolioPoint(weights=weights, exp_return=exp_return, risk=risk))
    return points


def efficient_frontier(points: Iterable[PortfolioPoint]) -> List[PortfolioPoint]:
    sorted_points = sorted(points, key=lambda p: (p.exp_return, p.risk))
    frontier: List[PortfolioPoint] = []
    min_risk = math.inf
    for point in sorted_points:
        if point.risk < min_risk - 1e-12:
            frontier.append(point)
            min_risk = point.risk
    return frontier


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def save_portfolio_csv(points: List[PortfolioPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tickers = list(next(iter(points)).weights.keys()) if points else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        header = ["expected_return", "risk"] + [f"weight_{t}" for t in tickers]
        writer.writerow(header)
        for p in points:
            row = [p.exp_return, p.risk] + [p.weights[t] for t in tickers]
            writer.writerow(row)


def render_svg(
    portfolios: List[PortfolioPoint],
    frontier: List[PortfolioPoint],
    singles: Dict[str, Tuple[float, float]],
    path: Path,
) -> None:
    width, height = 900, 600
    margin = 70
    all_points = portfolios + frontier
    if not all_points:
        raise ValueError("그래프를 그릴 포트폴리오 데이터가 없습니다.")
    min_risk = min(p.risk for p in all_points)
    max_risk = max(p.risk for p in all_points)
    min_ret = min(p.exp_return for p in all_points)
    max_ret = max(p.exp_return for p in all_points)

    def scale_x(value: float) -> float:
        if math.isclose(max_risk, min_risk):
            return width / 2
        return margin + (value - min_risk) / (max_risk - min_risk) * (width - 2 * margin)

    def scale_y(value: float) -> float:
        if math.isclose(max_ret, min_ret):
            return height / 2
        # SVG Y축은 아래로 증가하므로 역으로 매핑한다.
        return height - margin - (value - min_ret) / (max_ret - min_ret) * (height - 2 * margin)

    def circle(cx: float, cy: float, r: float, color: str, opacity: float = 1.0) -> str:
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{color}" fill-opacity="{opacity}" />'

    def text(x: float, y: float, content: str) -> str:
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Pretendard, Arial, sans-serif" '
            f'font-size="14" fill="#333">{content}</text>'
        )

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<g id="axes">',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333" stroke-width="2" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333" stroke-width="2" />',
        text(width / 2, height - margin / 2 + 10, "위험(표준편차)"),
        text(margin / 3, height / 2, "수익률",),
        '</g>',
    ]

    # 전체 포트폴리오 산점도
    svg_parts.append('<g id="portfolios" fill="#4a90e2" fill-opacity="0.4">')
    for p in portfolios:
        svg_parts.append(circle(scale_x(p.risk), scale_y(p.exp_return), 3, "#4a90e2", 0.35))
    svg_parts.append('</g>')

    # 효율적 투자 프런티어 라인
    svg_parts.append('<g id="frontier" fill="none" stroke="#e94e77" stroke-width="3">')
    if frontier:
        path_d = " ".join(
            f"{'M' if idx == 0 else 'L'} {scale_x(p.risk):.2f},{scale_y(p.exp_return):.2f}"
            for idx, p in enumerate(frontier)
        )
        svg_parts.append(f'<path d="{path_d}" />')
    svg_parts.append('</g>')

    # 프런티어 포인트 강조
    svg_parts.append('<g id="frontier_points">')
    for p in frontier:
        svg_parts.append(circle(scale_x(p.risk), scale_y(p.exp_return), 5, "#e94e77", 0.9))
    svg_parts.append('</g>')

    # 단일 자산 표시
    svg_parts.append('<g id="single_assets">')
    for ticker, (exp_ret, risk) in singles.items():
        x = scale_x(risk)
        y = scale_y(exp_ret)
        svg_parts.append(circle(x, y, 6, "#2ecc71", 0.9))
        svg_parts.append(text(x + 8, y - 8, ticker))
    svg_parts.append('</g>')

    svg_parts.append('</svg>')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    series = load_close_prices(DATA_PATH)
    stats = basic_statistics(series)
    cov = covariance_matrix(series)
    exp = expected_returns(series)
    tickers = list(series.keys())
    if len(tickers) < 3:
        raise ValueError("세 종목 이상이 필요합니다.")

    portfolios = generate_portfolios(tickers[:3], exp, cov, step=0.02)
    frontier_points = efficient_frontier(portfolios)

    summary = {
        "dataset": {
            "tickers": tickers,
            "date_start": min(ts.dates[0] for ts in series.values()).strftime("%Y-%m-%d"),
            "date_end": max(ts.dates[-1] for ts in series.values()).strftime("%Y-%m-%d"),
            "observations": min(len(ts.prices) for ts in series.values()),
        },
        "statistics": stats,
        "expected_returns": exp,
        "covariance": cov,
    }

    save_json(summary, OUTPUT_DIR / "analysis_summary.json")
    save_portfolio_csv(portfolios, OUTPUT_DIR / "portfolios.csv")
    save_portfolio_csv(frontier_points, OUTPUT_DIR / "efficient_frontier.csv")

    singles = {
        ticker: (exp[ticker], math.sqrt(cov[ticker][ticker]))
        for ticker in tickers[:3]
    }
    render_svg(portfolios, frontier_points, singles, SITE_ASSETS_DIR / "efficient_frontier.svg")

    print("분석이 완료되었습니다.")
    print(f"요약: {OUTPUT_DIR / 'analysis_summary.json'}")
    print(f"포트폴리오: {OUTPUT_DIR / 'portfolios.csv'}")
    print(f"프런티어: {OUTPUT_DIR / 'efficient_frontier.csv'}")
    print(f"SVG 그래프: {SITE_ASSETS_DIR / 'efficient_frontier.svg'}")


if __name__ == "__main__":
    main()
