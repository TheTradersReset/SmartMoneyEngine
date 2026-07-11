"""
Interactive review dashboard for SmartMoneyEngine pipeline output.

Loads enriched pipeline CSV data and renders an interactive Plotly
candlestick chart with SMC signal overlays.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "review"
DEFAULT_HTML_PATH = DEFAULT_OUTPUT_DIR / "review.html"
DEFAULT_PNG_PATH = DEFAULT_OUTPUT_DIR / "review.png"

SIGNAL_COLORS: dict[str, str] = {
    "Swing High": "#E74C3C",
    "Swing Low": "#3498DB",
    "HH": "#1ABC9C",
    "HL": "#16A085",
    "LH": "#E67E22",
    "LL": "#D35400",
    "Bullish BOS": "#00B894",
    "Bearish BOS": "#FF7675",
    "Bullish CHOCH": "#0984E3",
    "Bearish CHOCH": "#6C5CE7",
    "Bullish FVG": "rgba(0, 184, 148, 0.25)",
    "Bearish FVG": "rgba(255, 118, 117, 0.25)",
    "Bullish Order Block": "rgba(9, 132, 227, 0.30)",
    "Bearish Order Block": "rgba(230, 126, 34, 0.30)",
    "Liquidity Buy": "#FDCB6E",
    "Liquidity Sell": "#E17055",
}


class ReviewDashboardError(Exception):
    """Raised when the review dashboard cannot be generated."""


@dataclass(frozen=True)
class SignalOverlay:
    """Configuration for a pipeline signal overlay."""

    name: str
    column: str | None = None
    top_column: str | None = None
    bottom_column: str | None = None
    zone: bool = False
    liquidity: bool = False


POINT_OVERLAYS: tuple[SignalOverlay, ...] = (
    SignalOverlay(name="Swing High", column="Swing_High"),
    SignalOverlay(name="Swing Low", column="Swing_Low"),
    SignalOverlay(name="HH", column="HH"),
    SignalOverlay(name="HL", column="HL"),
    SignalOverlay(name="LH", column="LH"),
    SignalOverlay(name="LL", column="LL"),
    SignalOverlay(name="Bullish BOS", column="Bullish_BOS"),
    SignalOverlay(name="Bearish BOS", column="Bearish_BOS"),
    SignalOverlay(name="Bullish CHOCH", column="Bullish_CHOCH"),
    SignalOverlay(name="Bearish CHOCH", column="Bearish_CHOCH"),
)

ZONE_OVERLAYS: tuple[SignalOverlay, ...] = (
    SignalOverlay(
        name="Bullish FVG",
        top_column="Bullish_FVG_Top",
        bottom_column="Bullish_FVG_Bottom",
        zone=True,
    ),
    SignalOverlay(
        name="Bearish FVG",
        top_column="Bearish_FVG_Top",
        bottom_column="Bearish_FVG_Bottom",
        zone=True,
    ),
    SignalOverlay(
        name="Bullish Order Block",
        top_column="Bullish_OB_High",
        bottom_column="Bullish_OB_Low",
        zone=True,
    ),
    SignalOverlay(
        name="Bearish Order Block",
        top_column="Bearish_OB_High",
        bottom_column="Bearish_OB_Low",
        zone=True,
    ),
)

LIQUIDITY_OVERLAYS: tuple[SignalOverlay, ...] = (
    SignalOverlay(name="Liquidity Buy", column="Buy_Side_Liquidity", liquidity=True),
    SignalOverlay(name="Liquidity Sell", column="Sell_Side_Liquidity", liquidity=True),
)


class ReviewDashboard:
    """
    Build and export an interactive SmartMoneyEngine review dashboard.

    Parameters
    ----------
    input_csv : Path | None, optional
        Pipeline CSV path.
    output_html : Path | None, optional
        HTML export path.
    output_png : Path | None, optional
        PNG export path.
    """

    def __init__(
        self,
        input_csv: Path | None = None,
        output_html: Path | None = None,
        output_png: Path | None = None,
    ) -> None:
        self.input_csv = input_csv if input_csv is not None else DEFAULT_INPUT_CSV
        self.output_html = output_html if output_html is not None else DEFAULT_HTML_PATH
        self.output_png = output_png if output_png is not None else DEFAULT_PNG_PATH

    @staticmethod
    def load_pipeline_csv(path: Path | str) -> pd.DataFrame:
        """
        Load and prepare pipeline CSV data.

        Parameters
        ----------
        path : Path | str
            Pipeline CSV path.

        Returns
        -------
        pd.DataFrame
            Sorted dataframe with parsed timestamps.
        """
        csv_path = Path(path)
        if not csv_path.exists():
            raise ReviewDashboardError(f"Pipeline CSV not found: {csv_path}")

        logger.info("Loading pipeline CSV from %s", csv_path)
        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise ReviewDashboardError(f"Pipeline CSV is empty: {csv_path}")

        required = ("Date", "Open", "High", "Low", "Close")
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ReviewDashboardError(f"Pipeline CSV missing required columns: {missing}")

        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        if frame["Date"].isna().any():
            raise ReviewDashboardError("Pipeline CSV contains invalid Date values.")

        frame = frame.sort_values("Date").reset_index(drop=True)
        logger.info("Loaded %s pipeline rows.", len(frame))
        return frame

    @staticmethod
    def _build_signal_labels(frame: pd.DataFrame) -> pd.Series:
        """Build per-row signal labels for candle hover text."""
        labels: list[str] = []

        signal_map = {
            "Swing High": "Swing_High",
            "Swing Low": "Swing_Low",
            "HH": "HH",
            "HL": "HL",
            "LH": "LH",
            "LL": "LL",
            "Bullish BOS": "Bullish_BOS",
            "Bearish BOS": "Bearish_BOS",
            "Bullish CHOCH": "Bullish_CHOCH",
            "Bearish CHOCH": "Bearish_CHOCH",
            "Bullish FVG": "Bullish_FVG_Top",
            "Bearish FVG": "Bearish_FVG_Top",
            "Bullish OB": "Bullish_OB_High",
            "Bearish OB": "Bearish_OB_High",
            "Liquidity Buy": "Buy_Side_Liquidity",
            "Liquidity Sell": "Sell_Side_Liquidity",
        }

        for _, row in frame.iterrows():
            active = [name for name, column in signal_map.items() if column in frame.columns and pd.notna(row.get(column))]
            labels.append(", ".join(active) if active else "None")

        return pd.Series(labels, index=frame.index)

    @staticmethod
    def _legend_marker(color: str) -> dict[str, Any]:
        """Create a legend-only marker trace."""
        return {
            "x": [None],
            "y": [None],
            "mode": "markers",
            "marker": {"size": 10, "color": color, "symbol": "circle"},
            "showlegend": True,
        }

    def build_figure(self, frame: pd.DataFrame) -> go.Figure:
        """
        Build the interactive Plotly figure.

        Parameters
        ----------
        frame : pd.DataFrame
            Prepared pipeline dataframe.

        Returns
        -------
        go.Figure
            Interactive figure with candlesticks and overlays.
        """
        signal_labels = self._build_signal_labels(frame)
        hover_text = [
            (
                f"Date: {row['Date']}<br>"
                f"Open: {row['Open']}<br>"
                f"High: {row['High']}<br>"
                f"Low: {row['Low']}<br>"
                f"Close: {row['Close']}<br>"
                f"Signal: {signal_labels.iloc[index]}"
            )
            for index, row in frame.iterrows()
        ]

        figure = make_subplots(rows=1, cols=1)

        figure.add_trace(
            go.Candlestick(
                x=frame["Date"],
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                name="Price",
                increasing_line_color="#26A69A",
                decreasing_line_color="#EF5350",
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

        for overlay in ZONE_OVERLAYS:
            assert overlay.top_column is not None
            assert overlay.bottom_column is not None
            color = SIGNAL_COLORS[overlay.name]
            border_color = color.replace("0.25", "0.8").replace("0.30", "0.8")
            mask = frame[overlay.top_column].notna() & frame[overlay.bottom_column].notna()
            zone_rows = frame.loc[mask]
            for index, row in zone_rows.iterrows():
                start = pd.Timestamp(row["Date"]).isoformat()
                end_index = min(index + 1, len(frame) - 1)
                end = pd.Timestamp(frame.loc[end_index, "Date"]).isoformat()
                figure.add_shape(
                    type="rect",
                    x0=start,
                    x1=end,
                    y0=float(row[overlay.bottom_column]),
                    y1=float(row[overlay.top_column]),
                    fillcolor=color,
                    line={"width": 1, "color": border_color},
                    layer="below",
                )

            figure.add_trace(
                go.Scatter(
                    name=overlay.name,
                    **self._legend_marker(border_color if "rgba" in border_color else color),
                )
            )

        for overlay in POINT_OVERLAYS:
            assert overlay.column is not None
            mask = frame[overlay.column].notna()
            points = frame.loc[mask]
            if points.empty:
                figure.add_trace(
                    go.Scatter(
                        name=overlay.name,
                        **self._legend_marker(SIGNAL_COLORS[overlay.name]),
                    )
                )
                continue

            figure.add_trace(
                go.Scatter(
                    x=points["Date"],
                    y=points[overlay.column],
                    mode="markers",
                    name=overlay.name,
                    marker={
                        "size": 8,
                        "color": SIGNAL_COLORS[overlay.name],
                        "symbol": "diamond" if "BOS" in overlay.name or "CHOCH" in overlay.name else "circle",
                    },
                    hovertemplate=(
                        f"{overlay.name}<br>"
                        "Date: %{x}<br>"
                        "Price: %{y}<extra></extra>"
                    ),
                )
            )

        for overlay in LIQUIDITY_OVERLAYS:
            assert overlay.column is not None
            mask = frame[overlay.column].notna()
            points = frame.loc[mask]
            if points.empty:
                figure.add_trace(
                    go.Scatter(
                        name=overlay.name,
                        **self._legend_marker(SIGNAL_COLORS[overlay.name]),
                    )
                )
                continue

            figure.add_trace(
                go.Scatter(
                    x=points["Date"],
                    y=points[overlay.column],
                    mode="markers",
                    name=overlay.name,
                    legendgroup="Liquidity",
                    legendgrouptitle={"text": "Liquidity"},
                    marker={
                        "size": 5,
                        "color": SIGNAL_COLORS[overlay.name],
                        "symbol": "line-ew",
                    },
                    hovertemplate=(
                        f"{overlay.name}<br>"
                        "Date: %{x}<br>"
                        "Level: %{y}<extra></extra>"
                    ),
                )
            )

        figure.update_layout(
            title="SmartMoneyEngine Review Dashboard - NIFTY50 5m",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            hovermode="x unified",
            dragmode="zoom",
            xaxis={
                "rangeslider": {"visible": False},
                "fixedrange": False,
            },
            yaxis={"fixedrange": False},
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
            },
            height=900,
            margin={"l": 60, "r": 240, "t": 80, "b": 60},
        )
        figure.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
        figure.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

        return figure

    def export(self, figure: go.Figure) -> tuple[Path, Path]:
        """
        Export the dashboard to HTML and PNG.

        Returns
        -------
        tuple[Path, Path]
            Paths to the HTML and PNG exports.
        """
        self.output_html.parent.mkdir(parents=True, exist_ok=True)
        self.output_png.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting review HTML to %s", self.output_html)
        figure.write_html(str(self.output_html), include_plotlyjs="cdn", full_html=True)

        logger.info("Exporting review PNG to %s", self.output_png)
        figure.write_image(str(self.output_png), width=1920, height=1080, scale=2)

        return self.output_html, self.output_png

    def run(self) -> tuple[go.Figure, Path, Path]:
        """Load data, build the dashboard, and export outputs."""
        started = time.perf_counter()
        frame = self.load_pipeline_csv(self.input_csv)
        figure = self.build_figure(frame)
        html_path, png_path = self.export(figure)
        elapsed = time.perf_counter() - started
        logger.info(
            "Review dashboard completed in %.3fs | rows=%s | html=%s | png=%s",
            elapsed,
            len(frame),
            html_path,
            png_path,
        )
        return figure, html_path, png_path


def build_review_dashboard(
    input_csv: Path | str | None = None,
    output_html: Path | str | None = None,
    output_png: Path | str | None = None,
) -> tuple[go.Figure, Path, Path]:
    """Build and export the review dashboard."""
    dashboard = ReviewDashboard(
        input_csv=Path(input_csv) if input_csv is not None else None,
        output_html=Path(output_html) if output_html is not None else None,
        output_png=Path(output_png) if output_png is not None else None,
    )
    return dashboard.run()


def main() -> int:
    """CLI entry point."""
    try:
        _, html_path, png_path = build_review_dashboard()
        print("Review Dashboard Generated")
        print(f"Input CSV: {DEFAULT_INPUT_CSV}")
        print(f"HTML: {html_path}")
        print(f"PNG: {png_path}")
        return 0
    except ReviewDashboardError as exc:
        logger.error("Review dashboard error: %s", exc)
        print(f"Review dashboard error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected review dashboard failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
