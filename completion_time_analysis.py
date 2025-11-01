"""
Feature 2: Issue Completion Time Analysis (Focused)

- Computes time-to-close for CLOSED issues only (no fallback).
- ONE chart: Monthly median completion time (overall + top 3 labels by closed count).
- Short summary: count, median, mean, P90; fastest/slowest labels if sample >= 3.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import DataLoader
from model import Issue, State
import config


# ----------------------------- helpers ------------------------------------ #

def _to_utc_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    ts = pd.to_datetime(str(x), utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()

def _created_at(issue: Issue) -> Optional[datetime]:
    """Get created date from issue."""
    return _to_utc_dt(issue.created_date) if issue.created_date else None

def _updated_at(issue: Issue) -> Optional[datetime]:
    """Get updated date from issue."""
    return _to_utc_dt(issue.updated_date) if issue.updated_date else None

def _closed_at_from_events(issue: Issue) -> Optional[datetime]:
    """Find closed date from events."""
    cands = []
    for e in issue.events:
        if e.event_type and e.event_type.lower() == "closed" and e.event_date:
            ts = _to_utc_dt(e.event_date)
            if ts:
                cands.append(ts)
    return max(cands) if cands else None

def _closed_at(issue: Issue) -> Optional[datetime]:
    """Get closed date from issue events, or fallback to updated_date if closed."""
    ts = _closed_at_from_events(issue)
    if ts:
        return ts
    # Fallback: if issue is closed, use updated_date
    if issue.state == State.closed:
        return _updated_at(issue)
    return None

def _labels(issue: Issue) -> List[str]:
    """Get labels from issue."""
    return issue.labels if issue.labels else ["unlabeled"]

def _url(issue: Issue) -> Optional[str]:
    """Get URL from issue."""
    return issue.url if issue.url else f"https://github.com/python-poetry/poetry/issues/{issue.number}" if issue.number >= 0 else None

# -------------------------- main analysis --------------------------------- #

class CompletionAnalysis:
    """
    CLOSED issues only. One chart + concise summary.
    """

    def __init__(self):
        self.issues: List[Issue] = DataLoader().get_issues()
        self.user_filter: Optional[str] = config.get_parameter("user")
        self.label_filter: Optional[str] = config.get_parameter("label")
        self.since: Optional[str] = config.get_parameter("since")
        self.filtered_issues = self._filter_issues()

    def _filter_issues(self) -> List[Issue]:
        items = self.issues
        if self.user_filter:
            items = [i for i in items if i.creator == self.user_filter]
        if self.label_filter:
            items = [i for i in items if self.label_filter in _labels(i)]
        if self.since:
            start = pd.to_datetime(self.since, utc=True, errors="coerce")
            if not pd.isna(start):
                items = [i for i in items if (c := _created_at(i)) and c >= start.to_pydatetime()]
        return items

    def _completion_days(self, issue: Issue) -> Optional[float]:
        if issue.state != State.closed:
            return None
        c0 = _created_at(issue)
        c1 = _closed_at(issue)
        if not c0 or not c1:
            return None
        d = (c1 - c0).total_seconds() / 86400.0
        return d if d >= 0 else None

    def run(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("FEATURE 2: ISSUE COMPLETION TIME ANALYSIS (Closed Issues Only)")
        print("=" * 80)
        print(f"Analyzing {len(self.filtered_issues)} issues")
        if self.user_filter:
            print(f"Filtered by user:  {self.user_filter}")
        if self.label_filter:
            print(f"Filtered by label: {self.label_filter}")
        if self.since:
            print(f"Since (created ≥): {self.since}")
        print()

        closed = [i for i in self.filtered_issues if i.state == State.closed]
        print(f"Found {len(closed)} closed issues")

        res = self._analyze_closed_issues(closed)
        if res is None:
            print("No usable closed timestamps. Nothing to plot.")
            return {"closed": {}}
        return {"closed": res}

    def _analyze_closed_issues(self, closed_issues: List[Issue]) -> Optional[Dict[str, Any]]:
        rows = []
        for it in closed_issues:
            d = self._completion_days(it)
            if d is not None:
                rows.append({
                    "issue_number": it.number,
                    "completion_time": d,
                    "labels": _labels(it),
                    "title": it.title or "",
                    "url": _url(it),
                    "closed_at": _closed_at(it),
                })
        if not rows:
            return None

        df = pd.DataFrame.from_records(rows)
        s = df["completion_time"]

        # concise summary
        mean = float(s.mean())
        median = float(s.median())
        p90 = float(s.quantile(0.90))
        print(f"\nClosed issues analyzed: {len(df)}")
        print(f"Median time-to-close: {median:.1f} d  |  Mean: {mean:.1f} d  |  P90: {p90:.1f} d")

        # fastest / slowest labels (sample ≥ 3)
        lbl_df = df[["labels", "completion_time"]].explode("labels")
        stats = (
            lbl_df.groupby("labels")["completion_time"]
            .agg(["median", "count"])
            .query("count >= 3")
            .sort_values("median", ascending=True)
        )
        if not stats.empty:
            fastest = stats.iloc[0]
            slowest = stats.iloc[-1]
            print(f"Fastest label: {stats.index[0]} ({fastest['median']:.1f} d, n={int(fastest['count'])})")
            print(f"Slowest label: {stats.index[-1]} ({slowest['median']:.1f} d, n={int(slowest['count'])})")

        # month (by closed date) medians: overall + top 3 labels
        if df["closed_at"].isna().all():
            print("\nNo closed timestamps available to plot monthly trend.")
            return {"completion_df": df, "summary": {"count": len(df), "median": median, "mean": mean, "p90": p90}}

        df["closed_month"] = pd.to_datetime(df["closed_at"], utc=True).dt.tz_localize(None).dt.to_period("M").astype(str)

        overall = (
            df.groupby("closed_month")["completion_time"]
            .median()
            .reset_index()
            .rename(columns={"completion_time": "median_days"})
        )

        # top 3 labels by closed appearances
        label_counts = lbl_df["labels"].value_counts()
        top_labels = label_counts.head(3).index.tolist()

        if top_labels:
            # Recreate lbl_df with closed_month for monthly analysis
            lbl_df_monthly = df[["labels", "closed_month", "completion_time"]].explode("labels")
            label_lines = (
                lbl_df_monthly[lbl_df_monthly["labels"].isin(top_labels)]
                .groupby(["labels", "closed_month"])["completion_time"]
                .median()
                .reset_index()
                .rename(columns={"labels": "label"})
            )
        else:
            label_lines = pd.DataFrame()

        self._plot_monthly_medians(overall, label_lines)
        return {"completion_df": df, "summary": {"count": len(df), "median": median, "mean": mean, "p90": p90}}

    # --------------------------- plot --------------------------- #

    def _plot_monthly_medians(self, overall: pd.DataFrame, label_lines: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(11, 5))

        # overall line
        ax.plot(overall["closed_month"], overall["median_days"], marker="o", linewidth=2, label="All labels")

        # per-label lines (top 3) with default matplotlib colors
        if not label_lines.empty:
            for lab, sub in label_lines.groupby("label"):
                ax.plot(sub["closed_month"], sub["completion_time"], marker="o", linewidth=1.8, label=lab)

        ax.set_title("Monthly Median Time-to-Close (Overall + Top Labels)")
        ax.set_xlabel("Closed Month")
        ax.set_ylabel("Median Days")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        plt.show()


# --------------------------- module API ----------------------------------- #

def run(issues: List[Issue] = None, config_dict: Dict = None) -> Dict[str, Any]:
    analysis = CompletionAnalysis()
    if issues is not None:
        analysis.issues = issues
        analysis.filtered_issues = analysis._filter_issues()
    if config_dict and "since" in config_dict:
        analysis.since = str(config_dict["since"])
        analysis.filtered_issues = analysis._filter_issues()
    return analysis.run()

if __name__ == "__main__":
    run()
