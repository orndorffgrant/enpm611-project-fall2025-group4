"""
Feature 2: Issue Completion Time Analysis (Focused)

What it does (concise):
- Computes time-to-close for CLOSED issues.
- Shows ONE chart: Monthly median completion time (overall + top 3 labels by closed count).
- Prints a short summary (count, median, p90, fastest/slowest label).
- If there are NO closed issues, falls back to open-issue aging:
  * ONE chart: histogram of open ages
  * Short summary (count, median age, p90)

No seaborn. Single figure. Clean defaults. Minimal but useful.
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
    for attr in ("created_date", "created_at", "createdAt"):
        v = getattr(issue, attr, None)
        if v:
            return _to_utc_dt(v)
    return None

def _updated_at(issue: Issue) -> Optional[datetime]:
    for attr in ("updated_date", "updated_at", "updatedAt"):
        v = getattr(issue, attr, None)
        if v:
            return _to_utc_dt(v)
    return None

def _closed_at_from_events(issue: Issue) -> Optional[datetime]:
    evs = getattr(issue, "events", None) or []
    cands = []
    for e in evs:
        et = getattr(e, "event_type", None) or getattr(e, "type", None) or getattr(e, "event", None)
        if isinstance(et, str) and et.lower() == "closed":
            t = getattr(e, "event_date", None) or getattr(e, "created_at", None) or getattr(e, "timestamp", None)
            ts = _to_utc_dt(t)
            if ts:
                cands.append(ts)
    return max(cands) if cands else None

def _closed_at(issue: Issue) -> Optional[datetime]:
    for attr in ("closed_date", "closed_at", "closedAt"):
        v = getattr(issue, attr, None)
        if v:
            ts = _to_utc_dt(v)
            if ts:
                return ts
    ts = _closed_at_from_events(issue)
    if ts:
        return ts
    if getattr(issue, "state", None) == State.closed:
        return _updated_at(issue)
    return None

def _labels(issue: Issue) -> List[str]:
    labs = getattr(issue, "labels", None) or []
    out: List[str] = []
    for lb in labs:
        if isinstance(lb, str):
            out.append(lb)
        elif hasattr(lb, "name"):
            out.append(str(getattr(lb, "name")))
        else:
            out.append(str(lb))
    return out or ["unlabeled"]

def _url(issue: Issue) -> Optional[str]:
    for k in ("html_url", "url", "htmlUrl"):
        v = getattr(issue, k, None)
        if v:
            return str(v)
    num = getattr(issue, "number", None)
    return f"https://github.com/python-poetry/poetry/issues/{num}" if num else None

# -------------------------- main analysis --------------------------------- #

class CompletionAnalysis:
    """
    Single-focus analysis:
    - CLOSED: Monthly median completion time, overall + top 3 labels (one colored line chart)
    - OPEN fallback: Histogram of open ages
    """

    def __init__(self):
        self.issues: List[Issue] = DataLoader().get_issues()
        self.user_filter: Optional[str] = config.get_parameter("user")
        self.label_filter: Optional[str] = config.get_parameter("label")
        self.since: Optional[str] = config.get_parameter("since")
        self.filtered_issues = self._filter_issues()

    # ---------------- filters & core metrics ---------------- #

    def _filter_issues(self) -> List[Issue]:
        items = self.issues
        if self.user_filter:
            items = [i for i in items if getattr(i, "creator", None) == self.user_filter]
        if self.label_filter:
            items = [i for i in items if self.label_filter in _labels(i)]
        if self.since:
            start = pd.to_datetime(self.since, utc=True, errors="coerce")
            if start is not None and not pd.isna(start):
                kept = []
                for i in items:
                    c = _created_at(i)
                    if c and c >= start.to_pydatetime():
                        kept.append(i)
                items = kept
        return items

    def _completion_days(self, issue: Issue) -> Optional[float]:
        if getattr(issue, "state", None) != State.closed:
            return None
        c0 = _created_at(issue)
        c1 = _closed_at(issue)
        if not c0 or not c1:
            return None
        d = (c1 - c0).total_seconds() / 86400.0
        return d if d >= 0 else None

    def _open_age_days(self, issue: Issue) -> Optional[float]:
        if getattr(issue, "state", None) != State.open:
            return None
        c0 = _created_at(issue)
        if not c0:
            return None
        now = datetime.now(timezone.utc)
        return (now - c0).total_seconds() / 86400.0

    # ----------------------- public API ----------------------- #

    def run(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("FEATURE 2: ISSUE COMPLETION TIME ANALYSIS (Focused)")
        print("=" * 80)
        print(f"Analyzing {len(self.filtered_issues)} issues")
        if self.user_filter:
            print(f"Filtered by user:  {self.user_filter}")
        if self.label_filter:
            print(f"Filtered by label: {self.label_filter}")
        if self.since:
            print(f"Since (created ≥): {self.since}")
        print()

        closed = [i for i in self.filtered_issues if getattr(i, "state", None) == State.closed]
        open_ = [i for i in self.filtered_issues if getattr(i, "state", None) == State.open]
        print(f"Found {len(closed)} closed and {len(open_)} open issues")

        res_closed = self._analyze_closed_issues(closed)
        if res_closed is None:
            print("No usable closed timestamps → Open-Issue Aging fallback")
            return {"open": self._analyze_open_issues(open_)}

        return {"closed": res_closed}

    # ------------------- closed issues path ------------------- #

    def _analyze_closed_issues(self, closed_issues: List[Issue]) -> Optional[Dict[str, Any]]:
        records = []
        for it in closed_issues:
            d = self._completion_days(it)
            if d is not None:
                records.append({
                    "issue_number": getattr(it, "number", None),
                    "completion_time": d,
                    "labels": _labels(it),
                    "title": getattr(it, "title", "") or "",
                    "url": _url(it),
                    "created_at": _created_at(it),
                    "closed_at": _closed_at(it),
                })

        if not records:
            return None

        df = pd.DataFrame.from_records(records)
        s = df["completion_time"]

        # concise summary
        mean = float(s.mean())
        median = float(s.median())
        p90 = float(s.quantile(0.90))
        print(f"\nClosed issues analyzed: {len(df)}")
        print(f"Median time-to-close: {median:.1f} d  |  Mean: {mean:.1f} d  |  P90: {p90:.1f} d")

        # fastest / slowest labels (require small sample ≥ 3)
        lbl_rows = []
        for _, r in df.iterrows():
            for lab in r["labels"]:
                lbl_rows.append({"label": lab, "completion_time": r["completion_time"]})
        fastest_label = slowest_label = None
        if lbl_rows:
            lbl_df = pd.DataFrame(lbl_rows)
            stats = (
                lbl_df.groupby("label")["completion_time"]
                .agg(["median", "count"])
                .query("count >= 3")
                .sort_values("median", ascending=True)
            )
            if not stats.empty:
                fastest_label = stats.iloc[0]
                slowest_label = stats.iloc[-1]
                print(f"Fastest label: {stats.index[0]} ({fastest_label['median']:.1f} d, n={int(fastest_label['count'])})")
                print(f"Slowest label: {stats.index[-1]} ({slowest_label['median']:.1f} d, n={int(slowest_label['count'])})")

        # monthly median line(s): overall + top 3 labels by closed count
        df["closed_month"] = pd.to_datetime(df["closed_at"]).dt.to_period("M").astype(str)

        overall = (
            df.groupby("closed_month")["completion_time"]
            .median()
            .reset_index()
            .rename(columns={"completion_time": "median_days"})
        )

        # top 3 labels by number of closed appearances
        label_counts = pd.Series([lab for labs in df["labels"] for lab in labs]).value_counts()
        top_labels = label_counts.head(3).index.tolist()

        # pivot per label (median by month)
        label_lines = []
        if top_labels:
            rows = []
            for _, r in df.iterrows():
                for lab in r["labels"]:
                    if lab in top_labels:
                        rows.append({"label": lab, "closed_month": r["closed_month"], "completion_time": r["completion_time"]})
            if rows:
                ldf = pd.DataFrame(rows)
                label_lines = (
                    ldf.groupby(["label", "closed_month"])["completion_time"]
                    .median()
                    .reset_index()
                )

        # single colored chart
        self._plot_monthly_medians(overall, label_lines)

        return {
            "completion_df": df,
            "summary": {"count": len(df), "median": median, "mean": mean, "p90": p90},
        }

    # ------------------- open issues fallback ------------------- #

    def _analyze_open_issues(self, open_issues: List[Issue]) -> Dict[str, Any]:
        recs = []
        for it in open_issues:
            age = self._open_age_days(it)
            if age is not None:
                recs.append({
                    "issue_number": getattr(it, "number", None),
                    "age": age,
                    "title": getattr(it, "title", "") or "",
                    "url": _url(it),
                })
        if not recs:
            print("\nNo open issues to analyze.")
            return {"aging_df": pd.DataFrame()}

        df = pd.DataFrame.from_records(recs)
        s = df["age"]

        # concise summary
        mean = float(s.mean())
        median = float(s.median())
        p90 = float(s.quantile(0.90))
        print(f"\nOpen issues analyzed: {len(df)}")
        print(f"Median age: {median:.1f} d  |  Mean: {mean:.1f} d  |  P90: {p90:.1f} d")

        # one histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df["age"], bins=30)
        ax.set_title("Distribution of Open Issue Ages")
        ax.set_xlabel("Age (days)")
        ax.set_ylabel("Number of Issues")
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        plt.show()

        return {"aging_df": df, "summary": {"count": len(df), "median": median, "mean": mean, "p90": p90}}

    # --------------------------- plots --------------------------- #

    def _plot_monthly_medians(self, overall: pd.DataFrame, label_lines: List[pd.DataFrame]) -> None:
        fig, ax = plt.subplots(figsize=(11, 5))

        # overall line
        ax.plot(overall["closed_month"], overall["median_days"], marker="o", linewidth=2, label="All labels")

        # per-label lines (top 3)
        if isinstance(label_lines, pd.DataFrame) and not label_lines.empty:
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
    # allow overriding minimal filters via config if needed
    if config_dict and "since" in config_dict:
        analysis.since = str(config_dict["since"])
        analysis.filtered_issues = analysis._filter_issues()
    return analysis.run()

if __name__ == "__main__":
    run()