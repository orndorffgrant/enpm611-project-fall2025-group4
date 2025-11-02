from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from data_loader import DataLoader
from model import Issue, Event
import config

class TriageTimeAnalysis:
    def __init__(self):
        self.USER: Optional[str] = config.get_parameter("user")

    def run(self):
        """
        Main entry point for triage time analysis.
        Calls triage_time_analysis with show_plot=True by default.
        """
        self.triage_time_analysis(show_plot=True)

    def _first_assignment_event(self, issue: Issue) -> Optional[Event]:
        created = issue.created_date
        if created is None:
            return None
        sorted_events = sorted(issue.events, key=lambda ev: ev.event_date or created)
        for e in sorted_events:
            if not getattr(e, "event_type", None):
                continue
            et = str(e.event_type).strip().lower()
            if et in ("assigned", "assign", "assignment", "status_change",
                      "state_change"):
                return e
            if (getattr(e, "label", None) and "assign" in str(e.label).lower()) or \
               (getattr(e, "comment", None) and "assign" in str(e.comment).lower()):
                return e
        return None

    def triage_time_analysis(self, show_plot: bool = False) -> pd.DataFrame:
        """
        Compute time from issue creation to first ASSIGNED event, expressed in days.
        Returns a DataFrame with triage times in days and prints summary stats.
        """
        issues: List[Issue] = DataLoader().get_issues()
        rows = []
        for issue in issues:
            created = issue.created_date
            if created is None:
                continue
            first_assigned = self._first_assignment_event(issue)
            if first_assigned and getattr(first_assigned, "event_date", None):
                delta = first_assigned.event_date - created
                # convert to days (float)
                days = delta.total_seconds() / 3600.0 / 24.0
                rows.append({
                    "issue_number": issue.number,
                    "creator": issue.creator,
                    "created_date": created,
                    "assigned_date": first_assigned.event_date,
                    "triage_days": days
                })

        df = pd.DataFrame.from_records(rows)
        if df.empty:
            print("No triage/assignment events found in dataset.")
            return df

        summary = {
            "count": int(df.shape[0]),
            "mean_days": float(df["triage_days"].mean()),
            "median_days": float(df["triage_days"].median()),
            "min_days": float(df["triage_days"].min()),
            "max_days": float(df["triage_days"].max()),
            "std_days": float(df["triage_days"].std())
        }

        print("Triage time summary (days):")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if show_plot:
            plt.figure(figsize=(10, 5))
            plt.hist(df["triage_days"], bins=40, color="#2a9d8f", edgecolor="black")
            plt.title("Distribution of triage time (days)")
            plt.xlabel("Days from creation to first assignment")
            plt.ylabel("Number of issues")
            plt.grid(axis="y", alpha=0.4)
            plt.show()

        return df


if __name__ == "__main__":
    t = TriageTimeAnalysis()
    t.run()