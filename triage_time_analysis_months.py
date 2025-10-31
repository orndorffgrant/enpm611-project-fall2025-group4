from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import DataLoader
from model import Issue, Event
import config

# Average days per month (Gregorian average)
_DAYS_PER_MONTH = 365.25 / 12.0  # ≈ 30.4375

class TriageTimeAnalysis:
    def __init__(self):
        self.USER: Optional[str] = config.get_parameter("user")

    def run(self):
        issues: List[Issue] = DataLoader().get_issues()
        total_events = sum(
            len([e for e in issue.events if self.USER is None or e.author == self.USER])
            for issue in issues
        )
        output = f"Found {total_events} events across {len(issues)} issues"
        if self.USER is not None:
            output += f" for {self.USER}."
        else:
            output += "."
        print("\n\n" + output + "\n\n")

    def _first_assignment_event(self, issue: Issue) -> Optional[Event]:
        created = issue.created_date
        if created is None:
            return None
        sorted_events = sorted(issue.events, key=lambda ev: ev.event_date or created)
        for e in sorted_events:
            if not getattr(e, "event_type", None):
                continue
            et = str(e.event_type).strip().lower()
            if et in ("assigned", "assign", "assignment", "status_change", "state_change"):
                return e
            if (getattr(e, "label", None) and "assign" in str(e.label).lower()) or \
               (getattr(e, "comment", None) and "assign" in str(e.comment).lower()):
                return e
        return None

    def triage_time_analysis(self, show_plot: bool = False) -> pd.DataFrame:
        """
        Compute time from issue creation to first ASSIGNED event, expressed in months.
        Returns a DataFrame with triage times in months and prints summary stats.
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
                # convert to months (float): seconds -> days -> months
                months = delta.total_seconds() / 3600.0 / 24.0 / _DAYS_PER_MONTH
                rows.append({
                    "issue_number": issue.number,
                    "creator": issue.creator,
                    "created_date": created,
                    "assigned_date": first_assigned.event_date,
                    "triage_months": months
                })

        df = pd.DataFrame.from_records(rows)
        if df.empty:
            print("No triage/assignment events found in dataset.")
            return df

        summary = {
            "count": int(df.shape[0]),
            "mean_months": float(df["triage_months"].mean()),
            "median_months": float(df["triage_months"].median()),
            "min_months": float(df["triage_months"].min()),
            "max_months": float(df["triage_months"].max()),
            "std_months": float(df["triage_months"].std())
        }
        print("Triage time summary (months):")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if show_plot:
            plt.figure(figsize=(10, 5))
            plt.hist(df["triage_months"], bins=40, color="#2a9d8f", edgecolor="black")
            plt.title("Distribution of triage time (months)")
            plt.xlabel("Months from creation to first assignment")
            plt.ylabel("Number of issues")
            plt.grid(axis="y", alpha=0.4)
            plt.show()

        return df

if __name__ == "__main__":
    t = TriageTimeAnalysis()
    t.run()
