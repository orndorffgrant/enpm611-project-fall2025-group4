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
        issues: List[Issue] = DataLoader().get_issues()
        total_events = sum(len([e for e in issue.events if self.USER is None or e.author == self.USER]) for issue in issues)
        output = f"Found {total_events} events across {len(issues)} issues"
        if self.USER is not None:
            output += f" for {self.USER}."
        else:
            output += "."
        print("\n\n" + output + "\n\n")

    def triage_time_analysis(self, show_plot: bool = False) -> pd.DataFrame:
        """
        Compute time from issue creation to first ASSIGNED event.
        Returns a DataFrame with triage times (hours) and prints summary stats.
        """
        issues: List[Issue] = DataLoader().get_issues()
        rows = []
        for issue in issues:
            created = issue.created_date
            if created is None:
                continue
            # find first event that indicates assignment
            first_assigned = None
            # sort events by event_date, fall back to created if event_date is None
            sorted_events = sorted(issue.events, key=lambda ev: ev.event_date or created)
            for e in sorted_events:
                if not e.event_type:
                    continue
                et = e.event_type.strip().lower()
                if et in ("assigned", "assign", "assignment", "status_change", "state_change"):
                    first_assigned = e
                    break
                # some event payloads may use different encodings; also check label or comment heuristics
                if (e.label and "assign" in str(e.label).lower()) or (e.comment and "assign" in str(e.comment).lower()):
                    first_assigned = e
                    break
            if first_assigned and first_assigned.event_date:
                delta = first_assigned.event_date - created
                hours = delta.total_seconds() / 3600.0
                rows.append({
                    "issue_number": issue.number,
                    "creator": issue.creator,
                    "created_date": created,
                    "assigned_date": first_assigned.event_date,
                    "triage_hours": hours
                })

        df = pd.DataFrame.from_records(rows)
        if df.empty:
            print("No triage/assignment events found in dataset.")
            return df

        summary = {
            "count": int(df.shape[0]),
            "mean_hours": float(df["triage_hours"].mean()),
            "median_hours": float(df["triage_hours"].median()),
            "min_hours": float(df["triage_hours"].min()),
            "max_hours": float(df["triage_hours"].max()),
            "std_hours": float(df["triage_hours"].std())
        }
        print("Triage time summary (hours):")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if show_plot:
            plt.figure(figsize=(10, 5))
            plt.hist(df["triage_hours"], bins=40, color="#2a9d8f", edgecolor="black")
            plt.title("Distribution of triage time (hours)")
            plt.xlabel("Hours from creation to first assignment")
            plt.ylabel("Number of issues")
            plt.grid(axis="y", alpha=0.4)
            plt.show()

        return df

if __name__ == "__main__":
    t = TriageTimeAnalysis()
    t.run()
