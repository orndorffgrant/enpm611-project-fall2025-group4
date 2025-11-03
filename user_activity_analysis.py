from typing import List
import matplotlib.pyplot as plt

from data_loader import DataLoader
from labels import AREA_LABELS, KIND_LABELS
from model import Issue
import config


class UserActivityAnalysis:
    """
    Visualizes activity of a particular user broken down by type.
    """
    
    def __init__(self):
        """
        Constructor
        """
        # Parameter is passed in via command line (--user)
        self._user: str = config.get_parameter('user')
        if config.get_parameter('label'):
            raise RuntimeError("--label flag is not supported for feature 1")
        
    def _event_to_year_month(self, e):
        return f"{e["event_date"].year}-{e["event_date"].month:02d}"
    
    def run(self):
        """
        Starting point for this analysis.
        """
        user = self._user
        if not user:
            raise RuntimeError("--user is required for feature 1")
            
        print("Loading issues...")
        issues: List[Issue] = DataLoader().get_issues()
        
        # Get all issues authored by the user
        # structure them like events with type "opened" for easier processing alongside other events
        print("Gathering opened issues...")
        events = [
            {
                "event_type": "opened",
                "event_date": i.created_date,
                "author": user,
                "labels": i.labels
            } for i in issues
            if i.creator == user
        ]
        
        print("Gathering all other activity...")
        for i in issues:
            for e in i.events:
                if e.author == user:
                    events.append(
                        {
                            "event_type": e.event_type,
                            "event_date": e.event_date,
                            "author": user,
                            "labels": i.labels,
                        }
                    )
        
        events.sort(key=lambda e: e["event_date"])
        
        year_month_buckets = {self._event_to_year_month(e) for e in events}
        
        print("Collecting activity stats by label by month...")
        activity_by_label_by_year_month = {}
        for e in events:
            bucket = self._event_to_year_month(e)
            if bucket not in activity_by_label_by_year_month:
                activity_by_label_by_year_month[bucket] = {}
            for label in e["labels"]:
                if label not in activity_by_label_by_year_month[bucket]:
                    activity_by_label_by_year_month[bucket][label] = 0
                activity_by_label_by_year_month[bucket][label] += 1
        
        print("Collecting 'area' activity by month...")
        area_activity = {area: [] for area in AREA_LABELS}
        for year_month in sorted(activity_by_label_by_year_month.keys()):
            for area in AREA_LABELS:
                area_activity[area].append(activity_by_label_by_year_month[year_month].get(area, 0))
            
        print("Collecting 'kind' activity by month...")
        kind_activity = {kind: [] for kind in KIND_LABELS}
        for year_month in sorted(activity_by_label_by_year_month.keys()):
            for kind in KIND_LABELS:
                kind_activity[kind].append(activity_by_label_by_year_month[year_month].get(kind, 0))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        fig.suptitle(f"Activity of {user}")
        
        year_month_bucket_list = sorted(list(year_month_buckets))
        
        ax1.stackplot(
            year_month_bucket_list,
            area_activity.values(),
            labels=[area.lstrip("area/") for area in area_activity.keys()],
            colors=[
                "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0", "#fd7f6f", "#7eb0d5",
                "#ea5545", "#f46a9b", "#ef9b20", "#edbf33", "#ede15b", "#bdcf32", "#87bc45", "#27aeef", "#b33dc6", "#e60049",
                "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7", "#ffb400", "#a57c1b", "#9080ff",
            ]
        )
        ax1.legend(ncols=2, reverse=True)
        ax1.set_title("Activity Area")
        ax1.set_ylabel("Action Count")
        
        ax2.stackplot(
            year_month_bucket_list,
            kind_activity.values(),
            labels=[kind.lstrip("kind/") for kind in kind_activity.keys()],
            mouseover=True,
        )
        ax2.legend(reverse=True)
        ax2.set_title("Activity Kind")
        ax2.set_xlabel("Year Month")
        ax2.set_ylabel("Action Count")
        ax2.tick_params(axis="x", rotation=90)
        
        plt.show()
