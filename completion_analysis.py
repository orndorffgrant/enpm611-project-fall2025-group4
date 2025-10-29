"""
Feature 2: Issue Completion Time Analysis

Analyzes GitHub issues completion times and provides comprehensive statistics
and visualizations for both closed and open issues.
"""

from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dateutil import parser
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from model import Issue, Event, State
import config


class CompletionAnalysis:
    """
    Comprehensive analysis of issue completion times and aging patterns.
    Handles both closed issues (completion time) and open issues (aging).
    """
    
    def __init__(self):
        """Initialize the analysis with data loading and configuration."""
        self.issues: List[Issue] = DataLoader().get_issues()
        self.user_filter = config.get_parameter('user')
        self.label_filter = config.get_parameter('label')
        
        # Filter issues based on command line parameters
        self.filtered_issues = self._filter_issues()
        
        # Results storage
        self.results = {}
        
    def _filter_issues(self) -> List[Issue]:
        """Filter issues based on user and label parameters."""
        filtered = self.issues
        
        if self.user_filter:
            filtered = [issue for issue in filtered if issue.creator == self.user_filter]
            
        if self.label_filter:
            filtered = [issue for issue in filtered if self.label_filter in issue.labels]
            
        return filtered
    
    def _to_utc_timestamp(self, date_str: str) -> Optional[datetime]:
        """Convert date string to UTC datetime robustly."""
        if not date_str:
            return None
            
        try:
            # Parse the date string
            dt = parser.parse(date_str)
            
            # If timezone naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC
                dt = dt.astimezone(timezone.utc)
                
            return dt
        except (ValueError, TypeError):
            return None
    
    def _get_closed_date(self, issue: Issue) -> Optional[datetime]:
        """Get the actual closed date from issue events or updated_date."""
        # First try to find a 'closed' event
        for event in issue.events:
            if event.event_type == 'closed' and event.event_date:
                return self._to_utc_timestamp(str(event.event_date))
        
        # Fall back to updated_date if issue is closed
        if issue.state == State.closed and issue.updated_date:
            return self._to_utc_timestamp(str(issue.updated_date))
            
        return None
    
    def _calculate_completion_time(self, issue: Issue) -> Optional[float]:
        """Calculate completion time in days for a closed issue."""
        if issue.state != State.closed:
            return None
            
        created_date = self._to_utc_timestamp(str(issue.created_date)) if issue.created_date else None
        closed_date = self._get_closed_date(issue)
        
        if not created_date or not closed_date:
            return None
            
        # Calculate difference in days
        time_diff = closed_date - created_date
        return time_diff.total_seconds() / (24 * 3600)
    
    def _calculate_open_age(self, issue: Issue) -> Optional[float]:
        """Calculate age in days for an open issue."""
        if issue.state != State.open:
            return None
            
        created_date = self._to_utc_timestamp(str(issue.created_date)) if issue.created_date else None
        if not created_date:
            return None
            
        # Calculate age from creation to now
        now = datetime.now(timezone.utc)
        time_diff = now - created_date
        return time_diff.total_seconds() / (24 * 3600)
    
    def _get_issue_labels(self, issue: Issue) -> List[str]:
        """Get all labels for an issue, handling empty labels."""
        return issue.labels if issue.labels else ['unlabeled']
    
    def run(self) -> Dict:
        """Main analysis method that handles both closed and open issues."""
        print(f"\n{'='*80}")
        print("ISSUE COMPLETION TIME ANALYSIS")
        print(f"{'='*80}")
        print(f"Analyzing {len(self.filtered_issues)} issues")
        if self.user_filter:
            print(f"Filtered by user: {self.user_filter}")
        if self.label_filter:
            print(f"Filtered by label: {self.label_filter}")
        print()
        
        # Separate closed and open issues
        closed_issues = [issue for issue in self.filtered_issues if issue.state == State.closed]
        open_issues = [issue for issue in self.filtered_issues if issue.state == State.open]
        
        print(f"Found {len(closed_issues)} closed issues and {len(open_issues)} open issues")
        
        if len(closed_issues) > 0:
            self.results = self._analyze_closed_issues(closed_issues)
        else:
            print("No closed issues found. Analyzing open issue aging instead...")
            self.results = self._analyze_open_issues(open_issues)
        
        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"{'='*80}")
        
        return self.results
    
    def _analyze_closed_issues(self, closed_issues: List[Issue]) -> Dict:
        """Analyze completion times for closed issues."""
        print("\n1. ANALYZING CLOSED ISSUES")
        print("-" * 40)
        
        # Calculate completion times
        completion_data = []
        for issue in closed_issues:
            completion_time = self._calculate_completion_time(issue)
            if completion_time is not None:
                completion_data.append({
                    'issue_number': issue.number,
                    'completion_time': completion_time,
                    'labels': self._get_issue_labels(issue),
                    'is_assigned': len(issue.assignees) > 0,
                    'creator': issue.creator,
                    'title': issue.title,
                    'url': issue.url
                })
        
        if not completion_data:
            print("No valid completion times found for closed issues.")
            return self._analyze_open_issues([issue for issue in self.filtered_issues if issue.state == State.open])
        
        df = pd.DataFrame(completion_data)
        completion_times = df['completion_time']
        
        # Basic statistics
        print(f"Number of closed issues: {len(df)}")
        print(f"Average completion time: {completion_times.mean():.1f} days")
        print(f"Median completion time: {completion_times.median():.1f} days")
        print(f"90th percentile: {completion_times.quantile(0.9):.1f} days")
        print(f"95th percentile: {completion_times.quantile(0.95):.1f} days")
        
        # Distribution buckets
        fast = len(completion_times[completion_times <= 7])
        normal = len(completion_times[(completion_times > 7) & (completion_times <= 30)])
        slow = len(completion_times[(completion_times > 30) & (completion_times <= 90)])
        stale = len(completion_times[completion_times > 90])
        total = len(completion_times)
        
        print(f"\nDistribution buckets:")
        print(f"  Fast (≤7 days): {fast} ({fast/total*100:.1f}%)")
        print(f"  Normal (≤30 days): {normal} ({normal/total*100:.1f}%)")
        print(f"  Slow (≤90 days): {slow} ({slow/total*100:.1f}%)")
        print(f"  Stale (>90 days): {stale} ({stale/total*100:.1f}%)")
        
        # Analysis by labels
        self._analyze_by_labels(df)
        
        # Analysis by assignment
        self._analyze_by_assignment(df)
        
        # Monthly trend analysis
        self._analyze_monthly_trends(closed_issues)
        
        # Create visualizations
        self._create_completion_visualizations(df)
        
        return {
            'completion_times': completion_times,
            'completion_df': df,
            'statistics': {
                'count': len(df),
                'mean': completion_times.mean(),
                'median': completion_times.median(),
                'p90': completion_times.quantile(0.9),
                'p95': completion_times.quantile(0.95),
                'fast': fast,
                'normal': normal,
                'slow': slow,
                'stale': stale
            }
        }
    
    def _analyze_open_issues(self, open_issues: List[Issue]) -> Dict:
        """Analyze aging patterns for open issues."""
        print("\n1. ANALYZING OPEN ISSUES (AGING ANALYSIS)")
        print("-" * 50)
        
        # Calculate ages
        aging_data = []
        for issue in open_issues:
            age = self._calculate_open_age(issue)
            if age is not None:
                aging_data.append({
                    'issue_number': issue.number,
                    'age': age,
                    'labels': self._get_issue_labels(issue),
                    'creator': issue.creator,
                    'title': issue.title,
                    'url': issue.url
                })
        
        if not aging_data:
            print("No valid aging data found for open issues.")
            return {}
        
        df = pd.DataFrame(aging_data)
        ages = df['age']
        
        # Basic statistics
        print(f"Number of open issues: {len(df)}")
        print(f"Mean age: {ages.mean():.1f} days")
        print(f"Median age: {ages.median():.1f} days")
        print(f"90th percentile: {ages.quantile(0.9):.1f} days")
        
        # Age buckets
        recent = len(ages[ages <= 7])
        normal = len(ages[(ages > 7) & (ages <= 30)])
        old = len(ages[(ages > 30) & (ages <= 90)])
        stale = len(ages[ages > 90])
        total = len(ages)
        
        print(f"\nAge buckets:")
        print(f"  Recent (≤7 days): {recent} ({recent/total*100:.1f}%)")
        print(f"  Normal (≤30 days): {normal} ({normal/total*100:.1f}%)")
        print(f"  Old (≤90 days): {old} ({old/total*100:.1f}%)")
        print(f"  Stale (>90 days): {stale} ({stale/total*100:.1f}%)")
        
        # Top 10 stalest labels
        self._analyze_stalest_labels(df)
        
        # Top 15 oldest issues
        self._show_oldest_issues(df)
        
        # Create visualizations
        self._create_aging_visualizations(df)
        
        return {
            'ages': ages,
            'aging_df': df,
            'statistics': {
                'count': len(df),
                'mean': ages.mean(),
                'median': ages.median(),
                'p90': ages.quantile(0.9),
                'recent': recent,
                'normal': normal,
                'old': old,
                'stale': stale
            }
        }
    
    def _analyze_by_labels(self, df: pd.DataFrame):
        """Analyze completion times by labels."""
        print(f"\n2. COMPLETION TIME BY LABELS (Top 10)")
        print("-" * 45)
        
        # Flatten labels and calculate median completion time
        label_data = []
        for _, row in df.iterrows():
            for label in row['labels']:
                label_data.append({
                    'label': label,
                    'completion_time': row['completion_time']
                })
        
        if not label_data:
            print("No label data available.")
            return
        
        label_df = pd.DataFrame(label_data)
        label_stats = label_df.groupby('label')['completion_time'].agg(['median', 'count']).round(1)
        label_stats = label_stats[label_stats['count'] >= 3].sort_values('median', ascending=True).head(10)
        
        for label, stats in label_stats.iterrows():
            print(f"  {label}: {stats['median']:.1f} days median ({stats['count']} issues)")
    
    def _analyze_by_assignment(self, df: pd.DataFrame):
        """Analyze completion times by assignment status."""
        print(f"\n3. COMPLETION TIME BY ASSIGNMENT")
        print("-" * 35)
        
        assigned = df[df['is_assigned']]['completion_time']
        unassigned = df[~df['is_assigned']]['completion_time']
        
        print(f"  Assigned issues: {assigned.median():.1f} days median ({len(assigned)} issues)")
        print(f"  Unassigned issues: {unassigned.median():.1f} days median ({len(unassigned)} issues)")
    
    def _analyze_monthly_trends(self, closed_issues: List[Issue]):
        """Analyze monthly completion time trends."""
        print(f"\n4. MONTHLY COMPLETION TIME TRENDS")
        print("-" * 40)
        
        # Prepare monthly data
        monthly_data = []
        for issue in closed_issues:
            completion_time = self._calculate_completion_time(issue)
            created_date = self._to_utc_timestamp(str(issue.created_date)) if issue.created_date else None
            
            if completion_time is not None and created_date:
                monthly_data.append({
                    'created_date': created_date,
                    'completion_time': completion_time,
                    'year_month': created_date.strftime('%Y-%m')
                })
        
        if not monthly_data:
            print("No monthly trend data available.")
            return
        
        monthly_df = pd.DataFrame(monthly_data)
        monthly_stats = monthly_df.groupby('year_month')['completion_time'].agg(['median', 'count']).reset_index()
        monthly_stats = monthly_stats[monthly_stats['count'] >= 3]  # Only months with 3+ issues
        
        if len(monthly_stats) < 2:
            print("Not enough data points for trend analysis.")
            return
        
        # Create trend plot
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_stats['year_month'], monthly_stats['median'], marker='o', linewidth=2)
        plt.title('Monthly Median Completion Time Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Median Completion Time (days)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Trend analysis: {len(monthly_stats)} months of data")
        print(f"Latest median: {monthly_stats.iloc[-1]['median']:.1f} days")
        print(f"Earliest median: {monthly_stats.iloc[0]['median']:.1f} days")
    
    def _analyze_stalest_labels(self, df: pd.DataFrame):
        """Analyze stalest labels for open issues."""
        print(f"\n2. STALEST LABELS BY MEDIAN AGE (Top 10)")
        print("-" * 50)
        
        # Flatten labels and calculate median age
        label_data = []
        for _, row in df.iterrows():
            for label in row['labels']:
                label_data.append({
                    'label': label,
                    'age': row['age']
                })
        
        if not label_data:
            print("No label data available.")
            return
        
        label_df = pd.DataFrame(label_data)
        label_stats = label_df.groupby('label')['age'].agg(['median', 'count']).round(1)
        label_stats = label_stats[label_stats['count'] >= 2].sort_values('median', ascending=False).head(10)
        
        for label, stats in label_stats.iterrows():
            print(f"  {label}: {stats['median']:.1f} days median ({stats['count']} issues)")
    
    def _show_oldest_issues(self, df: pd.DataFrame):
        """Show top 15 oldest open issues."""
        print(f"\n3. TOP 15 OLDEST OPEN ISSUES")
        print("-" * 35)
        
        oldest = df.nlargest(15, 'age')
        
        for _, row in oldest.iterrows():
            print(f"  #{row['issue_number']}: {row['age']:.0f} days - {row['title'][:60]}...")
            print(f"    {row['url']}")
    
    def _create_completion_visualizations(self, df: pd.DataFrame):
        """Create visualizations for completion time analysis."""
        print(f"\n5. CREATING VISUALIZATIONS")
        print("-" * 30)
        
        # Histogram of completion times
        plt.figure(figsize=(10, 6))
        plt.hist(df['completion_time'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Issue Completion Times', fontsize=14, fontweight='bold')
        plt.xlabel('Completion Time (days)')
        plt.ylabel('Number of Issues')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Box plot by assignment
        plt.figure(figsize=(8, 6))
        assigned_data = [df[df['is_assigned']]['completion_time'], df[~df['is_assigned']]['completion_time']]
        plt.boxplot(assigned_data, labels=['Assigned', 'Unassigned'])
        plt.title('Completion Time by Assignment Status', fontsize=14, fontweight='bold')
        plt.ylabel('Completion Time (days)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Top 10 labels box plot
        label_data = []
        for _, row in df.iterrows():
            for label in row['labels']:
                label_data.append({
                    'label': label,
                    'completion_time': row['completion_time']
                })
        
        if label_data:
            label_df = pd.DataFrame(label_data)
            label_counts = label_df['label'].value_counts()
            top_labels = label_counts.head(10).index.tolist()
            
            if len(top_labels) > 0:
                plt.figure(figsize=(12, 8))
                label_data_for_plot = [label_df[label_df['label'] == label]['completion_time'].values 
                                     for label in top_labels]
                plt.boxplot(label_data_for_plot, labels=top_labels)
                plt.title('Completion Time by Top 10 Labels', fontsize=14, fontweight='bold')
                plt.xlabel('Label')
                plt.ylabel('Completion Time (days)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
    
    def _create_aging_visualizations(self, df: pd.DataFrame):
        """Create visualizations for aging analysis."""
        print(f"\n4. CREATING VISUALIZATIONS")
        print("-" * 30)
        
        # Histogram of open issue ages
        plt.figure(figsize=(10, 6))
        plt.hist(df['age'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Open Issue Ages', fontsize=14, fontweight='bold')
        plt.xlabel('Age (days)')
        plt.ylabel('Number of Issues')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Box plot of open issue ages
        plt.figure(figsize=(8, 6))
        plt.boxplot([df['age']], labels=['Open Issues'])
        plt.title('Distribution of Open Issue Ages (Box Plot)', fontsize=14, fontweight='bold')
        plt.ylabel('Age (days)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def run(issues: List[Issue] = None, config_dict: Dict = None) -> Dict:
    """
    Main entry point for the completion analysis.
    
    Args:
        issues: List of Issue objects (optional, will load if not provided)
        config_dict: Configuration dictionary (optional)
    
    Returns:
        Dictionary containing analysis results
    """
    analysis = CompletionAnalysis()
    return analysis.run()


if __name__ == '__main__':
    # Run the analysis when executed directly
    run()