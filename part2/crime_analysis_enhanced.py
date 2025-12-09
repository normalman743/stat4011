import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set Chinese display and plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output folder
output_dir = Path('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/analysis_output2')
output_dir.mkdir(exist_ok=True)
charts_dir = output_dir / 'charts'
charts_dir.mkdir(exist_ok=True)
stats_dir = output_dir / 'statistics'
stats_dir.mkdir(exist_ok=True)

print("📁 Output directories:")
print(f"   Charts: {charts_dir}")
print(f"   Statistics: {stats_dir}")

# Read data
df = pd.read_csv('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2020_to_Present_20250929.csv')

print("\n=== Dataset Basic Information ===")
print(f"Data shape: {df.shape}")
print(f"Data time range: {df['DATE OCC'].min()} to {df['DATE OCC'].max()}")

# ============================================================================
# === Feature Engineering ===
# ============================================================================

print("\n" + "="*80)
print("=== 🔧 Feature Engineering ===")
print("="*80)

# Time features
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month
df['MonthName'] = df['DATE OCC'].dt.strftime('%B')  # Full month name
df['Quarter'] = df['DATE OCC'].dt.quarter
df['QuarterName'] = 'Q' + df['Quarter'].astype(str)  # Q1, Q2, Q3, Q4
df['DayOfWeek'] = df['DATE OCC'].dt.dayofweek
df['DayName'] = df['DATE OCC'].dt.day_name()
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['WeekendLabel'] = df['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
df['WeekOfYear'] = df['DATE OCC'].dt.isocalendar().week

df['Hour'] = df['TIME OCC'] // 100
df['TimePeriod'] = pd.cut(df['Hour'], 
                          bins=[-1, 5, 11, 17, 21, 24],
                          labels=['Late Night (0-5)', 'Morning (6-11)', 'Afternoon (12-17)', 'Evening (18-21)', 'Night (22-24)'])

# Crime type features
df['CrimeCategory'] = df['Part 1-2'].map({1: 'Felony', 2: 'Misdemeanor'})
df['Crime_Type_Code'] = df['CrimeCategory'] + '_' + df['Crm Cd'].astype(str)

# Geographic features
df['UAC'] = df['AREA'].astype(str) + '_' + df['Rpt Dist No'].astype(str)
df['Area_Crime'] = df['AREA NAME'] + '_' + df['CrimeCategory']

# Victim features
df['AgeGroup'] = pd.cut(df['Vict Age'], 
                        bins=[0, 18, 35, 50, 65, 100], 
                        labels=['0-17', '18-34', '35-49', '50-64', '65+'])
df['VictimProfile'] = (df['Vict Sex'].fillna('U') + '_' + 
                       df['Vict Descent'].fillna('U') + '_' + 
                       df['AgeGroup'].astype(str))

# Combined features
df['YearMonth'] = df['DATE OCC'].dt.to_period('M').astype(str)
df['YearQuarter'] = df['Year'].astype(str) + 'Q' + df['Quarter'].astype(str)
df['Time_Crime'] = df['TimePeriod'].astype(str) + '_' + df['CrimeCategory']
df['Month_Crime'] = df['MonthName'] + '_' + df['CrimeCategory']
df['Weekend_Crime'] = df['WeekendLabel'] + '_' + df['CrimeCategory']

print("✓ Feature engineering completed!")

# ============================================================================
# === Analysis Configuration ===
# ============================================================================

# Define all analyzable dimensions - prioritize descriptive fields
ANALYSIS_CONFIG = {
    'Time Dimensions': {
        'Year': 'Year',
        'MonthName': 'Month',
        'QuarterName': 'Quarter',
        'DayName': 'Day of Week',
        'TimePeriod': 'Time Period',
        'Hour': 'Hour',
        'YearMonth': 'Year-Month',
        'YearQuarter': 'Year-Quarter',
        'WeekOfYear': 'Week of Year',
        'WeekendLabel': 'Weekday/Weekend'
    },
    'Geographic Dimensions': {
        'AREA NAME': 'Area Name',
        'Rpt Dist No': 'Reporting District',
    },
    'Crime Types': {
        'CrimeCategory': 'Crime Category',
        'Crm Cd Desc': 'Crime Description',
    },
    'Victims': {
        'Vict Sex': 'Gender',
        'Vict Descent': 'Descent',
        'AgeGroup': 'Age Group',
    },
    'Locations': {
        'Premis Desc': 'Premise Description',
    },
    'Weapons': {
        'Weapon Desc': 'Weapon Description',
    },
    'Status': {
        'Status Desc': 'Status Description',
    },
    'Combined Dimensions': {
        'Area_Crime': 'Area + Crime',
        'Time_Crime': 'Time + Crime',
        'Month_Crime': 'Month + Crime',
        'Weekend_Crime': 'Weekend + Crime'
    }
}

# ============================================================================
# === Core Analysis Functions ===
# ============================================================================

class CrimeAnalyzer:
    def __init__(self, df, charts_dir, stats_dir):
        self.df = df
        self.charts_dir = charts_dir
        self.stats_dir = stats_dir
        
        # Define chart configurations for different field types
        self.chart_configs = {
            # Time dimensions - use time series style charts
            'Year': 'timeseries_bar',
            'MonthName': 'seasonal',
            'QuarterName': 'simple_comparison',
            'DayName': 'weekly_pattern',
            'TimePeriod': 'daily_pattern',
            'Hour': 'hourly_pattern',
            'YearMonth': 'trend',
            'YearQuarter': 'trend',
            'WeekOfYear': 'distribution',
            'WeekendLabel': 'simple_comparison',
            
            # Geographic - use ranking charts
            'AREA NAME': 'ranking',
            'Rpt Dist No': 'distribution',
            
            # Crime types - use comparison charts
            'CrimeCategory': 'simple_comparison',
            'Crm Cd Desc': 'top_ranking',
            
            # Victims - use demographic charts
            'Vict Sex': 'demographic',
            'Vict Descent': 'demographic',
            'AgeGroup': 'age_distribution',
            
            # Locations - use ranking
            'Premis Desc': 'top_ranking',
            
            # Weapons - use ranking
            'Weapon Desc': 'top_ranking',
            
            # Status - use simple comparison
            'Status Desc': 'simple_comparison',
            
            # Combined - use heatmap style
            'Area_Crime': 'combined',
            'Time_Crime': 'combined',
            'Month_Crime': 'combined',
            'Weekend_Crime': 'simple_comparison',
        }
    
    def analyze_single_dimension(self, column, title, top_n=20, save=True):
        """Analyze single dimension"""
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print(f"{'='*80}")
        
        # Statistics
        stats = self.df[column].value_counts().head(top_n)
        pct = self.df[column].value_counts(normalize=True).head(top_n) * 100
        
        result = pd.DataFrame({
            'Category': stats.index,
            'Count': stats.values,
            'Percentage (%)': pct.values.round(2)
        })
        
        print(result.to_string(index=False))
        print(f"\n📈 Total categories: {self.df[column].nunique()}")
        print(f"📈 Total records: {self.df[column].count()}")
        print(f"📈 Missing values: {self.df[column].isna().sum()}")
        
        if save:
            # Save statistics
            result.to_csv(self.stats_dir / f'{column}_stats.csv', index=False, encoding='utf-8-sig')
            
            # Create charts
            self._create_charts(column, title, stats, top_n)
        
        return result
    
    def _create_charts(self, column, title, stats, top_n):
        """Create various charts based on field type"""
        chart_type = self.chart_configs.get(column, 'default')
        
        if chart_type == 'simple_comparison':
            self._create_simple_comparison(column, title, stats)
        elif chart_type == 'ranking':
            self._create_ranking_charts(column, title, stats)
        elif chart_type == 'top_ranking':
            self._create_top_ranking(column, title, stats)
        elif chart_type == 'demographic':
            self._create_demographic_charts(column, title, stats)
        elif chart_type == 'age_distribution':
            self._create_age_distribution(column, title, stats)
        elif chart_type == 'seasonal':
            self._create_seasonal_chart(column, title, stats)
        elif chart_type == 'weekly_pattern':
            self._create_weekly_pattern(column, title, stats)
        elif chart_type == 'daily_pattern':
            self._create_daily_pattern(column, title, stats)
        elif chart_type == 'hourly_pattern':
            self._create_hourly_pattern(column, title, stats)
        elif chart_type == 'timeseries_bar':
            self._create_timeseries_bar(column, title, stats)
        elif chart_type == 'trend':
            self._create_trend_chart(column, title, stats)
        elif chart_type == 'combined':
            self._create_combined_chart(column, title, stats)
        else:
            self._create_default_charts(column, title, stats, top_n)
    
    def _create_simple_comparison(self, column, title, stats):
        """For binary or few categories - pie + bar"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title} - Distribution', fontsize=16, fontweight='bold')
        
        labels = self._format_labels(stats.index, column)
        colors = sns.color_palette("husl", len(stats))
        
        # Pie chart
        ax1 = axes[0]
        wedges, texts, autotexts = ax1.pie(stats.values, labels=labels, autopct='%1.1f%%', 
                                             startangle=90, colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('Percentage Distribution', fontsize=12)
        
        # Bar chart with values
        ax2 = axes[1]
        bars = ax2.bar(range(len(stats)), stats.values, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(stats)))
        ax2.set_xticklabels(labels, rotation=0 if len(stats) <= 4 else 45, ha='right')
        ax2.set_ylabel('Count')
        ax2.set_title('Count Comparison', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_ranking_charts(self, column, title, stats):
        """For geographic areas - horizontal bar + map-style"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{title} - Ranking Analysis', fontsize=16, fontweight='bold')
        
        labels = self._format_labels(stats.index, column)
        
        # Horizontal ranking
        ax1 = axes[0]
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stats)))
        ax1.barh(range(len(stats)), stats.values, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(stats)))
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('Crime Count')
        ax1.set_title('Area Ranking (Descending)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # Percentage comparison
        ax2 = axes[1]
        pct = (stats / stats.sum() * 100).values
        bars = ax2.bar(range(len(stats)), pct, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(stats)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Percentage Share', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_top_ranking(self, column, title, stats):
        """For crime types with many categories - top 15 + others"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'{title} - Top Rankings', fontsize=16, fontweight='bold')
        
        # Top 15 horizontal bar
        ax1 = axes[0]
        top15 = stats.head(15).sort_values()
        labels = self._format_labels(top15.index, column)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top15)))
        
        bars = ax1.barh(range(len(top15)), top15.values, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(top15)))
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Count')
        ax1.set_title('Top 15 Categories', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {int(width):,}',
                    ha='left', va='center', fontsize=8)
        
        # Cumulative percentage
        ax2 = axes[1]
        cumsum_pct = (stats.cumsum() / stats.sum() * 100)
        ax2.plot(range(len(cumsum_pct)), cumsum_pct.values, 
                marker='o', linewidth=2, color='steelblue', markersize=4)
        ax2.fill_between(range(len(cumsum_pct)), cumsum_pct.values, alpha=0.3)
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.set_title('Cumulative Distribution (Pareto Analysis)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=80, color='r', linestyle='--', linewidth=2, label='80% Line')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_demographic_charts(self, column, title, stats):
        """For demographic data - pie + bar"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title} - Distribution', fontsize=16, fontweight='bold')
        
        labels = self._format_labels(stats.index, column)
        
        # Donut chart
        ax1 = axes[0]
        colors = sns.color_palette("Set2", len(stats))
        wedges, texts, autotexts = ax1.pie(stats.values, labels=labels, autopct='%1.1f%%',
                                             startangle=90, colors=colors, pctdistance=0.85)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax1.set_title('Percentage Distribution', fontsize=12)
        
        # Bar with percentage
        ax2 = axes[1]
        pct = (stats / stats.sum() * 100).values
        bars = ax2.bar(range(len(stats)), pct, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_xticks(range(len(stats)))
        ax2.set_xticklabels(labels, rotation=0)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Comparison', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%\n({int(stats.values[i]):,})',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_age_distribution(self, column, title, stats):
        """For age groups - pyramid style"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title} - Age Distribution', fontsize=16, fontweight='bold')
        
        # Sort by age order
        age_order = ['0-17', '18-34', '35-49', '50-64', '65+']
        ordered_stats = stats.reindex([x for x in age_order if x in stats.index])
        
        # Horizontal bar
        ax1 = axes[0]
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(ordered_stats)))
        ax1.barh(range(len(ordered_stats)), ordered_stats.values, 
                color=colors, edgecolor='black')
        ax1.set_yticks(range(len(ordered_stats)))
        ax1.set_yticklabels(ordered_stats.index)
        ax1.set_xlabel('Count')
        ax1.set_title('Count by Age Group', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # Percentage with line
        ax2 = axes[1]
        pct = (ordered_stats / ordered_stats.sum() * 100).values
        ax2.bar(range(len(ordered_stats)), pct, color=colors, edgecolor='black', alpha=0.7)
        ax2.plot(range(len(ordered_stats)), pct, marker='o', linewidth=2, 
                color='darkred', markersize=8)
        ax2.set_xticks(range(len(ordered_stats)))
        ax2.set_xticklabels(ordered_stats.index)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Percentage Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(pct):
            ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_seasonal_chart(self, column, title, stats):
        """For months - circular/seasonal pattern"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'{title} - Seasonal Pattern', fontsize=16, fontweight='bold')
        
        # Order by month
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        ordered_stats = stats.reindex([m for m in month_order if m in stats.index])
        
        # Line + bar chart
        ax1 = axes[0]
        x = range(len(ordered_stats))
        ax1.bar(x, ordered_stats.values, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.plot(x, ordered_stats.values, marker='o', linewidth=2, 
                color='darkblue', markersize=8, label='Trend')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m[:3] for m in ordered_stats.index], rotation=0)
        ax1.set_ylabel('Count')
        ax1.set_title('Monthly Distribution', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Polar chart
        ax2 = plt.subplot(122, projection='polar')
        theta = np.linspace(0, 2 * np.pi, len(ordered_stats), endpoint=False)
        values = ordered_stats.values
        theta = np.concatenate((theta, [theta[0]]))
        values = np.concatenate((values, [values[0]]))
        
        ax2.plot(theta, values, linewidth=2, color='darkblue', marker='o', markersize=8)
        ax2.fill(theta, values, alpha=0.25, color='skyblue')
        ax2.set_xticks(theta[:-1])
        ax2.set_xticklabels([m[:3] for m in ordered_stats.index])
        ax2.set_title('Circular Pattern', fontsize=12, pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_weekly_pattern(self, column, title, stats):
        """For days of week"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title} - Weekly Pattern', fontsize=16, fontweight='bold')
        
        # Order by day
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ordered_stats = stats.reindex([d for d in day_order if d in stats.index])
        
        # Bar chart
        ax1 = axes[0]
        colors = ['#FF6B6B' if d in ['Saturday', 'Sunday'] else '#4ECDC4' for d in ordered_stats.index]
        ax1.bar(range(len(ordered_stats)), ordered_stats.values, color=colors, edgecolor='black')
        ax1.set_xticks(range(len(ordered_stats)))
        ax1.set_xticklabels([d[:3] for d in ordered_stats.index])
        ax1.set_ylabel('Count')
        ax1.set_title('Crimes by Day of Week', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add weekend highlight
        weekend_avg = ordered_stats[['Saturday', 'Sunday']].mean()
        weekday_avg = ordered_stats[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean()
        ax1.axhline(y=weekend_avg, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Weekend Avg: {weekend_avg:,.0f}')
        ax1.axhline(y=weekday_avg, color='#4ECDC4', linestyle='--', linewidth=2, label=f'Weekday Avg: {weekday_avg:,.0f}')
        ax1.legend()
        
        # Percentage comparison
        ax2 = axes[1]
        pct = (ordered_stats / ordered_stats.sum() * 100).values
        ax2.bar(range(len(ordered_stats)), pct, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(ordered_stats)))
        ax2.set_xticklabels([d[:3] for d in ordered_stats.index])
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Percentage Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=100/7, color='gray', linestyle=':', linewidth=2, label='Equal Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_daily_pattern(self, column, title, stats):
        """For time periods of day"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title} - Daily Crime Pattern', fontsize=16, fontweight='bold')
        
        # Order by time
        period_order = ['Late Night (0-5)', 'Morning (6-11)', 'Afternoon (12-17)', 
                       'Evening (18-21)', 'Night (22-24)']
        ordered_stats = stats.reindex([p for p in period_order if p in stats.index])
        
        labels = ['Late Night\n(0-5)', 'Morning\n(6-11)', 'Afternoon\n(12-17)', 
                 'Evening\n(18-21)', 'Night\n(22-24)']
        
        # Bar with gradient
        ax1 = axes[0]
        colors = ['#2C3E50', '#F39C12', '#E74C3C', '#9B59B6', '#34495E']
        bars = ax1.bar(range(len(ordered_stats)), ordered_stats.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        ax1.set_xticks(range(len(ordered_stats)))
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel('Count')
        ax1.set_title('Crimes by Time Period', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Pie chart
        ax2 = axes[1]
        ax2.pie(ordered_stats.values, labels=labels, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title('Time Period Distribution', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_hourly_pattern(self, column, title, stats):
        """For 24-hour pattern"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(f'{title} - 24-Hour Crime Pattern', fontsize=16, fontweight='bold')
        
        # Sort by hour
        ordered_stats = stats.sort_index()
        
        # Line + area chart
        ax1 = axes[0]
        ax1.plot(ordered_stats.index, ordered_stats.values, 
                marker='o', linewidth=2, color='darkred', markersize=6)
        ax1.fill_between(ordered_stats.index, ordered_stats.values, alpha=0.3, color='red')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Count')
        ax1.set_title('Hourly Crime Distribution', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # Add time period backgrounds
        ax1.axvspan(0, 6, alpha=0.1, color='blue', label='Late Night')
        ax1.axvspan(6, 12, alpha=0.1, color='yellow', label='Morning')
        ax1.axvspan(12, 18, alpha=0.1, color='orange', label='Afternoon')
        ax1.axvspan(18, 22, alpha=0.1, color='purple', label='Evening')
        ax1.axvspan(22, 24, alpha=0.1, color='darkblue', label='Night')
        ax1.legend(loc='upper left')
        
        # Bar chart
        ax2 = axes[1]
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(ordered_stats)))
        ax2.bar(ordered_stats.index, ordered_stats.values, color=colors, edgecolor='black')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Count')
        ax2.set_title('Crime Count by Hour', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(range(0, 24))
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_timeseries_bar(self, column, title, stats):
        """For year trend"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{title} - Trend Analysis', fontsize=16, fontweight='bold')
        
        ordered_stats = stats.sort_index()
        
        # Bar chart
        ax1 = axes[0]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ordered_stats)))
        bars = ax1.bar(range(len(ordered_stats)), ordered_stats.values, 
                      color=colors, edgecolor='black')
        ax1.set_xticks(range(len(ordered_stats)))
        ax1.set_xticklabels(ordered_stats.index)
        ax1.set_ylabel('Count')
        ax1.set_title('Annual Crime Count', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Line trend
        ax2 = axes[1]
        ax2.plot(range(len(ordered_stats)), ordered_stats.values, 
                marker='o', linewidth=3, markersize=10, color='darkblue')
        ax2.set_xticks(range(len(ordered_stats)))
        ax2.set_xticklabels(ordered_stats.index)
        ax2.set_ylabel('Count')
        ax2.set_title('Trend Line', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add trend indication
        if len(ordered_stats) > 1:
            z = np.polyfit(range(len(ordered_stats)-1), ordered_stats.values[:-1], 1)
            p = np.poly1d(z)
            ax2.plot(range(len(ordered_stats)-1), p(range(len(ordered_stats)-1)), 
                    "r--", linewidth=2, alpha=0.7, label='Trend (excl. 2025)')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_trend_chart(self, column, title, stats):
        """For time series trend"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(f'{title} - Time Series Trend', fontsize=16, fontweight='bold')
        
        ordered_stats = stats.sort_index()
        
        # Line chart
        ax1 = axes[0]
        ax1.plot(range(len(ordered_stats)), ordered_stats.values, 
                marker='o', linewidth=2, markersize=4, color='steelblue')
        ax1.fill_between(range(len(ordered_stats)), ordered_stats.values, alpha=0.3)
        ax1.set_ylabel('Count')
        ax1.set_title('Trend Over Time', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=ordered_stats.mean(), color='r', linestyle='--', 
                   linewidth=2, label=f'Average: {ordered_stats.mean():,.0f}')
        ax1.legend()
        
        # Moving average
        ax2 = axes[1]
        if len(ordered_stats) > 7:
            ma = ordered_stats.rolling(window=min(7, len(ordered_stats)//2), center=True).mean()
            ax2.plot(range(len(ordered_stats)), ordered_stats.values, 
                    alpha=0.4, linewidth=1, label='Raw Data', color='gray')
            ax2.plot(range(len(ma)), ma.values, linewidth=3, 
                    color='darkred', label=f'{min(7, len(ordered_stats)//2)}-Period Moving Avg')
            ax2.set_ylabel('Count')
            ax2.set_title('Smoothed Trend', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_combined_chart(self, column, title, stats):
        """For combined dimensions"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'{title} - Combined Analysis', fontsize=16, fontweight='bold')
        
        # Top 15 horizontal bar
        ax1 = axes[0]
        top15 = stats.head(15).sort_values()
        labels = self._format_labels(top15.index, column)
        colors = plt.cm.Spectral(np.linspace(0.2, 0.9, len(top15)))
        
        ax1.barh(range(len(top15)), top15.values, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(top15)))
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Count')
        ax1.set_title(f'Top 15 Categories', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Grouped visualization
        ax2 = axes[1]
        top20 = stats.head(20)
        ax2.bar(range(len(top20)), top20.values, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Count')
        ax2.set_title('Top 20 Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _create_default_charts(self, column, title, stats, top_n):
        """Default chart for distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{title} - Distribution Analysis', fontsize=16, fontweight='bold')
        
        labels = self._format_labels(stats.index, column)
        
        # Bar chart
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(stats)), stats.values, color='steelblue', edgecolor='black')
        ax1.set_title(f'Top {top_n} Distribution (Bar Chart)', fontsize=12)
        ax1.set_xlabel('')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(stats)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Horizontal bar chart
        ax2 = axes[0, 1]
        sorted_stats = stats.sort_values()
        sorted_labels = self._format_labels(sorted_stats.index, column)
        ax2.barh(range(len(sorted_stats)), sorted_stats.values, color='coral', edgecolor='black')
        ax2.set_title(f'Top {top_n} Distribution (Horizontal Bar)', fontsize=12)
        ax2.set_xlabel('Count')
        ax2.set_ylabel('')
        ax2.set_yticks(range(len(sorted_stats)))
        ax2.set_yticklabels(sorted_labels)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Pie chart
        ax3 = axes[1, 0]
        if len(stats) <= 10:
            pie_labels = self._format_labels(stats.index, column)
            ax3.pie(stats.values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Top {min(len(stats), 10)} Percentage (Pie Chart)', fontsize=12)
        else:
            top10 = stats.iloc[:10]
            other_sum = stats.iloc[10:].sum()
            plot_data = pd.concat([top10, pd.Series({'Other': other_sum})])
            pie_labels = self._format_labels(plot_data.index, column)
            ax3.pie(plot_data.values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Top 10 + Other Percentage (Pie Chart)', fontsize=12)
        
        # Cumulative distribution chart
        ax4 = axes[1, 1]
        cumsum_pct = (stats.cumsum() / stats.sum() * 100)
        ax4.plot(range(len(cumsum_pct)), cumsum_pct.values, marker='o', color='green', linewidth=2)
        ax4.set_title('Cumulative Distribution Curve', fontsize=12)
        ax4.set_xlabel('Rank')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=80, color='r', linestyle='--', label='80% Line')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Chart saved: {column}_distribution.png")
    
    def _format_labels(self, labels, column):
        """Format labels for better display"""
        formatted = []
        for label in labels:
            # Truncate long labels
            label_str = str(label)
            if len(label_str) > 30:
                label_str = label_str[:27] + '...'
            formatted.append(label_str)
        return formatted
    
    def analyze_crosstab(self, col1, col2, title, top_n1=10, top_n2=10, save=True):
        """Cross-analysis"""
        print(f"\n{'='*80}")
        print(f"🔗 {title}")
        print(f"{'='*80}")
        
        # Filter Top N
        top_cat1 = self.df[col1].value_counts().head(top_n1).index
        top_cat2 = self.df[col2].value_counts().head(top_n2).index
        
        filtered_df = self.df[
            (self.df[col1].isin(top_cat1)) & 
            (self.df[col2].isin(top_cat2))
        ]
        
        # Crosstab
        crosstab = pd.crosstab(filtered_df[col1], filtered_df[col2], margins=True)
        print(crosstab)
        
        # Percentage crosstab
        crosstab_pct = pd.crosstab(filtered_df[col1], filtered_df[col2], normalize='all') * 100
        print(f"\nPercentage (%):")
        print(crosstab_pct.round(2))
        
        if save:
            # Save statistics
            crosstab.to_csv(self.stats_dir / f'{col1}_vs_{col2}_crosstab.csv', encoding='utf-8-sig')
            crosstab_pct.to_csv(self.stats_dir / f'{col1}_vs_{col2}_crosstab_pct.csv', encoding='utf-8-sig')
            
            # Create heatmap
            self._create_heatmap(crosstab.iloc[:-1, :-1], title, f'{col1}_vs_{col2}')
        
        return crosstab
    
    def _create_heatmap(self, data, title, filename):
        """Create heatmap"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'{title}', fontsize=16, fontweight='bold')
        
        # Format row and column labels
        row_labels = [str(idx)[:30] for idx in data.index]
        col_labels = [str(col)[:30] for col in data.columns]
        
        # Count heatmap
        sns.heatmap(data, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], 
                    cbar_kws={'label': 'Count'},
                    xticklabels=col_labels, yticklabels=row_labels)
        axes[0].set_title('Count Distribution', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Percentage heatmap
        data_pct = data / data.sum().sum() * 100
        sns.heatmap(data_pct, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
                    cbar_kws={'label': 'Percentage (%)'},
                    xticklabels=col_labels, yticklabels=row_labels)
        axes[1].set_title('Percentage Distribution (%)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{filename}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Heatmap saved: {filename}_heatmap.png")
    
    def analyze_time_series(self, time_col, title, save=True):
        """Time series analysis"""
        print(f"\n{'='*80}")
        print(f"📈 {title}")
        print(f"{'='*80}")
        
        ts_data = self.df[time_col].value_counts().sort_index()
        
        print(f"Time range: {ts_data.index.min()} to {ts_data.index.max()}")
        print(f"Average: {ts_data.mean():.2f}")
        print(f"Median: {ts_data.median():.2f}")
        print(f"Standard deviation: {ts_data.std():.2f}")
        print(f"Maximum: {ts_data.max()} (Time: {ts_data.idxmax()})")
        print(f"Minimum: {ts_data.min()} (Time: {ts_data.idxmin()})")
        
        if save:
            ts_data.to_csv(self.stats_dir / f'{time_col}_timeseries.csv', encoding='utf-8-sig')
            self._create_timeseries_chart(ts_data, title, time_col)
        
        return ts_data
    
    def _create_timeseries_chart(self, ts_data, title, filename):
        """Create time series chart"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(f'{title} - Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Line chart
        ax1 = axes[0]
        ax1.plot(ts_data.index, ts_data.values, marker='o', linewidth=2, markersize=4)
        ax1.set_title('Time Trend', fontsize=12)
        ax1.set_ylabel('Crime Count')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=ts_data.mean(), color='r', linestyle='--', label=f'Average: {ts_data.mean():.0f}')
        ax1.legend()
        
        # Moving average
        ax2 = axes[1]
        if len(ts_data) > 7:
            ma7 = ts_data.rolling(window=7, center=True).mean()
            ax2.plot(ts_data.index, ts_data.values, alpha=0.3, label='Raw Data')
            ax2.plot(ma7.index, ma7.values, linewidth=2, color='red', label='7-Period Moving Average')
            ax2.set_title('Moving Average Trend', fontsize=12)
            ax2.set_ylabel('Crime Count')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{filename}_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Time series chart saved: {filename}_timeseries.png")

# ============================================================================
# === Interactive Menu ===
# ============================================================================

def show_menu():
    """Display analysis menu"""
    print("\n" + "="*80)
    print("🎯 Crime Data Analysis System")
    print("="*80)
    print("\n【Analysis Modes】")
    print("1. 🚀 Full Analysis (Generate all statistics and charts)")
    print("2. 🎨 Single Dimension Analysis (Select specific field)")
    print("3. 🔗 Cross Analysis (Two fields cross)")
    print("4. 📈 Time Series Analysis")
    print("5. 📊 Quick Summary")
    print("0. Exit")
    print("="*80)

def run_full_analysis(analyzer):
    """Run full analysis"""
    print("\n🚀 Starting full analysis...\n")
    
    # Single dimension analysis
    for category, fields in ANALYSIS_CONFIG.items():
        print(f"\n{'='*80}")
        print(f"📁 {category}")
        print(f"{'='*80}")
        for field, desc in fields.items():
            if field in analyzer.df.columns:
                analyzer.analyze_single_dimension(field, desc, top_n=20)
    
    # Key cross analyses - use descriptive fields
    key_crosstabs = [
        ('CrimeCategory', 'TimePeriod', 'Crime Category × Time Period'),
        ('CrimeCategory', 'MonthName', 'Crime Category × Month'),
        ('AREA NAME', 'CrimeCategory', 'Area × Crime Category'),
        ('DayName', 'CrimeCategory', 'Day of Week × Crime Category'),
        ('AgeGroup', 'CrimeCategory', 'Age Group × Crime Category'),
        ('Premis Desc', 'CrimeCategory', 'Premise × Crime Category'),
    ]
    
    print(f"\n{'='*80}")
    print("🔗 Cross Analysis")
    print(f"{'='*80}")
    for col1, col2, title in key_crosstabs:
        analyzer.analyze_crosstab(col1, col2, title)
    
    # Time series analysis
    time_series_fields = ['YearMonth', 'Year', 'Quarter']
    print(f"\n{'='*80}")
    print("📈 Time Series Analysis")
    print(f"{'='*80}")
    for field in time_series_fields:
        analyzer.analyze_time_series(field, f'{field} Time Series')
    
    print("\n✅ Full analysis completed!")
    print(f"📁 Results saved in: {output_dir}")

def interactive_mode(analyzer):
    """Interactive mode"""
    while True:
        show_menu()
        choice = input("\nChoose (0-5): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            run_full_analysis(analyzer)
        elif choice == '2':
            # Display available fields
            print("\nAnalyzable fields:")
            all_fields = []
            for i, (category, fields) in enumerate(ANALYSIS_CONFIG.items(), 1):
                print(f"\n{i}. {category}:")
                for j, (field, desc) in enumerate(fields.items(), 1):
                    print(f"   {field}: {desc}")
                    all_fields.append(field)
            
            field = input("\nEnter field name: ").strip()
            if field in analyzer.df.columns:
                analyzer.analyze_single_dimension(field, field)
            else:
                print("❌ Field does not exist!")
        elif choice == '3':
            col1 = input("Field 1: ").strip()
            col2 = input("Field 2: ").strip()
            if col1 in analyzer.df.columns and col2 in analyzer.df.columns:
                analyzer.analyze_crosstab(col1, col2, f'{col1} × {col2}')
            else:
                print("❌ Fields do not exist!")
        elif choice == '4':
            field = input("Time field (Year/YearMonth/Quarter): ").strip()
            if field in analyzer.df.columns:
                analyzer.analyze_time_series(field, f'{field} Time Series')
            else:
                print("❌ Field does not exist!")
        elif choice == '5':
            print("\n📊 Quick Summary:")
            print(f"Total records: {len(analyzer.df):,}")
            print(f"Time range: {analyzer.df['DATE OCC'].min()} to {analyzer.df['DATE OCC'].max()}")
            print(f"Number of areas: {analyzer.df['AREA NAME'].nunique()}")
            print(f"Number of crime types: {analyzer.df['Crm Cd Desc'].nunique()}")
            print(f"Felony percentage: {(analyzer.df['CrimeCategory']=='Felony').mean()*100:.1f}%")

# ============================================================================
# === Main Program ===
# ============================================================================

if __name__ == "__main__":
    analyzer = CrimeAnalyzer(df, charts_dir, stats_dir)
    
    # Ask for mode
    print("\n" + "="*80)
    print("Choose run mode:")
    print("1. Interactive mode (manual selection)")
    print("2. Full analysis (auto generate all)")
    print("="*80)
    
    mode = input("Choose (1/2): ").strip()
    
    if mode == '2':
        run_full_analysis(analyzer)
    else:
        interactive_mode(analyzer)