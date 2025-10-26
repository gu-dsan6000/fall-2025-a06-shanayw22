#!/usr/bin/env python3
"""
Problem 2: Cluster Usage Analysis

Analyze cluster usage patterns to understand which clusters are most heavily used.
Extract cluster IDs, application IDs, and timestamps to create visualizations.

Author: sw1449
"""

import argparse
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, min, max, countDistinct, first, to_timestamp, input_file_name
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cluster usage analysis')
    parser.add_argument(
        'master_url',
        nargs='?',
        default='local[*]',
        help='Spark master URL (e.g., spark://HOST:7077 or local[*])'
    )
    parser.add_argument(
        '--net-id',
        required=True,
        help='Your net ID for output filenames'
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Directory containing log files (default: data/raw)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/output',
        help='Directory for output files (default: data/output)'
    )
    parser.add_argument(
        '--skip-spark',
        action='store_true',
        help='Skip Spark processing and regenerate visualizations from existing CSVs'
    )
    return parser.parse_args()


def process_with_spark(spark, data_path, output_dir):
    """Process logs with Spark to extract cluster and application information."""
    # Load all log files
    logs_df = spark.read.text(f"{data_path}/*/*.log")
    
    # Extract file path information
    logs_with_path = logs_df.withColumn(
        'file_path',
        input_file_name()
    )
    
    # Extract cluster ID, application ID, and app number from path
    logs_parsed = logs_with_path.select(
        regexp_extract('file_path', r'application_(\d+)_(\d+)', 1).alias('cluster_id'),
        regexp_extract('file_path', r'application_(\d+)_(\d+)', 2).alias('app_number'),
        regexp_extract('file_path', r'application_(\d+_\d+)', 0).alias('application_id'),
        regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
        col('value')
    ).filter(col('cluster_id') != '').filter(col('timestamp') != '')
    
    # Get start and end times for each application
    app_times = logs_parsed.groupBy('cluster_id', 'application_id', 'app_number').agg(
        min('timestamp').alias('start_time'),
        max('timestamp').alias('end_time')
    ).orderBy('cluster_id', 'app_number')
    
    # Collect to list
    timeline_data = app_times.collect()
    
    # Write timeline CSV
    timeline_file = output_dir / "problem2_timeline.csv"
    with open(timeline_file, 'w') as f:
        f.write("cluster_id,application_id,app_number,start_time,end_time\n")
        for row in timeline_data:
            f.write(f"{row.cluster_id},{row.application_id},{row.app_number},{row.start_time},{row.end_time}\n")
    print(f"✓ Written timeline data to {timeline_file}")
    
    # Aggregate cluster summary statistics
    cluster_summary = app_times.groupBy('cluster_id').agg(
        countDistinct('application_id').alias('num_applications'),
        min('start_time').alias('cluster_first_app'),
        max('end_time').alias('cluster_last_app')
    ).orderBy('cluster_id')
    
    cluster_summary_data = cluster_summary.collect()
    
    # Write cluster summary CSV
    cluster_summary_file = output_dir / "problem2_cluster_summary.csv"
    with open(cluster_summary_file, 'w') as f:
        f.write("cluster_id,num_applications,cluster_first_app,cluster_last_app\n")
        for row in cluster_summary_data:
            f.write(f"{row.cluster_id},{row.num_applications},{row.cluster_first_app},{row.cluster_last_app}\n")
    print(f"✓ Written cluster summary to {cluster_summary_file}")
    
    # Calculate overall statistics
    total_clusters = cluster_summary.count()
    total_apps = sum([row.num_applications for row in cluster_summary_data])
    avg_apps_per_cluster = total_apps / total_clusters if total_clusters > 0 else 0
    
    # Write stats file
    stats_file = output_dir / "problem2_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Total unique clusters: {total_clusters}\n")
        f.write(f"Total applications: {total_apps}\n")
        f.write(f"Average applications per cluster: {avg_apps_per_cluster:.2f}\n\n")
        f.write("Most heavily used clusters:\n")
        
        # Sort by number of applications
        sorted_clusters = sorted(cluster_summary_data, key=lambda x: x.num_applications, reverse=True)
        for row in sorted_clusters:
            f.write(f"  Cluster {row.cluster_id}: {row.num_applications} applications\n")
    print(f"✓ Written statistics to {stats_file}")
    
    return timeline_file, cluster_summary_file, stats_file


def create_visualizations(output_dir):
    """Create bar chart and density plot visualizations."""
    # Read the CSVs
    cluster_summary_df = pd.read_csv(output_dir / "problem2_cluster_summary.csv")
    timeline_df = pd.read_csv(output_dir / "problem2_timeline.csv")
    
    # Parse timestamps
    timeline_df['start_time_dt'] = pd.to_datetime(timeline_df['start_time'], format='%y/%m/%d %H:%M:%S')
    timeline_df['end_time_dt'] = pd.to_datetime(timeline_df['end_time'], format='%y/%m/%d %H:%M:%S')
    
    # Calculate duration in seconds
    timeline_df['duration_seconds'] = (timeline_df['end_time_dt'] - timeline_df['start_time_dt']).dt.total_seconds()
    
    # Bar chart: Applications per cluster
    plt.figure(figsize=(10, 6))
    ax = cluster_summary_df.sort_values('num_applications', ascending=False).plot(
        x='cluster_id',
        y='num_applications',
        kind='bar',
        color='steelblue',
        legend=False
    )
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Applications')
    ax.set_title('Applications per Cluster')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(cluster_summary_df.sort_values('num_applications', ascending=False)['num_applications']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    bar_chart_file = output_dir / "problem2_bar_chart.png"
    plt.savefig(bar_chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created bar chart: {bar_chart_file}")
    
    # Density plot: Job duration distribution for largest cluster
    largest_cluster = cluster_summary_df.sort_values('num_applications', ascending=False).iloc[0]['cluster_id']
    largest_cluster_data = timeline_df[timeline_df['cluster_id'] == largest_cluster]
    
    plt.figure(figsize=(10, 6))
    
    # Use log scale for x-axis
    sns.histplot(data=largest_cluster_data, x='duration_seconds', kde=True, log_scale=True)
    plt.xlabel('Job Duration (seconds, log scale)')
    plt.ylabel('Frequency')
    plt.title(f'Job Duration Distribution - Cluster {largest_cluster} (n={len(largest_cluster_data)})')
    plt.tight_layout()
    
    density_plot_file = output_dir / "problem2_density_plot.png"
    plt.savefig(density_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created density plot: {density_plot_file}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_spark:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName(f"ClusterUsageAnalysis-{args.net_id}") \
            .master(args.master_url) \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        
        try:
            print("Starting Spark processing...")
            process_with_spark(spark, args.data_dir, output_dir)
        finally:
            spark.stop()
    else:
        print("Skipping Spark processing, using existing CSVs...")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(output_dir)
    
    print("\n✓ Problem 2 analysis complete!")


if __name__ == '__main__':
    main()
