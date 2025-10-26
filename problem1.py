#!/usr/bin/env python3
"""
Problem 1: Log Level Distribution Analysis

This script analyzes the distribution of log levels (INFO, WARN, ERROR, DEBUG)
across all Spark cluster log files.

Author: sw1449
"""

import argparse
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, count, rand


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analysis of log level distribution')
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
    return parser.parse_args()


def extract_log_level(message):
    """Extract log level from log message."""
    import re
    log_levels = ['DEBUG', 'INFO', 'WARN', 'ERROR']
    for level in log_levels:
        if level in message[:50]:  # Check first 50 chars to avoid false matches
            return level
    return None


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"LogLevelAnalysis-{args.net_id}") \
        .master(args.master_url) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # Load all log files
        data_path = args.data_dir
        logs_df = spark.read.text(f"{data_path}/*/*.log")
        
        # Parse log levels
        logs_parsed = logs_df.select(
            regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)', 1).alias('log_level'),
            col('value').alias('log_entry')
        ).filter(col('log_level') != '')
        
        # Count log levels
        log_counts = logs_parsed.groupBy('log_level').count().orderBy('log_level')
        
        # Collect results
        log_counts_list = log_counts.collect()
        
        # Write counts to CSV
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        counts_file = output_dir / f"problem1_counts.csv"
        with open(counts_file, 'w') as f:
            f.write("log_level,count\n")
            for row in log_counts_list:
                f.write(f"{row['log_level']},{row['count']}\n")
        print(f"✓ Written counts to {counts_file}")
        
        # Sample 10 random log entries with their levels
        sample_entries = logs_parsed.orderBy(rand()).limit(10)
        sample_list = sample_entries.collect()
        
        sample_file = output_dir / f"problem1_sample.csv"
        with open(sample_file, 'w') as f:
            f.write("log_entry,log_level\n")
            for row in sample_list:
                # Escape quotes in log entry
                entry = row['log_entry'].replace('"', '""')
                f.write(f'"{entry}",{row["log_level"]}\n')
        print(f"✓ Written sample entries to {sample_file}")
        
        # Calculate summary statistics
        total_lines = logs_df.count()
        total_with_levels = logs_parsed.count()
        unique_levels = len(log_counts_list)
        
        # Calculate percentages
        level_counts_dict = {row['log_level']: row['count'] for row in log_counts_list}
        total_with_levels_int = sum(level_counts_dict.values())
        
        summary_file = output_dir / f"problem1_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Total log lines processed: {total_lines:,}\n")
            f.write(f"Total lines with log levels: {total_with_levels_int:,}\n")
            f.write(f"Unique log levels found: {unique_levels}\n\n")
            f.write("Log level distribution:\n")
            
            # Sort by count descending
            sorted_levels = sorted(level_counts_dict.items(), key=lambda x: x[1], reverse=True)
            for level, cnt in sorted_levels:
                percentage = (cnt / total_with_levels_int * 100) if total_with_levels_int > 0 else 0
                f.write(f"  {level:<6}: {cnt:>10,} ({percentage:>5.2f}%)\n")
        print(f"✓ Written summary to {summary_file}")
        
        print("\n✓ Problem 1 analysis complete!")
        
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
