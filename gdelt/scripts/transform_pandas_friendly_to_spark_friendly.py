import pandas as pd
import sys
import os

input_path = "/home/glue_user/workspace/data/gdelt_core.parquet"
output_path = "/home/glue_user/workspace/data/gdelt_core_fixed.parquet"

# Ensure data directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"Reading {input_path}...")
try:
    # Read with Pandas
    df = pd.read_parquet(input_path)
    
    # DROP the problematic column (Spark crashes on nanosecond timestamps)
    # Don't need 'date_ts' because we have 'label_week'
    if 'date_ts' in df.columns:
        print("Dropping 'date_ts' column to fix Nanosecond issue...")
        df = df.drop(columns=['date_ts'])
        
    # Save back to Parquet (Pandas writes cleaner schemas by default now)
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Success! Saved fixed file to: {output_path}")
    
except Exception as e:
    print(f"Error: {e}")