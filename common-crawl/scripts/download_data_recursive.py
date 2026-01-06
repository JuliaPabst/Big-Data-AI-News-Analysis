import os
import boto3

# --- CONFIGURATION ---
S3_BUCKET = "bigdata-mapping-ai-project"     
S3_PREFIX = "common-crawl/"         
LOCAL_BASE_DIR = "/home/glue_user/workspace/common-crawl/data"

def download_recursive():
    s3 = boto3.client('s3')
    
    print(f"--- Scanning s3://{S3_BUCKET}/{S3_PREFIX} ---")

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

    count = 0
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            s3_key = obj['Key']
            
            # Skip if it's just a folder placeholder (ends in /)
            if s3_key.endswith('/'):
                continue
                
            # FILTER: Only download relevant text files
            if s3_key.endswith('.txt') or s3_key.endswith('.html') or s3_key.endswith('.json'):
                
                # 1. Calculate the relative path to maintain structure
                relative_path = os.path.relpath(s3_key, S3_PREFIX)
                
                if relative_path.startswith(".."):
                    relative_path = os.path.basename(s3_key)
                
                local_file_path = os.path.join(LOCAL_BASE_DIR, relative_path)
                local_folder = os.path.dirname(local_file_path)

                # 2. Create local subfolder if it doesn't exist
                if not os.path.exists(local_folder):
                    os.makedirs(local_folder)

                # 3. Download
                if not os.path.exists(local_file_path):
                    print(f"Downloading: {relative_path}")
                    s3.download_file(S3_BUCKET, s3_key, local_file_path)
                    count += 1
                else:
                    print(f"Skipping (Exists): {relative_path}")

    print(f"--- Complete. Downloaded {count} new files. ---")

if __name__ == "__main__":
    download_recursive()