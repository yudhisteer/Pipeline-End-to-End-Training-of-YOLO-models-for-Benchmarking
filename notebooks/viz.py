import boto3
import tarfile
import io
import json
import re
import pandas as pd
from IPython.display import display

# Change the number of Log lines more (e.g., display up to 100 lines)
pd.set_option('display.max_rows', 100)

s3 = boto3.client('s3')

bucket_name = 'cyudhist-yolo-pipeline-models-503561429929'

response = s3.list_objects_v2(Bucket=bucket_name)
keys_info = [obj for obj in response.get('Contents', []) if obj['Key'].endswith('output.tar.gz')]

metrics_data = []

for obj in keys_info:
    key = obj['Key']
    last_modified = obj['LastModified']

    match = re.search(r'(pipelines-[^/]+-TrainYOLOStep-[^/]+)/', key)
    execution_id = match.group(1) if match else key

    obj_body = s3.get_object(Bucket=bucket_name, Key=key)
    file_like = io.BytesIO(obj_body['Body'].read())

    with tarfile.open(fileobj=file_like, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith('evaluation_metrics.json'):
                f = tar.extractfile(member)
                if f:
                    metrics = json.load(f)
                    metrics_data.append({
                        'execution_id': execution_id,
                        'last_modified': last_modified,
                        'recall': metrics.get('recall'),
                        'map_50': metrics.get('map_50')
                    })

# Sort by execution date
metrics_data = sorted(metrics_data, key=lambda x: x['last_modified'])

dates = [m['last_modified'] for m in metrics_data]
recalls = [m['recall'] for m in metrics_data]
maps = [m['map_50'] for m in metrics_data]

# Table
df = pd.DataFrame(metrics_data)

# Convert the execution date and time into a string for easy viewing (optional)
df['execution_date'] = df['last_modified'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Select the columns you want to display
display_df = df[['execution_date','execution_id',  'recall', 'map_50']]

# Display as a table on Jupyter
display(display_df)
