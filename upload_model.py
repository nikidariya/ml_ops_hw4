import boto3
from botocore.client import Config

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4')
)

try:
    s3.create_bucket(Bucket='models')
    print('Bucket создан!')
except Exception as e:
    print(f'Bucket уже существует: {e}')

s3.upload_file('model_v1.pth', 'models', 'model_v1.pth')
print('Модель загружена в MinIO')

response = s3.list_objects_v2(Bucket='models')
for obj in response.get('Contents', []):
    print(f'Файл: {obj["Key"]} — {obj["Size"]} байт')
