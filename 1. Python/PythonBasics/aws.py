import boto3

s3 = boto3.client('s3')

bucket_name = 'buckethaibc101907'
file_name = 'test.ipynm'
object_name = 'Test.ipynm'

try:
    s3.upload_file(file_name, bucket_name, object_name)
    print(f"File {file_name} uploaded to bucket {bucket_name} as {object_name}.")
except Exception as e:
    print(f"Error uploading file: {e}")

    
