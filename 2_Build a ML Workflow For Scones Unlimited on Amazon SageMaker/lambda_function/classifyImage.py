import json
import base64
import boto3

ENDPOINT = "image-classification-2024-10-09-14-30-17-308"

runtime_client = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    image = base64.b64decode(event['body']['image_data'])

    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',  
        Body=image
    )
    
    inferences = response['Body'].read().decode('utf-8')

    event['inferences'] = inferences
    
    return {
        "statusCode": 200,
        "body": {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": event['inferences'],
        }
    }
