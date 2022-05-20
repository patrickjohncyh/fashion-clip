import boto3
from botocore.errorfactory import ClientError



class S3Client:

    def __init__(self, aws_key=None, aws_secret=None, region_name='us-west-2', **kwargs):
        # if key and secret are supplied, init boto with that
        if aws_key and aws_secret:
            self.s3_client = boto3.client('s3',
                                          aws_access_key_id=aws_key,
                                          aws_secret_access_key=aws_secret,
                                          region_name=region_name,
                                          **kwargs)
        else:
            # if not, initialize without keys - which means the context should run this code with appropriate IAM role
            self.s3_client = boto3.client('s3', **kwargs)

    def check_if_key_exist_in_bucket(self, bucket_name, file_name):
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=file_name)
        except ClientError:
            # Not found
            return False

        return True

    def download_file_from_bucket(self, bucket_name, s3_file_name, target_file_name):
        return self.s3_client.download_file(bucket_name, s3_file_name, target_file_name)

    def upload_file_to_bucket(self, local_file_name, bucket_name, s3_file_name):
        return self.s3_client.upload_file(local_file_name, bucket_name, s3_file_name)

    def put_buffer_to_s3_file(self, buffer, bucket_name, s3_file_name):
        return self.s3_client.put_object(Bucket=bucket_name, Key=s3_file_name, Body=buffer.getvalue())

    def get_matching_s3_objects_as_list(self, bucket_name, prefix='', suffix='', infix=''):
        """
        Return a list from the generator in the class

        :param bucket_name: N=name of the S3 bucket.
        :param prefix: only fetch objects whose key starts with this prefix (optional).
        :return: list of all objects in the bucket
        """
        return [obj for obj in self.get_matching_s3_objects(bucket_name=bucket_name,
                                                            prefix=prefix,
                                                            suffix=suffix,
                                                            infix=infix)]

    def get_matching_s3_objects(self, bucket_name, prefix='', suffix='', infix=''):
        """
        Generator to list objects in an S3 bucket (inspired by:  https://alexwlchan.net/2018/01/listing-s3-keys-redux/)
        Boto3 recommendation is for the "list_objects_v2" method ->
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects_v2

        :param bucket_name: N=name of the S3 bucket.
        :param prefix: only fetch objects whose key starts with this prefix (optional).
        """
        kwargs = {'Bucket': bucket_name}

        if prefix:
            kwargs['Prefix'] = prefix

        while True:
            resp = self.s3_client.list_objects_v2(**kwargs)

            try:
                contents = resp['Contents']
            except KeyError:
                return

            for obj in contents:
                if obj['Key'].endswith(suffix) and infix in obj['Key']:
                    yield obj

            # The S3 API is paginated, returning up to 1000 keys at a time.
            # Pass the continuation token into the next response, until we
            # reach the final page (when this field is missing).
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break

        return
