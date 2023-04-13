################################################################
# Driver Script to download test documents into the raw folder #
################################################################
import os
from azure.storage.blob import BlobClient, ContainerClient
from tqdm import tqdm

local_path = os.path.join(os.path.dirname(__file__), "raw", "pdfs")
os.makedirs(local_path, exist_ok=True)

blob_endpoint = "https://batrainingdata.blob.core.windows.net"
sas_token = "?sv=2021-12-02&ss=b&srt=co&sp=rltf&se=2024-04-15T22:00:00Z&st=2023-04-13T11:03:44Z&spr=https,http&sig=sJhE2%2F0jHSJG8GKa19pJbd4eT9VuF6dSot4ndGy4OUQ%3D"
containername = "pdfs"
container_sas_url = blob_endpoint + "/" + containername + sas_token

container_client = ContainerClient.from_container_url(container_sas_url)
blobs = container_client.list_blobs()
for blob in tqdm(blobs):
    if os.path.exists(os.path.join(local_path, blob.name)):
        continue
    blob_client = BlobClient.from_blob_url(blob_endpoint + "/" + containername + "/" + blob.name + sas_token)
    download_path = os.path.join(local_path, blob.name)
    with open(download_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    tqdm.write("Downloaded blob: {}".format(blob.name))

tqdm.write("All files downloaded to {}".format(local_path))










