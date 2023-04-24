import type { NextApiRequest, NextApiResponse } from 'next';
import { BlobServiceClient, ContainerClient, RestError } from '@azure/storage-blob';
import { withMethods } from 'lib/middlewares';


async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  var { name: filename } = req.query;
  filename = filename as string;
  if (!filename) {
    res.status(400).end();
    return;
  }

  const blobServiceClient = BlobServiceClient.fromConnectionString(
    `DefaultEndpointsProtocol=https;AccountName=${process.env.AZURE_STORAGE_ACCOUNT_NAME};AccountKey=${process.env.AZURE_STORAGE_ACCESS_KEY};EndpointSuffix=core.windows.net`
  );
  const containerClient: ContainerClient = blobServiceClient.getContainerClient(
    process.env.AZURE_STORAGE_CONTAINER_NAME as string
  );

  switch (req.method) {
    case 'GET': {
      try {

        // get the blob
        const blobClient = containerClient.getBlobClient(filename);
        const blob = await blobClient.download();

        if (filename.endsWith('.pdf')) {
          res.setHeader('Content-Type', 'application/pdf');
        } else {
          res.setHeader('Content-Type', 'application/octet-stream');
          res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
        }
        res.status(200).send(blob.readableStreamBody);
        return;
      } catch (error) {
        if (error instanceof RestError) {
          if (error.statusCode === 404) {
            return res.status(404).end(`File ${filename} not found.`);
          }
        }
        res.status(500).end(`Failed to get file ${filename}.`);
      }
    }

    default:
      return res.status(405).end();
  }
}

export default withMethods(['GET'], handler);