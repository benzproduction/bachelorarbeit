import type { NextApiRequest, NextApiResponse } from "next";
import { withMethods } from "lib/middlewares";
import formidable, { errors as formidableErrors } from "formidable";
import { BlobServiceClient, ContainerClient } from "@azure/storage-blob";
import mime from "mime";

export const FormidableError = formidableErrors.FormidableError;

export const parseForm = async (
  req: NextApiRequest
): Promise<{
  fields: Record<string, string>;
  files: formidable.Files;
}> => {
  return new Promise(async (resolve, reject) => {
    try {
      const form = formidable({
        maxFileSize: 100 * 1024 * 1024, // 100 MB
        filename: (name, ext, part) => {
          return `${Date.now()}-${name}.${mime.getExtension(
            part.mimetype || ""
          )}`;
        },
        multiples: true, // req.files to be arrays of files
      });

      const fieldArray = [] as { fieldName: string; value: string }[];

      form.on("field", (fieldName, value) => {
        fieldArray.push({ fieldName, value });
      });

      form.parse(req, function (err, fields, files) {
        if (err) {
          let error = err;
          reject(error);
        } else {
          const fields = fieldArray.reduce((acc, { fieldName, value }) => {
            acc[fieldName] = value;
            return acc;
          }, {} as Record<string, string>);
          resolve({ fields, files });
        }
      });
    } catch (error) {
      reject(error);
    }
  });
};

async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const blobServiceClient = BlobServiceClient.fromConnectionString(
      `DefaultEndpointsProtocol=https;AccountName=${process.env.AZURE_STORAGE_ACCOUNT_NAME};AccountKey=${process.env.AZURE_STORAGE_ACCESS_KEY};EndpointSuffix=core.windows.net`
    );

    const { fields, files } = await parseForm(req);
    const { containerName } = fields;
    var uploadedFiles = files.files;

    if (!containerName) {
      throw new FormidableError("No containerName provided", 400, 400);
    }
    if (!Array.isArray(uploadedFiles)) {
      uploadedFiles = [uploadedFiles];
    }
    const containerClient: ContainerClient =
      blobServiceClient.getContainerClient(containerName);

    const createContainerResponse = await containerClient.createIfNotExists({});
    if (!createContainerResponse) {
      throw new FormidableError("Container creation failed", 500, 500);
    }

    for (const file of uploadedFiles) {
      const blobName = `${file.originalFilename}`;
      const blockBlobClient = containerClient.getBlockBlobClient(blobName);
      const uploadBlobResponse = await blockBlobClient.uploadFile(
        file.filepath
      );
      if (!uploadBlobResponse) {
        throw new FormidableError("Upload failed", 500, 500);
      }
    }

    return res.status(200).json({ message: "Files uploaded successfully" });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error." });
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};

export default withMethods(["POST"], handler);
