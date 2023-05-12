import { withMethods } from "lib/middlewares";
import type { NextApiRequest, NextApiResponse } from "next";
import { Configuration, OpenAIApi } from "azure-openai";
import redisClient from "lib/redis";
// Redis node js examples: https://github.com/redis/node-redis/blob/master/examples

const configuration = new Configuration({
  azure: {
    apiKey: process.env.OPENAI_API_KEY,
    endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  },
});
const openai = new OpenAIApi(configuration);

const VECTOR_FIELD_NAME = "content_vector";
const url = process.env.REDIS_URL;

const VECTOR_DIM = 1536;
const DISTANCE_METRIC = "COSINE";
var INDEX_NAME = "real_estate_index";
const PREFIX = "real_estate";

function float32Buffer(arr: number[]) {
  return Buffer.from(new Float32Array(arr).buffer);
}

async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { query, top_k = 5, index } = req.body;

  if (!query) {
    res.status(400).json({ message: "No query provided." });
    return;
  }
  const topK = Number(top_k) || 5;

  if (index) {
    INDEX_NAME = index;
  }
  // test if the index exists and if not return a 404
  try {
    const indexInfo = await redisClient.ft.info(INDEX_NAME);
    if (!indexInfo) {
      return res.status(404).json({ message: "Index not found." });
    }
  } catch (err) {
    console.error(err);
    return res.status(500).json({ message: "Internal server error." });
  }

  const embeddings_params = {
    model: "text-embedding-ada-002",
    input: query,
  };
  try {
    const openai_response = await openai.createEmbedding(embeddings_params);
    const embeddings_arr = openai_response.data.data[0].embedding;

    if (!redisClient.isReady) {
      await redisClient.connect();
    }

    const results = await redisClient.ft.search(
      INDEX_NAME,
      `*=>[${"KNN"} ${topK} @${VECTOR_FIELD_NAME} $BLOB AS vector_score]`,
      {
        PARAMS: {
          BLOB: float32Buffer(embeddings_arr),
        },
        SORTBY: {
          BY: "vector_score",
          DIRECTION: "DESC",
        },
        LIMIT: {
          from: 0,
          size: topK,
        },
        DIALECT: 2,
        RETURN: [
          "vector_score",
          "filename",
          "text_chunk",
          "text_chunk_index",
          "page",
        ],
      }
    );

    const uniqueFilenames = new Set<string>();

    results.documents.forEach((document) => {
      uniqueFilenames.add(document.value.filename as string);
    });

    return res.status(200).json({
      results: results.documents,
      files: Array.from(uniqueFilenames),
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ message: "Internal server error." });
  }
}

export default withMethods(["POST"], handler);
