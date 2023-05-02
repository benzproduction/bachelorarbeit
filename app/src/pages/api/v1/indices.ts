import type { NextApiRequest, NextApiResponse } from "next";
import redisClient from "lib/redis";
import { withMethods } from "lib/middlewares";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const keys = await redisClient.keys("*:*");
    const uniqueIndices = new Set(keys.map((key) => key.split(":")[0]));
    return res.status(200).json(Array.from(uniqueIndices).sort());
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error." });
  }
}

export default withMethods(["GET"], handler);
