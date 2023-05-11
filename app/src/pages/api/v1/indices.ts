import type { NextApiRequest, NextApiResponse } from "next";
import redisClient from "lib/redis";
import { withMethods } from "lib/middlewares";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const results = await redisClient.sendCommand(["FT._LIST"]);
    return res.status(200).json(results);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error." });
  }
}

export default withMethods(["GET"], handler);
