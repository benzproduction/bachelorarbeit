import type { NextApiRequest, NextApiResponse } from "next";
import redisClient from "lib/redis";
import { withMethods } from "lib/middlewares";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // get all unique indices, that are stored in redis

    res.status(200).json(["real_estate_index"]);
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error." });
  }
}

export default withMethods(["GET"], handler);
