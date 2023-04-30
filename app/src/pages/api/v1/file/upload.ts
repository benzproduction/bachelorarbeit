import type { NextApiRequest, NextApiResponse } from "next";
import { withMethods } from "lib/middlewares";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // upload file to azure blob storage to the specified container

    return res.status(501).json({ message: "Not implemented." });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error." });
  }
}

export default withMethods(["POST"], handler);
