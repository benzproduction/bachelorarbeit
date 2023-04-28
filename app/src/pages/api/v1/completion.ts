import { Configuration, OpenAIApi } from "azure-openai";
import { withMethods } from "lib/middlewares";
import type { NextApiRequest, NextApiResponse } from "next";

const configuration = new Configuration({
  azure: {
    apiKey: process.env.OPENAI_API_KEY,
    endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  },
});
const openai = new OpenAIApi(configuration);

const MODELNAME = "davinci";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { prompt } = req.body;
  if (!prompt) {
    res.status(400).json({ message: "No prompt provided." });
    return;
  }

  const params = {
    model: MODELNAME,
    prompt: prompt,
    max_tokens: 1024,
    temperature: 0.9,
    top_p: 1,
    n: 1,
    // 'frequency_penalty': 0,
    // 'presence_penalty': 0,
    // 'stop': ["\n", " Human:", " AI:"]
  };

  try {
    const response = await openai.createCompletion(params);
    res.status(200).json(response.data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error completing prompt." });
  }
}

export default withMethods(["POST"], handler);
