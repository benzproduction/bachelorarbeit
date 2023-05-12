import { createClient } from "redis";

const client = createClient({
  url: process.env.REDIS_URL,
  password: process.env.REDIS_PASSWORD,
});
client.connect();

export default client;
