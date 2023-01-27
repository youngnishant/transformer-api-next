// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import generate from "../../inference";

export default async function handler(req, res) {
  const data = await generate();
  console.log("xx", data);
  res.status(200).json(data);
}
