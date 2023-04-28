import type { NextApiRequest, NextApiResponse } from "next";
import { withMethods } from "lib/middlewares";
import { google } from "googleapis";

async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { question, answer } = req.body;
  try {
    const target = ["https://www.googleapis.com/auth/spreadsheets"];
    const jwt = new google.auth.JWT(
      process.env.GOOGLE_SHEETS_CLIENT_EMAIL,
      undefined,
      (process.env.GOOGLE_SHEETS_PRIVATE_KEY || "").replace(/\\n/g, "\n"),
      target
    );

    const sheets = google.sheets({ version: "v4", auth: jwt });
    const request = {
      spreadsheetId: process.env.GOOGLE_SHEETS_SPREADSHEET_ID,
      range: "A2:B2",
      valueInputOption: "USER_ENTERED",
      insertDataOption: "INSERT_ROWS",
      resource: {
        values: [[question, answer]],
      },
    };
    const response = (await sheets.spreadsheets.values.append(request)).data;

    res.status(200).json({ response });
  } catch (error) {
    console.log(error);
    return res.status(500).json({ message: "Internal server error." });
  }
}

export default withMethods(["POST"], handler);
