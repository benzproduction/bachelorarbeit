import { withMethods } from 'lib/middlewares';
import type { NextApiRequest, NextApiResponse } from 'next';
import { Configuration, OpenAIApi } from "azure-openai";
import redisClient from 'lib/redis';

const configuration = new Configuration({
    azure: {
        apiKey: process.env.OPENAI_API_KEY,
        endpoint: process.env.AZURE_OPENAI_ENDPOINT,
    }
});
const openai = new OpenAIApi(configuration);

const VECTOR_FIELD_NAME = 'content_vector'
const INDEX_NAME = 'real_estate_index';

function float32Buffer(arr: number[]) {
    return Buffer.from(new Float32Array(arr).buffer);
}
const topK = 3;

const prompt_prefix = `<|im_start|>system
Assistant helping an employee with general questions regarding a for them unknown knowledge base. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. 
Do not generate answers that don't use the sources below. Answer the question in the language of the employees question.
If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format.
Each source has a name followed by colon and the actual information ending with a semicolon, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
Begin!
Question: {question}

Sources:
{sources}
<|im_end|>
{chat_history}
`

const follow_up_questions_prompt_content = `Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook. 
    Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
    Try not to repeat questions that have already been asked.
    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'`

const query_prompt_template = `Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about employee healthcare plans and the employee handbook.
    Generate a search query based on the conversation and the new question. 
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    If the question is not in English, translate the question to English before generating the search query.

Chat History:
{chat_history}

Question:
{question}

Search query:
`

function get_chat_history_as_text(history: any, include_last_turn = true, approx_max_tokens = 1000) {
    let history_text = "";
    for (const h of history.slice().reverse()) {
        history_text = `<|im_start|>user\n${h.user}\n<|im_end|>\n<|im_start|>assistant\n${h.bot}<|im_end|>\n${history_text}`;
        if (history_text.length > approx_max_tokens * 4) {
            break;
        }
    }
    return history_text;
}


async function handler(
    req: NextApiRequest,
    res: NextApiResponse,
) {
    const { question, history } = req.body;

    console.log('question', question);



    if (!question) {
        return res.status(400).json({ message: 'No question in the request' });
    }
    // OpenAI recommends replacing newlines with spaces for best results
    const sanitizedQuestion = question.trim().replaceAll('\n', ' ');
    const embeddings_params = {
        'model': "text-embedding-ada-002",
        'input': sanitizedQuestion
    }

    try {
        const openai_response = await openai.createEmbedding(embeddings_params);
        const embeddings_arr = openai_response.data.data[0].embedding;

        const s = await redisClient.ft.search(INDEX_NAME, `*=>[${'KNN'} ${topK} @${VECTOR_FIELD_NAME} $BLOB AS vector_score]`, {
            PARAMS: {
                BLOB: float32Buffer(embeddings_arr)
            },
            SORTBY: {
                BY: 'vector_score',
                DIRECTION: 'DESC'
            },
            LIMIT: {
                from: 0,
                size: topK
            },
            DIALECT: 2,
            RETURN: ['vector_score', 'filename', 'text_chunk', 'text_chunk_index']
        });
        const sources = s.documents;
        const sourceString = sources
            .map((source: any) => {
                return `${source.value.filename}: ${source.value.text_chunk};`;
            })
            .join("\n");

        const prompt = prompt_prefix
            .replace('{question}', sanitizedQuestion)
            .replace('{sources}', sourceString)
            // .replace('{chat_history}', get_chat_history_as_text(history))
            .replace('{follow_up_questions_prompt}', follow_up_questions_prompt_content);

        console.log(prompt)

        const response = await openai.createCompletion({
            model: 'turbo',
            prompt: prompt,
            temperature: 0.7,
            max_tokens: 1024,
            n: 1,
            stop: ["<|im_end|>", "<|im_start|>"]
        });
        res.status(200).json({ text: response.data.choices[0].text, sources: sources });

    } catch (error: any) {
        console.log('error', error);
        res.status(500).json({ error: error.message || 'Something went wrong' });
    }
}

export default withMethods(['POST'], handler);