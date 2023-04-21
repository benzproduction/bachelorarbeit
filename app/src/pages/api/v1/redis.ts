import type { NextApiRequest, NextApiResponse } from 'next';
import redisClient from 'lib/redis';
import { withMethods } from 'lib/middlewares';




async function handler(req: NextApiRequest, res: NextApiResponse) {
    try {

    
    // if (!redisClient.isReady) {
    //     await redisClient.connect();
    // }
    
    // get all unique file names, that are stored in redis

    // @ts-ignore
    const result = await redisClient.ft.aggregate('real_estate_index', '*', {
        LOAD: "@filename",
        STEPS: [
            {
                type: "GROUPBY",
                REDUCE: {
                    type: "TOLIST",
                    property: "@filename",
                }
            }
        ],
        DIALECT: 2
    });
    const filenames = result.results[0].__generated_aliastolistfilename

    res.status(200).json({ filenames });

    } catch (error) {
        console.log(error)
        return res.status(500).json({ message: 'Internal server error.' });
    }
}


export default withMethods(['GET'], handler);