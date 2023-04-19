import {
  GetServerSidePropsContext,
  NextApiRequest,
  NextApiResponse,
} from 'next';
import { getServerSession as getNextAuthServerSession } from 'next-auth';
import { authOptions } from 'pages/api/auth/[...nextauth]';

type WithContext = {
  ctx: GetServerSidePropsContext;
};

type WithRequest = {
  req: NextApiRequest;
  res: NextApiResponse;
};

export const getServerSession = async (props: WithContext | WithRequest) => {
  if ('ctx' in props) {
    return getNextAuthServerSession(props.ctx.req, props.ctx.res, authOptions);
  } else {
    return getNextAuthServerSession(props.req, props.res, authOptions);
  }
};
