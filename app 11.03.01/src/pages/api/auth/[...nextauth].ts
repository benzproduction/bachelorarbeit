import NextAuth, { NextAuthOptions } from 'next-auth';

/**
 * Request a pwc identity staging environment from Service Now
 * Use following format for redirect url: <url>/api/auth/callback/pwc
 * Don't forget to include http://localhost:3000/api/auth/callback/pwc as well
 */

export const authOptions: NextAuthOptions = {
  providers: [
    {
      id: 'pwc',
      name: 'PwC Identity',
      type: 'oauth',
      client: {
        client_id: process.env.PWC_IDENT_CLIENT_ID,
        client_secret: process.env.PWC_IDENT_SECRET,
        token_endpoint_auth_method: 'client_secret_post',
      },
      wellKnown: process.env.PWC_IDENT_URL,
      checks: ['state'],
      authorization: {
        params: {
          scope: 'openid email profile cloudEmail', // insert your scope
        },
      },
      profile(profile) {
        return {
          id: profile.uid,
          email: profile.preferredMail,
          name: profile.name,
        };
      },
    },
  ],
  theme: {
    colorScheme: 'light',
    brandColor: '#D04A02',
    logo: '/images/logo.png',
  },
  // callbacks: {
  //   jwt: async ({ token, account }) => {
  //     if (account) {
  //       token.accessToken = account.access_token;
  //     }
  //     return token;
  //   },
  // },
};

export default NextAuth(authOptions);
