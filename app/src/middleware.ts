import { withAuth } from 'next-auth/middleware';
import { NextResponse } from 'next/server';

/** 
 * Send bearer token to backend for all /api/v1 requests
 */
export default withAuth(async function middleware(req) {
  const accessToken = req.nextauth.token?.accessToken;
  if(!accessToken) {
    return NextResponse.next();
  }

  const headers = new Headers(req.headers);
  headers.set('Authorization', `Bearer ${accessToken}`);

  return NextResponse.next({ request: { headers } });
});

export const config = { matcher: ['/api/v1/:path*'] };
