const { withSuperjson } = require('next-superjson')

const config = {
  reactStrictMode: true,
  swcMinify: true,
  // images: {
  //   domains: [],
  // },
  // async rewrites() {
  //   return [
  //     {
  //       source: '/api/v1/:path*',
  //       destination: `${process.env.BACKEND_URL}/api/v1/:path*`
  //     }
  //   ]
  // },
  i18n: {
    locales: ["en"],
    defaultLocale: "en",
  },
  output: 'standalone',
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=31536000; includeSubDomains; preload',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'sameorigin',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Pragma',
            value: 'no-cache',
          },
          {
            key: 'Referrer-Policy',
            value: 'same-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'geolocation=(), camera=(), microphone=()',
          },
          {
            key: 'X-Powered-By',
            value: 'PwC Advisory Incubator',
          }
        ]
      }
    ]
  }
}

module.exports = withSuperjson()(config);
