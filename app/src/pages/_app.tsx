import type { AppProps } from 'next/app';
import { SessionProvider, signIn, useSession } from 'next-auth/react';
import MainLayout from 'components/Layout';
import SwrProvider from 'providers/SwrProvider';

import 'styles/globals.css';
import 'styles/default.scss';

type AppLayoutProps = AppProps & {
  Component: ExtendedNextPage;
  pageProps: any;
};

const MyApp = ({
  Component,
  pageProps: { session, ...pageProps },
}: AppLayoutProps) => {
  const Layout = Component.layout ?? ((page: any) => <>{page.children}</>);
  const authEnabled = !Component.auth || Component.auth.enabled;

  const AuthWrapper = authEnabled
    ? (page: any) => <Auth>{page.children}</Auth>
    : (page: any) => <>{page.children}</>;

  return (
    <SessionProvider session={session}>
      {/* Uncomment to enable auth */}
      {/* <AuthWrapper> */}
        <SwrProvider>
          <MainLayout title="Knowledge Base Investigator">
            <Layout>
              <Component {...pageProps} />
            </Layout>
          </MainLayout>
        </SwrProvider>
      {/* </AuthWrapper> */}
    </SessionProvider>
  );
};

const Auth: React.FC<{ children: any }> = ({ children }) => {
  const { status } = useSession({
    required: true,
    onUnauthenticated: () => signIn('pwc'),
  });

  if (status === 'loading') return null;
  return children;
};

export default MyApp;
