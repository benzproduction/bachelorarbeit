import { ReactNode, FC } from 'react';
import Head from 'next/head';
import Footer from './Footer';
import Header from './Header';
import { LoadingIndicator } from 'components/LoadingIndicator';

type Props = {
  title?: string;
  children?: ReactNode;
};

const environment = () => {
  if (typeof window == 'undefined') return null;
  const isStaging = window?.location.hostname.match(/stg/g)?.length;
  const isLocal = window?.location.hostname.match(/localhost/g)?.length;

  return isStaging ? 'Staging' : isLocal ? 'Localhost' : null;
};

const Layout: FC<Props> = ({ children, title }) => {
  const env = environment();
  const suffix = env ? ` - ${env}` : '';

  return (
    <>
      <LoadingIndicator />
      <div className="flex h-screen flex-col">
        <Head>
          <title>{`${title}${suffix}`}</title>
          <meta charSet="utf-8" />
          <meta
            name="viewport"
            content="initial-scale=1.0, width=device-width"
          />
          <meta
            name="description"
            content="PwC Advisory Incubator Development"
          />
        </Head>

        <Header title={title} />
        <div className="flex-auto overflow-auto bg-colors-background-default-value">
          {children}
        </div>
        <Footer />
      </div>
    </>
  );
};

export default Layout;
