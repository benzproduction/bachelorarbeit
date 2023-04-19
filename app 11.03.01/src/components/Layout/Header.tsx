import { useSession } from 'next-auth/react';
import {  useMemo } from 'react';
import { useRouter } from 'next/router';

import {
  Header as AppkitHeader,
} from '@appkit4/react-components/header';

import { Avatar } from '@appkit4/react-components/avatar';

type Props = {
  title?: string;
};

const Header: React.FC<Props> = ({ title }) => {
  const { data: session } = useSession();
  const userName = session?.user?.name;
  const router = useRouter();

  const onClickLogo = () => router.push('/');

  const initials = useMemo(() => {
    if (!userName) return '';
    const [firstName, lastName] = userName.split(' ');
    return `${firstName[0]}${lastName[0]}`;
  }, [userName]);

  return (
    <AppkitHeader
      onClickLogo={onClickLogo}
      titleTemplate={() => title}
      userTemplate={() => (
        <Avatar label={initials} role="button" disabled={false}></Avatar>
      )}
    />
  );
};

export default Header;
