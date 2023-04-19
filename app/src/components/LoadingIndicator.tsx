import { atom, useAtom } from 'jotai';
import { Loading } from '@appkit4/react-components';

const LoadingState = atom(false);

export const useLoading = () => {
  const [isLoading, setLoading] = useAtom(LoadingState);
  return { isLoading, setLoading };
};

export const LoadingIndicator = () => {
  const { isLoading } = useLoading();

  const animClass = isLoading ? 'right-0' : 'right-[-70px]';

  return (
    <div className={`fixed bottom-0 mb-4 p-4 transition-all ${animClass}`}>
      <Loading loadingType="circular" indeterminate />
    </div>
  );
};
