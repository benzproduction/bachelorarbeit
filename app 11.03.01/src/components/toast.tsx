import { toaster, Notification } from '@appkit4/react-components';

type ToastProps = {
  text: string;
  type: 'error' | 'warning' | 'success';
  position?: string;
  duration?: number;
};

const toast = ({
  text,
  type,
  position = 'topRight',
  duration = 2000,
}: ToastProps) =>
  toaster.notify(<Notification status={type} message={text} />, {
    position,
    duration,
  });

export default toast;
