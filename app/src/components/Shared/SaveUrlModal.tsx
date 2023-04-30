import { Modal } from "@appkit4/react-components/modal";
import { Button } from "@appkit4/react-components/button";

type Props = {
  visible: boolean;
  onClose: (save: boolean) => void;
};

const SaveUrlModal = ({ visible, onClose }: Props) => {
  const handleClose = (save?: boolean) => {
    onClose(save ?? false);
  };

  return (
    <Modal
      closable={false}
      visible={visible}
      onCancel={handleClose}
      title="Would you like to permanently persists the contents from the URL in the database?"
      footer={
        <>
          <Button onClick={() => handleClose()} kind="secondary">
            No
          </Button>
          {/* <Button onClick={() => handleClose(true)}>Yes</Button> */}
        </>
      }
    >
      At the current moment, this feature is not yet implemented.
    </Modal>
  );
};

export default SaveUrlModal;
