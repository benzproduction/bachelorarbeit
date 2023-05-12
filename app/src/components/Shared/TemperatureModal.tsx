import { Modal } from "@appkit4/react-components/modal";
import { InputNumber } from "@appkit4/react-components";

type Props = {
  visible: boolean;
  onClose: () => void;
  temperature: number;
  setTemperature: (temperature: number) => void;
};

const TemperatureModal = ({
  visible,
  onClose,
  temperature,
  setTemperature,
}: Props) => {
  const handleTemperatureChange = (value: string | number) => {
    if (typeof value === "string") {
      return setTemperature(parseFloat(value));
    } else {
      return setTemperature(value);
    }
  };
  const handleClose = () => {
    onClose();
  };

  return (
    <Modal visible={visible} onCancel={handleClose} title="Change Temperature">
      <div className="m-4">
        <InputNumber
          title="Temperature"
          min={0}
          max={1}
          step={0.1}
          value={temperature}
          onChange={handleTemperatureChange}
        />
      </div>
    </Modal>
  );
};

export default TemperatureModal;
