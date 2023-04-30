import { Modal } from "@appkit4/react-components/modal";
import { Button } from "@appkit4/react-components/button";
import { Upload } from "@appkit4/react-components/upload";
import { Input } from "@appkit4/react-components";

type Props = {
  visible: boolean;
  onClose: () => void;
};

const NewKBModal = ({ visible, onClose }: Props) => {
  const onChange = (file: File, fileList: FileList): void => {
    console.log("onChange", file, fileList);
  };
  const uploadFiles = (fileList: any) => {
    console.log(fileList);
  };
  return (
    <Modal
      visible={visible}
      onCancel={() => onClose()}
      title="Create a new knowledge base"
      modalStyle={{ width: "33.75rem" }}
      footerStyle={{ paddingTop: "8px", marginTop: "-8px", minHeight: "64px" }}
      footer={
        <>
          <Button onClick={() => onClose()} kind="secondary">
            Close
          </Button>
          <Button onClick={() => onClose()} kind="primary" disabled>
            Create
          </Button>
        </>
      }
    >
      <Input
        className="mb-4"
        title="Name of the knowledge base"
        name="KB-name"
        allowClear={true}
      />
      <Upload
        onChange={onChange}
        multiple={true}
        autoUpload={true}
        onUpload={uploadFiles}
        acceptFileType=".PPTX,.DOCX,.PDF,.TXT"
        uploadInstruction="You can upload PPTX, DOCX, TXT or PDF files. The max file size is 100mb."
        uploadTitle="Upload your file"
        maxFileSize={100 * 1024 * 1024}
        config={{
          trigger: false,
          type: "inline",
          size: true,
        }}
      ></Upload>
    </Modal>
  );
};

export default NewKBModal;
