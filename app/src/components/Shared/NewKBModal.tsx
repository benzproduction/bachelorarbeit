import { Modal } from "@appkit4/react-components/modal";
import { Button } from "@appkit4/react-components/button";
import { Upload, FileModel } from "@appkit4/react-components/upload";
import { Input } from "@appkit4/react-components";
import { useState } from "react";
import toast from "components/toast";

type Props = {
  visible: boolean;
  onClose: () => void;
};

const NewKBModal = ({ visible, onClose }: Props) => {
  const [fileList, setFileList] = useState<FileModel[]>();
  const [KBName, setKBName] = useState("");
  const [KBNameError, setKBNameError] = useState(false);

  const handleClose = () => {
    setKBName("");
    setKBNameError(false);
    setFileList([]);
    onClose();
  };

  const onChange = (file: File, fileList: FileModel[]): void => {
    setFileList(fileList);
  };
  const uploadFiles = (fileList: any) => {
    console.log(fileList);
  };

  const handleKBNameChange = (
    value: string,
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const valid = checkKBName(value);
    setKBNameError(!valid);
    setKBName(value);
  };

  const checkKBName = (value: string) => {
    // This name may only contain lowercase letters, numbers, and hyphens, and must begin with a letter or a number.
    // Each hyphen must be preceded and followed by a non-hyphen character.
    // The name must also be between 3 and 63 characters long.
    const regex = /^(?=.{3,63}$)[a-z0-9]+(?:[-][a-z0-9]+)*$/;
    return regex.test(value);
  };

  const handleCreate = async () => {
    if (fileList === undefined) return;
    // send the files to the backend /api/v1/file/upload
    // the KBName should be set as query parameter
    const formData = new FormData();
    formData.append("KBName", KBName);
    for (let i = 0; i < fileList.length; i++) {
      formData.append("files", fileList[i].originFile as File);
    }

    try {
      const res = await fetch("/api/v1/file/upload", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        onClose();
        toast({
          text: "Successfully created the knowledge base",
          type: "success",
          duration: 3000,
        });
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <Modal
      visible={visible}
      onCancel={handleClose}
      title="Create a new knowledge base"
      modalStyle={{ width: "33.75rem" }}
      footerStyle={{ paddingTop: "8px", marginTop: "-8px", minHeight: "64px" }}
      footer={
        <>
          <Button onClick={handleClose} kind="secondary">
            Close
          </Button>
          <Button
            onClick={() => handleCreate()}
            kind="primary"
            disabled={
              KBName === "" || fileList === undefined || fileList.length === 0
            }
          >
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
        value={KBName}
        onChange={handleKBNameChange}
        required
        error={KBNameError}
        errorNode={
          <p className="error-input-text">
            This name may only contain lowercase letters, numbers, and hyphens,
            and must begin with a letter or a number. Each hyphen must be
            preceded and followed by a non-hyphen character. The name must also
            be between 3 and 63 characters long.
          </p>
        }
      />
      <div className="p-4">
        <Upload
          onChange={onChange}
          multiple={true}
          autoUpload={false}
          onUpload={uploadFiles}
          acceptFileType=".PPTX,.DOCX,.PDF,.TXT"
          uploadInstruction="You can upload PPTX, DOCX, TXT or PDF files. The max file size is 100mb."
          //   uploadTitle="Upload your file"
          maxFileSize={100 * 1024 * 1024}
          config={{
            trigger: false,
            type: "inline",
            size: true,
          }}
        ></Upload>
      </div>
    </Modal>
  );
};

export default NewKBModal;
