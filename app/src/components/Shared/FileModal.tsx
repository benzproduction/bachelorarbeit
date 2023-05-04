import { Modal } from "@appkit4/react-components/modal";

type FileOptions = {
  page?: number;
  view?: "Fit" | "FitH" | "FitV";
  toolbar?: boolean;
};

export type Props = {
  visible: boolean;
  onClose: () => void;
  fileName: string;
  fileOptions?: FileOptions;
  title?: string;
};

const FileModal = ({
  visible,
  onClose,
  fileName,
  fileOptions,
  title,
}: Props) => {
  const formatFileUrl = (fileName: string, fileOptions?: FileOptions) => {
    var fileUrl = "/api/v1/file/" + fileName;
    if (fileOptions) {
      const { page, view, toolbar } = fileOptions;
      if (page) {
        fileUrl += "#page=" + page;
      }
      if (toolbar === false) {
        fileUrl += "&toolbar=0";
      }
      if (view) {
        fileUrl += "&view=" + view;
      }
    }
    return fileUrl;
  };

  return (
    <Modal
      visible={visible}
      onCancel={() => onClose()}
      title={title || ""}
      modalStyle={{ width: "90vh", height: "80vh" }}
      bodyStyle={{ height: "100%" }}
    >
      {/* Create an iframe to display a PDF file */}
      <iframe
        src={formatFileUrl(fileName, fileOptions)}
        width="100%"
        height="100%"
        style={{ border: "none" }}
      />
    </Modal>
  );
};

export default FileModal;
