import { Modal } from "@appkit4/react-components/modal";
import { Button } from "@appkit4/react-components/button";
import React, {
  FunctionComponent,
  ReactNode,
  useCallback,
  useEffect,
  useState,
} from "react";
import { Steppers, Stepper } from "@appkit4/react-components/stepper";
import { Source } from "./Answer";
import { List, ListItem } from "@appkit4/react-components/list";
import { useTextSelection } from "hooks";
import { createPortal } from "react-dom";
import { css } from "@emotion/css";
import { FeedsComments } from "@appkit4/react-components/feeds-comments";

const Portal: FunctionComponent<{ children: ReactNode }> = ({ children }) => {
  const root = document.querySelector("#__next") as HTMLElement;
  return createPortal(children, root);
};

type PopUnderProps = {
  target?: HTMLElement;
  onListItemClick: (item: any) => void;
};

const PopUnder = ({ target, onListItemClick }: PopUnderProps) => {
  const [portalVisible, setPortalVisible] = useState(false);
  const [portalPosition, setPortalPosition] = useState({ x: 0, y: 0 });
  const [selectionContent, setSelectionContent] = useState("");

  const { isCollapsed, clientRect, textContent, range } =
    useTextSelection(target);

  // add a event listener for the mouse up event
  useEffect(() => {
    const handleMouseUp = (event: any) => {
      if (!isCollapsed && clientRect && textContent) {
        setPortalVisible(true);
        setPortalPosition({
          x: clientRect.left + clientRect.width / 3,
          y: clientRect.top + 20,
        });
        setSelectionContent(textContent);
        console.log("range", range);
      } else {
        if (!event.target.classList.contains("eval-option")) {
          setPortalVisible(false);
        }
      }
    };
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isCollapsed, clientRect, textContent]);

  const style = css`
    position: absolute;
    left: ${portalPosition.x}px;
    top: ${portalPosition.y}px;
    margin-left: -50px;
    width: 200px;
    z-index: 10000;
  `;

  const handleClick = (e: any, item: any) => {
    onListItemClick({
      text: selectionContent,
      type: item.type,
      color: item.color,
    });
    setPortalVisible(false);
  };

  const renderListItem = (item: any, index: number) => {
    return (
      <ListItem
        key={index}
        role="button"
        className="cursor-pointer eval-option"
        onClick={(e) => handleClick(e, item)}
        //   aria-selected={item.id === selectedId}
        //   onClick={() => { setSelectedId(item.id) }}
        //   onKeyDown={e => handleKeyDown(item, e)}
      >
        <span
          className="primary-text eval-option"
          style={{
            color: item.color,
          }}
        >
          {item.title}
        </span>
      </ListItem>
    );
  };

  return (
    <>
      {portalVisible && (
        <Portal>
          <div className={style}>
            <List
              renderItem={renderListItem}
              itemKey="id"
              bordered={true}
              data={[
                {
                  id: 2,
                  title: "Misleading information",
                  color: "#D04A02",
                  type: "misleading",
                },
                {
                  id: 3,
                  title: "Inaccurate information",
                  color: "#EB8C00",
                  type: "inaccurate",
                },
                {
                  id: 4,
                  title: "Irrelevant information",
                  color: "#E86153",
                  type: "irrelevant",
                },
                {
                  id: 5,
                  title: "Missing source",
                  color: "#822720",
                  type: "missing-source",
                },
              ]}
              style={{ display: "inline-block" }}
            ></List>
          </div>
        </Portal>
      )}
    </>
  );
};

export type SelectionType = "misleading" | "inaccurate" | "irrelevant" | "missing-source";

export interface Selection {
  text: string;
  type: SelectionType;
  color?: string;
  id?: string;
}

type Props = {
  visible: boolean;
  onClose: () => void;
  answer: string;
  sources: Source[];
};

const EvaluationModal = ({ visible, onClose, answer, sources }: Props) => {
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [target, setTarget] = useState<HTMLElement>();
  const ref = useCallback((el: any) => {
    if (el != null) {
      setTarget(el);
    } else {
      setTarget(undefined);
    }
  }, []);
  const [selections, setSelections] = useState<Selection[]>([]);
  const [showMissingInfo, setShowMissingInfo] = useState(false);
  const [missingInfosContent, setMissingInfosContent] = useState("");

  const addMissingInfo = (event: {
    missingInfosContent?: string | undefined;
    uploadedFiles: [];
    fileImageUrls: [];
  }) => {
    console.log("add", event);
  };

  const onStepperChange = (i: number) => {
    setActiveStepIndex(i);
  };

  const onDone = () => {
    onClose();
    setTimeout(() => {
      setActiveStepIndex(0);
    }, 500);
  };

  const onListItemClick = (item: any) => {
    // add the selection to the list and make sure its id is unique
    item.id = Math.random().toString(36).substr(2, 9);
    setSelections([...selections, item]);
  };

  const renderItem = (item: Selection, index: number) => {
    return (
      <ListItem
        key={index}
        role="radio"
        aria-live="off"
        onClick={handleSelectionListClick}
        data-id={item.id}
      >
        <span className="primary-text">{item.type}</span>
        <span
          aria-label="close"
          tabIndex={0}
          role="button"
          className="Appkit4-icon icon-close-outline"
          aria-hidden="true"
        ></span>
      </ListItem>
    );
  };
  const formatAnswer = (answer: string) => {
    let answerCopy = answer;
    selections.forEach((selection) => {
      const regex = new RegExp(selection.text, "g");
      answerCopy = answerCopy.replace(
        regex,
        `<span class="eval-selection ${selection.type}" style="color: ${selection.color}">${selection.text}</span>`
      );
    });

    // find all instances where a source is mentioned and convert to a link
    // a source should be formatted as [source name], the link has to be generated from that

    sources.forEach((source) => {
      const regex = new RegExp(`\\[${source.value.filename}\\]`, "g");
      answerCopy = answerCopy.replace(
        regex,
        `(<a href="/api/v1/file/${source.value.filename}#page=1" target="_blank" class="ap-link">${source.value.filename}</a>)`
      );
    });

    return answerCopy;
  };

  const handleSelectionListClick = (e: any) => {
    e.stopPropagation();
    // if the user clicks on the close icon, remove the selection from the list
    if (e.target.className.includes("icon-close-outline")) {
      const id = e.target.parentElement.getAttribute("data-id");
      const filteredSelections = selections.filter((s) => s.id !== id);
      setSelections(filteredSelections);
    }
    // else select the item and show the popunder
    else {
      const id = e.target.getAttribute("data-id");
      const selectedSelection = selections.find((s) => s.id === id);
      if (selectedSelection) {
        const selection = window.getSelection();
        if (selection && target) {
          const range = document.createRange();
          //
        }
      }
    }
  };

  return (
    <>
      <Modal
        visible={visible}
        title={"Answer Evaluation"}
        ariaLabel={"Answer Evaluation"}
        onCancel={onClose}
        modalStyle={{ width: "67.5rem" }}
        footerStyle={{
          paddingTop: "8px",
          marginTop: "-8px",
          minHeight: "64px",
        }}
        header={""}
        icons={""}
        footer={
          <>
            {activeStepIndex > 0 && (
              <Button
                kind="secondary"
                onClick={() => setActiveStepIndex(activeStepIndex - 1)}
              >
                Back
              </Button>
            )}
            {activeStepIndex < 2 && (
              <Button
                kind="primary"
                onClick={() => setActiveStepIndex(activeStepIndex + 1)}
              >
                Next
              </Button>
            )}
            {activeStepIndex === 2 && (
              <Button kind="primary" onClick={onDone}>
                Done
              </Button>
            )}
          </>
        }
        bodyStyle={{ minHeight: "92px" }}
      >
        <div className="flex flex-col gap-10 items-center">
          <Steppers
            readonly
            space={296}
            activeIndex={activeStepIndex}
            onActiveIndexChange={onStepperChange}
            hasTooltip={false}
          >
            <Stepper label="Sources" status="normal"></Stepper>
            <Stepper label="Provided Answer" status="normal"></Stepper>
            <Stepper label="Your Answer" status="normal"></Stepper>
          </Steppers>
          {activeStepIndex === 0 && <div>Source Eval</div>}
          {activeStepIndex === 1 && (
            <div className="flex flex-1 mb-4">
              <div className="w-2/3 p-5 h-[300px] flex flex-row items-center">
                <p
                  id="answer"
                  ref={ref}
                  dangerouslySetInnerHTML={{ __html: formatAnswer(answer) }}
                ></p>
                <PopUnder target={target} onListItemClick={onListItemClick} />
              </div>
              <div className="inline-block h-[300px] min-h-[1em] w-0.5 self-stretch bg-gray-500 opacity-100 dark:opacity-50"></div>
              <div className="w-1/3 p-5 h-[300px] flex flex-col items-center overflow-y-auto">
                <Button
                  kind="secondary"
                  compact
                  icon="icon-plus-outline"
                  onClick={() => setShowMissingInfo(true)}
                >
                  Report missing information
                </Button>
                <List
                  itemKey="id"
                  bordered={false}
                  data={selections}
                  renderItem={renderItem}
                  width={"100%"}
                  style={{ display: "inline-block" }}
                ></List>
              </div>
              {showMissingInfo && (
                <div className="report-missing-info">
                  <FeedsComments
                    type={"addCommentPanel"}
                    title="Missing Information"
                    addBtnName="Report"
                    commentsContent={missingInfosContent}
                    onCloseClick={() => setShowMissingInfo(false)}
                    onAddClick={addMissingInfo}
                    showAttachment={true}
                    maxLength={420}
                    onCommentContentChange={setMissingInfosContent}
                  ></FeedsComments>
                </div>
              )}
            </div>
          )}
          {activeStepIndex === 2 && (
            <div> Provide a corrected answer (your own)</div>
          )}
        </div>
      </Modal>
    </>
  );
};

export default EvaluationModal;
