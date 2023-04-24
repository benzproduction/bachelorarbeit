import { Panel } from "@appkit4/react-components/panel";
import { Tabs, Tab } from "@appkit4/react-components/tabs";
import { Button } from "@appkit4/react-components/button";
import { Table, Column } from "@appkit4/react-components/table";
import { useEffect, useState } from "react";
import EvaluationModal from "./EvaluationModal";
import { Tooltip } from "@appkit4/react-components/tooltip";

export type Source = {
  id: string;
  value: {
    vector_score: string;
    filename: string;
    text_chunk: string;
  };
};
type FlattenSource = {
  id: string;
  vector_score: string;
  filename: string;
  text_chunk: string;
};

type Props = {
  answer: string;
  sources: Source[];
  onClose?: () => void;
};

const Answer = ({ answer, sources: originalSources, onClose }: Props) => {
  const [currentView, setCurrentView] = useState("answer");
  const [sources, setSources] = useState<FlattenSource[]>([]);
  const [showEvaluationModal, setShowEvaluationModal] = useState(false);

  useEffect(() => {
    // flatten sources
    const newSources = originalSources.map((source) => {
      return {
        id: source.id,
        vector_score: source.value.vector_score,
        filename: source.value.filename,
        text_chunk: source.value.text_chunk,
      };
    });
    // default sort by vector score
    newSources.sort((a, b) => {
      return parseFloat(b.vector_score) - parseFloat(a.vector_score);
    });
    setSources(newSources);
  }, [originalSources]);

  const onTabChange = (i: number, value: string) => {
    setCurrentView(value);
  };

  const onSort = (sortKey: string, sortPhase: number) => {
    // console.log(sortKey, sortPhase);
  };
  const sortByVectorScore_1 = (a: any, b: any) => {
    return parseFloat(a.vector_score) - parseFloat(b.vector_score);
  };
  const sortByVectorScore_2 = (a: any, b: any) => {
    return parseFloat(b.vector_score) - parseFloat(a.vector_score);
  };

  const extraNode = (
    <div className="ap-extra-template-container" style={{ display: "flex" }}>
      <Tabs type="filled" onTabChange={onTabChange}>
        <Tab icon="icon-particulates-outline" value="answer"></Tab>
        <Tab icon="icon-aggregate-outline" value="sources"></Tab>
      </Tabs>
      <button
        type="button"
        aria-label="Close"
        className="ap-modal-header-icon ap-modal-header-close"
        onClick={onClose}
      >
        <span className="Appkit4-icon icon-close-outline height ap-font-medium"></span>
      </button>
    </div>
  );

  const footer = (
    <div className="mt-4 flex gap-3 flex-row items-center">
      {currentView === "answer" && (
        <Tooltip content="Copied" trigger="click" distance={4} position="top">
          <button
            className="copy-btn"
            onClick={() => {
              navigator.clipboard.writeText(answer);
            }}
          >
            <span className="Appkit4-icon icon-copy-outline"></span>
          </button>
        </Tooltip>
      )}
      <Button kind="primary" onClick={() => setShowEvaluationModal(true)}>
        Evaluate
      </Button>
    </div>
  );

  const renderFilenameCell = (row: any, field: string) => {
    if (!(field in row)) return "";

    const filename = row[field];

    // TODO: Do this with embeded iframe and modal and directly show the page
    return (
      <a
        href={`/api/v1/file/${filename}#page=1`}
        target="_blank"
        rel="noreferrer"
        className="ap-link"
      >
        {filename}
      </a>
    );
  };

  const formatAnswer = (answer: string) => {
    let answerCopy = answer;
    sources.forEach((source) => {
      const regex = new RegExp(`\\[${source.filename}\\]`, "g");
      answerCopy = answerCopy.replace(
        regex,
        `(<a href="/api/v1/file/${source.filename}#page=1" target="_blank" class="ap-link">${source.filename}</a>)`
      );
    });

    return answerCopy;
  };

  return (
    <div className="ap-panel-with-extra-container">
      <Panel
        title={currentView === "answer" ? "Generated Answer:" : "Used Sources:"}
        extra={extraNode}
        footer={footer}
      >
        {currentView === "answer" && (
          <p dangerouslySetInnerHTML={{ __html: formatAnswer(answer) }} />
        )}
        {currentView === "sources" && (
          <Table originalData={sources} hasTitle condensed onSort={onSort}>
            <Column
              sortKey="vector_score"
              field="vector_score"
              sortFunc1={sortByVectorScore_1}
              sortFunc2={sortByVectorScore_2}
            >
              Score
            </Column>
            <Column field="filename" renderCell={renderFilenameCell}>
              Filename
            </Column>
            <Column field="text_chunk">Text Chunk</Column>
          </Table>
        )}
      </Panel>
      <EvaluationModal
        visible={showEvaluationModal}
        onClose={() => setShowEvaluationModal(false)}
        answer={answer}
        sources={originalSources}
      />
    </div>
  );
};

export default Answer;
