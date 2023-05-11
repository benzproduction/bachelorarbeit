import PageHeader from "components/Shared/PageHeader";
import { NextPage } from "next";
import { useRouter } from "next/router";
import { Input, TextArea } from "@appkit4/react-components/field";
import { useState } from "react";
import Answer, { Source } from "components/Prompt/Answer";
import { Button } from "@appkit4/react-components";
import toast from "components/toast";
import { Loading } from "@appkit4/react-components/loading";
import { FileModal } from "components/Shared";
import SaveUrlModal from "components/Shared/SaveUrlModal";
import { Select } from "@appkit4/react-components";
import useSWR from "swr";
import { OptionButton } from "components/Shared/OptionButton";
import NewKBModal from "components/Shared/NewKBModal";

const PromptPage: NextPage = () => {
  const [prompt, setPrompt] = useState(`<|im_start|>system \n
You are an intelligent assistant helping an employee with general questions regarding a for them unknown knowledge base. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. 
Do not generate answers that don't use the sources below. Answer the question in the language of the employees question.
If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format.
Each source has a name followed by colon and the actual information ending with a semicolon, always include the source name for each fact you use in the response.
Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
The employee is asking: {question}
Sources:{sources}
<|im_end|>`);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<any>("");
  const [showAnswer, setShowAnswer] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const [fileModalVisible, setFileModalVisible] = useState(false);
  const [saveUrlModalVisible, setSaveUrlModalVisible] = useState(false);
  const [createKBModelVisible, setCreateKBModelVisible] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState("real_estate_index");

  const { data: indices } = useSWR("/api/v1/indices", {
    onSuccess: (data) => {
      setSelectedIndex(data[0]);
    },
    revalidateOnFocus: false,
  });

  const onQuestionChange = (value: any, event: any) => {
    setQuestion(value);
  };
  const onPromptChange = (value: any, event: any) => {
    setPrompt(value);
  };

  const checkPrompt = () => {
    // check if the prompt includes the following values:
    // <|im_start|>system in the start of the prompt
    // {question} in the prompt
    // {sources} in the prompt
    // <|im_end|>system in the end of the prompt

    // if not, show a toast with the error message for the missing value

    if (!prompt.includes("<|im_start|>system")) {
      return toast({
        text: "The prompt must start with <|im_start|>system",
        type: "error",
        duration: 3000,
      });
    }
    if (!prompt.includes("{question}")) {
      return toast({
        text: "The prompt must include {question} to deliver the question",
        type: "error",
        duration: 3000,
      });
    }
    if (!prompt.includes("{sources}")) {
      return toast({
        text: "The prompt must include {sources} to answer your question",
        type: "error",
        duration: 3000,
      });
    }
    if (!prompt.includes("<|im_end|>")) {
      return toast({
        text: "The prompt must include <|im_end|>system",
        type: "error",
        duration: 3000,
      });
    }
    return true;
  };
  const checkQuestion = () => {
    // if the question includes an http/https link,
    // show a modal which asks the user if he wants to permanently save the link as a source

    if (question.includes("http://") || question.includes("https://")) {
      setSaveUrlModalVisible(true);
      console.log(
        "the question includes an url therefore we exit the normal flow"
      );
      return false;
    }
    return true;
  };

  const onSubmit = async () => {
    const promptOk = checkPrompt();
    const questionOk = checkQuestion();
    if (promptOk && questionOk) {
      try {
        setLoading(true);
        // query the sources from /api/v1/source
        // body has to include the question and top_k
        // top_k is the number of sources to return
        const res1 = await fetch("/api/v1/source", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: question,
            top_k: 4,
            index: selectedIndex,
          }),
        });
        if (res1.ok) {
          const sources = await res1.json();
          setSources(sources.results);

          const sourceString = sources.results
            .map((source: Source) => {
              return `${source.value.filename}: ${source.value.text_chunk};`;
            })
            .join("\n");

          // use the sources to generate the complete prompt
          const completePrompt = prompt
            .replace("{question}", question)
            .replace("{sources}", sourceString);

          // query the answer from /api/v1/completion
          // body has to include the prompt
          const res2 = await fetch("/api/v1/completion", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              prompt: completePrompt,
            }),
          });

          if (res2.ok) {
            const answer = await res2.json();
            setLoading(false);
            setAnswer(answer);
            setShowAnswer(true);
            // fetch("/api/v1/qa_dataset", {
            //   method: "POST",
            //   headers: {
            //     "Content-Type": "application/json",
            //   },
            //   body: JSON.stringify({
            //     question: question,
            //     answer: answer,
            //     sources: sources.results,
            //   }),
            // });
          } else {
            toast({
              text: `Something went wrong (Status: ${res2.status})`,
              type: "error",
              duration: 3000,
            });
            setLoading(false);
          }
        } else {
          setLoading(false);
          toast({
            text: `Something went wrong (Status: ${res1.status})`,
            type: "error",
            duration: 3000,
          });
        }
      } catch (e) {
        console.log(e);
        setLoading(false);
      }
    }
  };

  const saveUrl2DB = async (url: string) => {};
  const useUrlasSource = async (url: string) => {};

  return (
    <div className="flex flex-col gap-10">
      <PageHeader
        title="Prompt"
        subtitle="Welcome to the prompt page! Here, you can ask any question regarding the documents uploaded to our system. Our AI-powered system will generate an answer to your question, and you also have the option to adjust the prompt used to generate the answer. Once the answer is generated, you can view the sources used to generate it. We encourage you to evaluate the generated answer using the 'evaluate' button, to ensure that our system is working properly. We're here to help you find the information you need, so don't hesitate to ask!"
      />
      <div className="flex flex-row w-full gap-5 self-center max-w-4xl ">
        <div className="flex flex-col w-1/2 gap-4 relative">
          <Select
            searchable={false}
            data={indices?.map((index: string) => {
              return { label: index, value: index };
            })}
            placeholder="Knowledge Base"
            value={selectedIndex}
            onSelect={(val) => setSelectedIndex(val as string)}
          />
          <OptionButton
            options={[
              {
                text: "Create a new knowledge base",
                onClick: () => {
                  setCreateKBModelVisible(true);
                },
              },
              {
                text: "Add documents to a knowledge base",
                onClick: () => {},
              },
            ]}
            wrapperStyle={{
              position: "absolute",
              top: "-2.75rem",
              right: "0",
            }}
          />
          <Input
            title="Your Question"
            value={question}
            onChange={onQuestionChange}
            label="Your Question"
            name="question"
            allowClear={true}
          />
          <div className="flex flex-row">
            <Button
              kind="primary"
              type="submit"
              onClick={() => {
                onSubmit();
              }}
            >
              Submit
            </Button>
          </div>
        </div>
        <div className="flex flex-col w-1/2 relative">
          <TextArea
            maxLength={1024}
            title="Your Prompt-template"
            value={prompt}
            onChange={onPromptChange}
            name="prompt"
          />
          <OptionButton
            options={[
              {
                text: "Save the current prompt",
                onClick: () => {},
              },
              {
                text: "Load a saved prompt",
                onClick: () => {},
              },
            ]}
            wrapperStyle={{
              position: "absolute",
              top: "-2.75rem",
              right: "0",
            }}
          />
        </div>
      </div>
      {loading && (
        <Loading
          loadingType="circular"
          indeterminate={true}
          compact={false}
        ></Loading>
      )}
      {showAnswer && (
        <Answer
          onClose={() => {
            setShowAnswer(false);
          }}
          answer={answer?.choices[0]?.text}
          sources={sources}
        />
      )}
      <FileModal
        visible={fileModalVisible}
        onClose={() => {
          setFileModalVisible(false);
        }}
        fileName="pv12.pdf"
        fileOptions={{
          page: 1,
          toolbar: false,
        }}
      />
      <SaveUrlModal
        visible={saveUrlModalVisible}
        onClose={(save: boolean) => {
          console.log("Save: ", save);
          setSaveUrlModalVisible(false);
        }}
      />
      <NewKBModal
        visible={createKBModelVisible}
        onClose={() => setCreateKBModelVisible(false)}
      />
    </div>
  );
};

export default PromptPage;
