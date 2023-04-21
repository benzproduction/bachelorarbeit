import PageHeader from "components/Shared/PageHeader";
import { NextPage } from "next";
import { useRouter } from "next/router";
import { Input, TextArea } from "@appkit4/react-components/field";
import { useState } from "react";

const PromptPage: NextPage = () => {
  const [prompt, setPrompt] = useState(`<|im_start|>system \n
  You are an intelligent assistant helping an employee with general questions regarding a for them unknown knowledge base. Be brief in your answers.
  Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. 
  Do not generate answers that don't use the sources below. 
  If asking a clarifying question to the user would help, ask the question.
  For tabular information return it as an html table. Do not return markdown format.
  Each source has a name followed by colon and the actual information ending with a semicolon, always include the source name for each fact you use in the response.
  Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
  The employee is asking: {injected_prompt}\n
  Sources:
  {sources}
  <|im_end|>`);
  const [question, setQuestion] = useState("");

  const onQuestionChange = (value: any, event: any) => {
    setQuestion(value);
  };
  const onPromptChange = (value: any, event: any) => {
    setPrompt(value);
  };

  const questionSuffix = (
    <button
      className="cursor-pointer mx-2 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      type="submit"
    >
      <span className="Appkit4-icon icon-particulates-outline"></span>
    </button>
  );

  return (
    <div className="flex flex-col gap-10">
      <PageHeader title="Prompt" subtitle="This is the prompt page" />
      <div className="flex flex-row w-full gap-5 self-center max-w-4xl ">
        <div className="flex flex-col w-1/2">
          <Input
            title="Your Question"
            value={question}
            onChange={onQuestionChange}
            label="Your Question"
            suffix={questionSuffix}
            name="question"
          />
        </div>
        <div className="flex flex-col w-1/2">
          <TextArea
            maxLength={1024}
            title="Your Prompt"
            value={prompt}
            onChange={onPromptChange}
            name="prompt"
          />
        </div>
      </div>
    </div>
  );
};

export default PromptPage;
