import CardPattern from "components/Shared/Card";
import { NextPage } from "next";
import { useRouter } from "next/router";

let patterns = [
  {
    link: "/prompt",
    name: "Prompt",
    prefixIconClass: "Appkit4-icon icon-ask-question-outline",
    description:
      "Ask a question about the knowledge base and investigate the answer and sources.",
  },
  {
    link: "/chat",
    name: "Chat",
    prefixIconClass: "Appkit4-icon icon-comment-outline",
    description:
      "Chat with the AI assistant about the knowledge base.",
  },
  {
    link: "/summary",
    name: "Summary",
    prefixIconClass: "Appkit4-icon icon-file-outline",
    description:
      "Summarize a file from the knowledge base.",
  },
];

type Props = {};

const Home: NextPage<Props> = () => {
  const router = useRouter();
  const onClick = (event: React.MouseEvent<HTMLElement>, url?: string) => {
    if(!url) return;
    router.push(url);
  };
  return (
    <div className="card-grid-page">
      <div className="ap-pattern-card-grid-second">
        {patterns.map((item, index: number) => {
          return (
            <CardPattern
              className="second"
              item={item}
              key={index}
              onClick={(event) => onClick(event, item.link)}
            ></CardPattern>
          );
        })}
      </div>
    </div>
  );
};

export default Home;
