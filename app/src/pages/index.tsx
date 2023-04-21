import { GetServerSideProps, NextPage } from "next";

export const getServerSideProps: GetServerSideProps = async (context) => {
  return {
    props: {
      name: 'John Smith'
    },
  };
};

type Props = {
  name: string;
}


const Home: NextPage<Props> = ({ name }) => {
  return (
    <div className="flex items-center justify-center w-full h-full">
    </div>
  )
}


export default Home;