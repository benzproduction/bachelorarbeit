import { NextPage } from "next";
import { useRouter } from "next/router";
import { Table, Column } from "@appkit4/react-components/table";
import { Pagination } from "@appkit4/react-components/pagination";
import useSWR from "swr";
import { useState } from "react";

const ChatPage: NextPage = () => {
  const pageSize = 5;
  const [current, setCurrent] = useState(1);
  const onPageChange = (page: number) => {
    setCurrent(page);
  };

  const { data: raw_filenames } = useSWR("/api/v1/redis");

  // the raw_filenames is a string list of filenames
  // we need to convert it to a list of objects with id and name
  const filenames = raw_filenames
    ? {
        filenames: raw_filenames.filenames.map(
          (filename: string, index: number) => {
            return { id: index, name: filename };
          }
        ),
      }
    : { filenames: [] };

  const onRowClick = (event: MouseEvent, index: number, data: any) => {
    console.log("row clicked", index, data);
  };

  return (
    <div className="flex flex-col w-full h-full">
      <Table
        originalData={filenames.filenames}
        condensed
        pageSize={pageSize}
        currentPage={current}
        onRowClick={onRowClick}
        hasTitle
        className="file-table"
      >
        <Column field="name" sortKey="name">
          File Name
        </Column>
      </Table>
      <Pagination
        key={1}
        current={current}
        total={
          filenames.filenames.length % pageSize === 0
            ? filenames.filenames.length / pageSize
            : Math.floor(filenames.filenames.length / pageSize) + 1
        }
        onPageChange={onPageChange}
      ></Pagination>
    </div>
  );
};

export default ChatPage;
