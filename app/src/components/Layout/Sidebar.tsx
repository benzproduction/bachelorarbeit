import {
  Navigation,
  NavigationItem,
} from "@appkit4/react-components/navigation";
import React from "react";
import { useRouter } from "next/router";

const Sidebar = () => {
  const router = useRouter();
  const navList: NavigationItem[] = [
    {
      name: "Getting started",
      prefixIcon: "upload",
    },
    {
      name: "Prompt",
      prefixIcon: "venn-abc",
    },
    {
      name: "Chat",
      prefixIcon: "comment",
    },
    {
      name: "Summary",
      prefixIcon: "file",
    }
  ];

  const [collapsed, setCollapsed] = React.useState(false);

  const selectedIndex = () => {
    const path = router.asPath.split("/")[1];
    if (path === "") {
      return 0;
    } else {
      return navList.findIndex((item) => item.name.toLowerCase() === path);
    }
  };

  const redirect = (
    event: React.MouseEvent<HTMLElement> | React.KeyboardEvent<HTMLElement>,
    item: NavigationItem,
    index: number
  ) => {
    if (item.name === navList[0].name) {
      router.push("/");
    } else {
      router.push(`/${item.name.toLowerCase()}`);
    }
  };

  const redirectSub = (
    event: React.MouseEvent<HTMLElement> | React.KeyboardEvent<HTMLElement>,
    item: NavigationItem | undefined,
    index: number | undefined,
    indexParent: number
  ) => {
    // routing logic here
    console.log(event, item, index, indexParent);
  };

  const onCollapseEvent = (
    collapsed: boolean,
    event: React.MouseEvent<HTMLElement> | React.KeyboardEvent<HTMLElement>
  ) => {
    setCollapsed(collapsed);
  };

  return (
    <Navigation
      width={330}
      className="h-[calc(100vh-4rem)]"
      showTooltip={true}
      hasHeader={false}
      navList={navList}
      onClickItem={redirect}
      onClickSubItem={redirectSub}
      selectedIndex={selectedIndex()}
      onClickCollapseEvent={onCollapseEvent}
    ></Navigation>
  );
};

export default Sidebar;
