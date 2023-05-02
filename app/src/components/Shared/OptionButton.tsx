import { memo, useState, useRef, CSSProperties } from "react";
import styles from "./../../styles/OptionButton.module.scss";
import { useOnClickOutside } from "hooks";
import { List, ListItem } from "@appkit4/react-components/list";
import cn from "classnames";

type ItemOptions = {
  text: string;
  onClick?: () => void;
};

type OptionButtonProps = {
  options: Array<ItemOptions>;
  icon?: string;
  isDirectionUp?: boolean;
  wrapperStyle?: CSSProperties;
};

const OptionButton = (props: OptionButtonProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const directionStyle = props.isDirectionUp
    ? styles.optionButton__options_up
    : "";
  const handleClick = () => {
    setIsOpen(!isOpen);
  };
  useOnClickOutside(ref, () => {
    setIsOpen(false);
  });

  const renderItem = (item: ItemOptions, index: number) => {
    return (
      <ListItem key={index} role="option" onClick={item.onClick}>
        <span className="primary-text">{item.text}</span>
      </ListItem>
    );
  };

  return (
    <div
      className={styles.optionButton__wrapper}
      ref={ref}
      style={props.wrapperStyle}
    >
      <button className={styles.optionButton__icon} onClick={handleClick}>
        {props.icon ?? (
          <span className="Appkit4-icon icon-horizontal-more-outline ap-font-medium"></span>
        )}
      </button>

      {isOpen && (
        <>
          <div className={cn(styles.optionButton__options, directionStyle)}>
            <List
              itemKey="text"
              bordered
              data={props.options}
              renderItem={renderItem}
              width={331}
              style={{ display: "inline-block" }}
            />
          </div>
        </>
      )}
    </div>
  );
};

export { OptionButton };
