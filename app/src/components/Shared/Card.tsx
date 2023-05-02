import React from "react";
import styles from "./../../styles/Card.module.scss";
import classNames from "classnames";
import Image from "next/image";
export interface CardItemType {
  name?: string;
  description?: string;
  dateTextIcon?: string;
  dateText?: string;
  link?: string;
  params?: any;
  src?: string; // If use src here, there is no need to use like content: url("./placeholder.svg") in css to show the image
  prefixIconClass?: string;
  timeTextIcon?: string;
  timeText?: string;
}
export interface CardProps {
  item?: CardItemType;
  onKeydown?: (
    event: React.KeyboardEvent<HTMLElement>,
    url?: string,
    params?: any
  ) => void;
  onClick?: (
    event: React.MouseEvent<HTMLElement>,
    url?: string,
    params?: any
  ) => void;
  className?: string;
}
const CardPattern = React.forwardRef<HTMLElement, CardProps>(
  (props: CardProps, ref) => {
    const { item, onClick, onKeydown, className } = props;
    return (
      <div
        className={classNames(styles.ap_pattern_card, className)}
        role="link"
        tabIndex={0}
        onClick={(event) => onClick?.(event, item?.link, item?.params)}
        onKeyDown={(event) => onKeydown?.(event, item?.link, item?.params)}
      >
        <div
          className={classNames(
            styles.ap_pattern_card_according,
            styles.component_thumb
          )}
        >
          {item?.src ? (
            <Image alt="" src={item?.src} priority />
          ) : (
            <img alt="" />
          )}
        </div>
        <div className={styles.ap_pattern_card_description}>
          <div className={styles.ap_pattern_card_head}>
            {item?.prefixIconClass && (
              <span
                className={classNames("prefixIcon", item?.prefixIconClass)}
                aria-label={item?.name}
              ></span>
            )}
            <span className={styles.ap_pattern_card_name}>{item?.name}</span>
          </div>
          <p className={styles.ap_pattern_card_desc}>{item?.description}</p>
        </div>
        <div className={styles.card_pattern_footer}>
          <span className={styles.component_footer_date}>
            {item?.dateTextIcon && (
              <span
                className={classNames("prefixIcon", item?.dateTextIcon)}
                aria-label={item?.dateText}
              ></span>
            )}
            <span className={styles.footer_text}>{item?.dateText}</span>
          </span>
          <span className={styles.component_footer_time}>
            {item?.timeTextIcon && (
              <span
                className={classNames("prefixIcon", item?.timeTextIcon)}
                aria-label={item?.timeText}
              ></span>
            )}
            <span className={styles.footer_text}>{item?.timeText}</span>
          </span>
        </div>
      </div>
    );
  }
);

CardPattern.displayName = "Card";

export default CardPattern;
