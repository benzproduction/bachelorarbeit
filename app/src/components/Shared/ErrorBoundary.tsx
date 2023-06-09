import React, { Component, ErrorInfo, ReactNode } from "react";
import { Button } from "@appkit4/react-components";
import PageHeader from "./PageHeader";

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);

    // Define a state variable to track whether is an error or not
    this.state = { hasError: false };
  }

  public static getDerivedStateFromError(_: Error): State {
    // Update state so the next render will show the fallback UI.
    return { hasError: true };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div>
          <PageHeader
            title="Something went wrong"
            subtitle="We are sorry, but something went wrong. Please try again later."
          />
          <div className="h-[65vh] items-center justify-center px-16 pt-4">
            <div
              className="flex flex-col items-center justify-center w-[80%] h-full"
              style={{
                margin: "0 auto",
              }}
            >
              <svg
                className="ap-illustration-svg"
                data-name="Layer 1"
                id="Layer_1"
                viewBox="0 0 492.22 291.61"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  className="cls-1"
                  d="M589.76,189.42v.2"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-1"
                  d="M608.93,165.23V165"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-1"
                  d="M592.22,310.77v.35"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-1"
                  d="M586.44,347.9l-.34-.36"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-2"
                  d="M531,358.29c.34-.66.71-1.31,1.1-2L527.76,358Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <g>
                  <path
                    className="cls-3"
                    d="M153.85,372.35c-.09-.15-.17-.3-.26-.44.43.75.87,1.49,1.31,2.22l-.32-.53Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-3"
                    d="M152.48,369.91h0l-.52-1c.24.45.48.88.72,1.31C152.62,370.15,152.55,370,152.48,369.91Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-3"
                    d="M156.17,376.17l-.6-1c.87,1.4,1.77,2.78,2.7,4.1-.52-.74-1-1.49-1.53-2.26Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-3"
                    d="M152,369h0c-1.34-2.56-2.62-5.22-3.8-8C149.35,363.73,150.62,366.41,152,369Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-3"
                    d="M547.76,364.92l3.23.35c.34-.66.71-1.32,1.1-2Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-3"
                    d="M620.51,244.45C582.75,178.45,480,219.63,480,219.63S439.36,238,384.51,212c-41.82-19.85-88.83-29.57-135.15-21.95-56.12,9.2-115,37.87-111.9,118.13,0,.48,0,1,0,1.47,1,20.16,4.67,36.8,10.35,50.53-.06-.16-.12-.32-.19-.48L158.27,363l9.68-.36,9.67,1.68,21.19-5.77,28.32,5.5,22.11-1.41L262.6,360l7.83,1.58,11.28.17,31.78,3.58,15.67-3.49,22.33,2.71,6-.61,23.26,4.19,10.37-1.92,19.11-2.27,12.21-4,8.75,1.42,12.66.67,21.88,3.59,35.93-5.25,20.27,3.85,7.6-.35,16.39.83,6.65-2.13h0c.16-.25.32-.5.49-.75C566.39,342.23,658.3,310.44,620.51,244.45Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-3"
                    d="M549.79,367.84l0,0Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <g>
                  <polygon
                    className="cls-4"
                    points="177.56 205.72 177.56 205.72 183.56 204.25 177.56 205.72"
                  ></polygon>
                  <polygon
                    className="cls-4"
                    points="226.21 205.98 226.46 206.35 226.21 205.98 226.21 205.98"
                  >
                    {" "}
                  </polygon>

                  <rect
                    className="cls-4"
                    height="17.19"
                    transform="translate(-225.23 472.7) rotate(-83.07)"
                    x="313.1"
                    y="357.28"
                  ></rect>
                  <polygon
                    className="cls-4"
                    points="256.71 210.5 256.71 210.5 245.21 208.73 256.71 210.5"
                  ></polygon>
                  <polygon
                    className="cls-4"
                    points="232.51 207.37 233.45 208.36 233.45 208.36 232.51 207.37"
                  ></polygon>
                  <path
                    className="cls-4"
                    d="M313.49,365.32l15.67-3.49,22.33,2.71,6-.61,23.26,4.19,10.37-1.92,19.11-2.27,12.21-4,8.75,1.42,12.66.67,9.16,1.5a83.18,83.18,0,1,0-148.7.79Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <path
                  className="cls-5"
                  d="M443.24,329a94.4,94.4,0,0,1-15.95,31.67l3.89.63,12.66.67,9.5,1.56a86.31,86.31,0,0,0,5-13.21,84.72,84.72,0,0,0-16.54-78A94.07,94.07,0,0,1,443.24,329Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <g>
                  <path
                    className="cls-5"
                    d="M353.51,317.7q9.56-5.51,19.55-11.07c23.89-13.32,46.74-25.31,67.16-35.36a90.68,90.68,0,0,0-8.69-8.4c-14,6.88-27.59,16.12-42.28,20.79a50.68,50.68,0,0,1-13.51,2.67c-22.18,1-49.89-7.35-70.33,1.33a84.11,84.11,0,0,0-9.14,28.09C314.41,310.21,334.07,315.05,353.51,317.7Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-5"
                    d="M374.68,309.53Q366.63,314,359,318.38a111,111,0,0,0,15.51.72c15.84-.57,29.24-7,42.4-14.74,10.16-6.32,21.12-14.83,32.4-20.92a78.45,78.45,0,0,0-6.22-8.8C424.29,283.12,401,294.87,374.68,309.53Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <path
                  className="cls-1"
                  d="M445.5,207.4h-.34"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-4"
                  d="M288.86,362.55c18.84-13.69,48.84-32.41,85.82-53,66.85-37.27,114.49-55.75,118-49.43,2.1,3.77-12.33,15.74-38.18,32.49.46,1,1.39,3.3,1.42,3.4,44.28-27.86,72.89-50.18,69.83-55.67-4.42-7.93-72.79,21.75-152.71,66.31-35.37,19.72-67.07,39-91,55.15Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <g>
                  <polygon
                    className="cls-5"
                    points="226.21 205.98 226.46 206.35 226.21 205.98 226.21 205.98"
                  ></polygon>
                  <polygon
                    className="cls-5"
                    points="232.51 207.37 233.45 208.36 233.45 208.36 232.51 207.37"
                  ></polygon>
                  <polygon
                    className="cls-5"
                    points="245.21 208.73 249.56 209.4 249.56 209.4 245.21 208.73"
                  ></polygon>
                  <path
                    className="cls-5"
                    d="M316.11,362.52c2.25.16,4.46.41,6.66.73l6.39-1.42,22.33,2.71,6-.61,23.26,4.19,10.37-1.92,9.65-1.15c17.76-6.59,34.35-17.08,51.81-25.6a52.64,52.64,0,0,1,7.75-3.34c.13-1,.24-2,.32-3q.3-3.58.3-7.11c-3,1-5.95,2.18-9.08,3.41-30.23,11.71-54.6,33.74-88.6,32.78-20.51-.42-38.28-9.68-58.41-11.45a43.36,43.36,0,0,0-6,0,83.56,83.56,0,0,0,4.63,11.9A53.16,53.16,0,0,1,316.11,362.52Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <g>
                  <polygon
                    className="cls-6"
                    points="219.24 133.03 215.56 133.03 215.56 131.36 218.87 131.36 218.87 130.19 215.56 130.19 215.56 128.73 219.16 128.73 219.16 127.46 214.06 127.46 214.06 134.29 219.24 134.29 219.24 133.03"
                  ></polygon>
                  <path
                    className="cls-6"
                    d="M364.85,291.46h1.5a1.15,1.15,0,0,1,.81.24,1.34,1.34,0,0,1,.32.79c0,.27.07.56.09.86a2.7,2.7,0,0,0,.15.77h1.51a.87.87,0,0,1-.16-.35,2.74,2.74,0,0,1-.08-.43c0-.15,0-.3,0-.45s0-.27,0-.38a4.63,4.63,0,0,0-.07-.52,1.75,1.75,0,0,0-.17-.47,1.4,1.4,0,0,0-.3-.38,1.17,1.17,0,0,0-.46-.23v0a1.58,1.58,0,0,0,.83-.67,2,2,0,0,0,.26-1,1.94,1.94,0,0,0-.14-.73,1.78,1.78,0,0,0-.41-.6,1.86,1.86,0,0,0-.63-.41,2.14,2.14,0,0,0-.82-.15h-3.69v6.83h1.51Zm0-3h1.64a1.19,1.19,0,0,1,.78.23.91.91,0,0,1,.25.72,1,1,0,0,1-.25.74,1.14,1.14,0,0,1-.78.23h-1.64Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-6"
                    d="M377.49,291.46H379a1.12,1.12,0,0,1,.81.24,1.35,1.35,0,0,1,.33.79c0,.27.07.56.09.86a2.7,2.7,0,0,0,.15.77h1.5a1,1,0,0,1-.16-.35,2.76,2.76,0,0,1-.07-.43c0-.15,0-.3,0-.45s0-.27,0-.38a3.09,3.09,0,0,0-.07-.52,1.47,1.47,0,0,0-.17-.47,1.2,1.2,0,0,0-.29-.38,1.24,1.24,0,0,0-.46-.23v0a1.62,1.62,0,0,0,.83-.67,2,2,0,0,0,.25-1,1.94,1.94,0,0,0-.14-.73,1.75,1.75,0,0,0-.4-.6,2.08,2.08,0,0,0-.63-.41,2.19,2.19,0,0,0-.83-.15H376v6.83h1.5Zm0-3h1.65a1.14,1.14,0,0,1,.77.23.92.92,0,0,1,.26.72,1,1,0,0,1-.26.74,1.09,1.09,0,0,1-.77.23h-1.65Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-6"
                    d="M389.23,293.25a3,3,0,0,0,1,.75,3.72,3.72,0,0,0,2.83,0,3.06,3.06,0,0,0,1.05-.75,3.22,3.22,0,0,0,.66-1.12,3.94,3.94,0,0,0,.23-1.39,4.07,4.07,0,0,0-.23-1.42,3.36,3.36,0,0,0-.66-1.15,3.08,3.08,0,0,0-1.05-.76,3.72,3.72,0,0,0-2.83,0,3,3,0,0,0-1,.76,3.36,3.36,0,0,0-.66,1.15,4.07,4.07,0,0,0-.23,1.42,3.94,3.94,0,0,0,.23,1.39A3.22,3.22,0,0,0,389.23,293.25Zm.71-3.38a2.3,2.3,0,0,1,.32-.75,1.7,1.7,0,0,1,.57-.53,2,2,0,0,1,1.73,0,1.7,1.7,0,0,1,.57.53,2.3,2.3,0,0,1,.32.75,3.57,3.57,0,0,1,.1.87,3.31,3.31,0,0,1-.1.83,2.25,2.25,0,0,1-.32.74,1.78,1.78,0,0,1-.57.52,2.06,2.06,0,0,1-1.73,0,1.78,1.78,0,0,1-.57-.52,2.25,2.25,0,0,1-.32-.74,3.31,3.31,0,0,1-.1-.83A3.57,3.57,0,0,1,389.94,289.87Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-6"
                    d="M407.46,293.34l0-.45c0-.15,0-.27,0-.38a4.63,4.63,0,0,0-.07-.52,1.75,1.75,0,0,0-.17-.47,1.23,1.23,0,0,0-.3-.38,1.2,1.2,0,0,0-.45-.23v0a1.59,1.59,0,0,0,.82-.67,2,2,0,0,0,.26-1,1.94,1.94,0,0,0-.14-.73,1.75,1.75,0,0,0-.4-.6,2,2,0,0,0-.64-.41,2.14,2.14,0,0,0-.82-.15h-3.69v6.83h1.51v-2.66h1.5a1.15,1.15,0,0,1,.81.24,1.35,1.35,0,0,1,.33.79c0,.27.06.56.08.86a2.38,2.38,0,0,0,.16.77h1.5a.87.87,0,0,1-.16-.35A2.74,2.74,0,0,1,407.46,293.34Zm-1.72-3.19a1.12,1.12,0,0,1-.78.23h-1.64v-1.92H405a1.17,1.17,0,0,1,.78.23.91.91,0,0,1,.25.72A1,1,0,0,1,405.74,290.15Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-6"
                    d="M344.27,330.15a20.06,20.06,0,0,0-6.67-5,22.9,22.9,0,0,0-16.49-.65,17.47,17.47,0,0,0-6.13,4.1l-.19-.2,2.76-15.7H346.1V301.11H307.48l-6.82,38.23H313.3a14.41,14.41,0,0,1,4.45-4.2,12.39,12.39,0,0,1,6.12-1.33,12.22,12.22,0,0,1,5,1,11.17,11.17,0,0,1,3.75,2.77,11.62,11.62,0,0,1,2.37,4.15,15.74,15.74,0,0,1,.79,5,15.05,15.05,0,0,1-.84,5,13.34,13.34,0,0,1-2.37,4.25,11.76,11.76,0,0,1-3.75,3,10.78,10.78,0,0,1-5,1.14,11.1,11.1,0,0,1-7.8-2.81,11.5,11.5,0,0,1-3.56-7.56h-14a21.16,21.16,0,0,0,2.22,9.53,20.73,20.73,0,0,0,5.68,6.87,24,24,0,0,0,8.15,4.09,34.31,34.31,0,0,0,9.63,1.34,25.5,25.5,0,0,0,9.88-1.73,24.72,24.72,0,0,0,8.14-5.19,25.33,25.33,0,0,0,5.59-7.95,23.86,23.86,0,0,0,2.07-9.92,28.83,28.83,0,0,0-1.43-9.14A22.13,22.13,0,0,0,344.27,330.15Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-6"
                    d="M396.91,307.44a19.84,19.84,0,0,0-8.1-5.93,27.73,27.73,0,0,0-19.55,0,19.56,19.56,0,0,0-8.05,5.93,30.34,30.34,0,0,0-5.48,11,60.68,60.68,0,0,0-2,16.94,63,63,0,0,0,2,17.33,30.37,30.37,0,0,0,5.48,11.11,19.65,19.65,0,0,0,8.05,5.93,27.73,27.73,0,0,0,19.55,0,19.93,19.93,0,0,0,8.1-5.93,30.1,30.1,0,0,0,5.54-11.11,63.45,63.45,0,0,0,2-17.33,61.09,61.09,0,0,0-2-16.94A30.07,30.07,0,0,0,396.91,307.44Zm-6.66,35.11a39.1,39.1,0,0,1-1.24,8.1,16.9,16.9,0,0,1-3.36,6.66,8.18,8.18,0,0,1-6.66,2.77,7.92,7.92,0,0,1-6.52-2.77,17.23,17.23,0,0,1-3.31-6.66,39.1,39.1,0,0,1-1.24-8.1q-.19-4.2-.19-7.16c0-1.19,0-2.62,0-4.3s.14-3.41.34-5.18a37,37,0,0,1,1-5.29,18,18,0,0,1,1.93-4.69,10.28,10.28,0,0,1,3.16-3.36,8.49,8.49,0,0,1,4.79-1.28,8.75,8.75,0,0,1,4.84,1.28,10.79,10.79,0,0,1,3.26,3.36,16.39,16.39,0,0,1,1.92,4.69,47.81,47.81,0,0,1,1,5.29,49.32,49.32,0,0,1,.4,5.18c0,1.68,0,3.11,0,4.3Q390.44,338.35,390.25,342.55Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-6"
                    d="M457.36,318.45a29.91,29.91,0,0,0-5.53-11,20,20,0,0,0-8.1-5.93,26.89,26.89,0,0,0-9.83-1.78,26.58,26.58,0,0,0-9.72,1.78,19.5,19.5,0,0,0-8.05,5.93,30.34,30.34,0,0,0-5.49,11,60.72,60.72,0,0,0-2,16.94,63.06,63.06,0,0,0,2,17.33,30.37,30.37,0,0,0,5.49,11.11,19.59,19.59,0,0,0,8.05,5.93,26.58,26.58,0,0,0,9.72,1.78,26.89,26.89,0,0,0,9.83-1.78,20.06,20.06,0,0,0,8.1-5.93,29.94,29.94,0,0,0,5.53-11.11,63,63,0,0,0,2-17.33A60.68,60.68,0,0,0,457.36,318.45Zm-12.2,24.1a38.44,38.44,0,0,1-1.23,8.1,16.9,16.9,0,0,1-3.36,6.66,8.21,8.21,0,0,1-6.67,2.77,7.92,7.92,0,0,1-6.51-2.77,17.23,17.23,0,0,1-3.31-6.66,37.81,37.81,0,0,1-1.24-8.1q-.19-4.2-.2-7.16c0-1.19,0-2.62.05-4.3s.15-3.41.35-5.18a37,37,0,0,1,1-5.29,18.3,18.3,0,0,1,1.92-4.69,10.49,10.49,0,0,1,3.16-3.36,8.51,8.51,0,0,1,4.79-1.28,8.73,8.73,0,0,1,4.84,1.28,10.58,10.58,0,0,1,3.26,3.36,16.13,16.13,0,0,1,1.93,4.69,47.81,47.81,0,0,1,1,5.29,46.71,46.71,0,0,1,.39,5.18c0,1.68.05,3.11.05,4.3C445.36,337.36,445.3,339.75,445.16,342.55Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <path
                  className="cls-7"
                  d="M550.77,365.69l-20.93-1.24L527.48,366l2.08,1.63L524,366.34l.83-1.37-5.14.64,2.36,2-20.28-6.33-30.14,5,12.09,1.9-40.28-5.38-11.39-.63-9.17-1.58L411,364.39l-11.1,2.1-.76-.45-4.77,2.68v-2.17l-3.41.39-8.35,1.62,11.5,1.78-24.2-3.14.94,1-7.24-2.38.51.74-6.73-1.84-3.84.26,6.14,1.29-18.32-2.14V365L334,363v1.12L330.07,363l-3.15.87v-1.26l-12,2.94,6.73,1.36L282,362.09l-10.14.61-9.12-2.23v1.39l-1.58-1-.5.61-1.29-.28-9.4,1.62-21.92,1.32,4.93,1-33-5.95-20.89,5.16,5.64,1.06-9.35-1.25-1,.23-2.52-.75-.5.33L168,363l-7.94.43,4.14,1.95-12.43-3.63.57,1.25-4.21-2a100.41,100.41,0,0,0,10.1,18.38h388.3A51.68,51.68,0,0,1,550.77,365.69Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <g>
                  <path
                    className="cls-8"
                    d="M552.56,362.54l-6.65,2.13-16.39-.83-7.6.35-20.27-3.85-35.93,5.25L443.84,362l-12.66-.67-8.75-1.42-12.21,4-19.11,2.27-10.37,1.92-23.26-4.19-6,.61-22.33-2.71-15.67,3.49-31.78-3.58-11.28-.17L262.6,360l-13.36,2.61L227.13,364l-28.32-5.5-21.19,5.77L168,362.6l-9.68.36-10.61-3.28c.17.42.33.85.51,1.26l4.21,2-.57-1.25,12.43,3.63-4.14-1.95L168,363l3.3,1.08.5-.33,2.52.75,1-.23,9.35,1.25-5.64-1.06L200,359.25l33,5.95-4.93-1L250,362.84l9.4-1.62,1.29.28.5-.61,1.58,1v-1.39l9.12,2.23,10.14-.61,39.62,4.82-6.73-1.36,12-2.94v1.26l3.15-.87,3.92,1.12V363l7.33,2v-.91l18.32,2.14L353.5,365l3.84-.26,6.73,1.84-.51-.74,7.24,2.38-.94-1,24.2,3.14-11.5-1.78,8.35-1.62,3.41-.39v2.17l4.77-2.68.76.45,11.1-2.1,11.94-3.79,9.17,1.58,11.39.63,40.28,5.38-12.09-1.9,30.14-5,20.28,6.33-2.36-2,5.14-.64-.83,1.37,5.55,1.27L527.48,366l2.36-1.53,20.93,1.24.22-.42-3.23-.35,4.33-1.62Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-8"
                    d="M527.44,376.45,509,373.76H480.84L464,377l-24.9-.36-37.95,2.64H400l-37.56-2.46-43.38-2.76-42.31,2.76H247.34l-16.52-4.65-22.31,2.62-7-2.2H187.36l-25.44-4.4L152,369l.52,1,9.44-.45,25.5,3.77,14.13.62,7,1.49,22.06-2.07,16.79,4.06,29.52-.08,42.21-1.93,43.44,2L381,379.28H158.25c13.35,19.09,31.71,30,51.08,35.82,47.46,14.31,100.86-1.74,100.86-1.74s57.15-22,91.62-16.18,78.78,37.26,115.35,23.74c26.37-9.75,26.43-26,29.41-41.64H419.22l19.85-1.38,24.9.37L480.84,375H509l18.48,2.69,15-2.69,5,.42c.1-.41.2-.82.31-1.23l-5.26-.45Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <path
                  className="cls-1"
                  d="M231.59,176h.34"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-1"
                  d="M528.69,188.94v-.2"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-1"
                  d="M516.42,203.45v.2"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <g>
                  <path
                    className="cls-9"
                    d="M422.35,238.66H422a1.48,1.48,0,1,0,0,3h.34a1.48,1.48,0,0,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M347.89,215.78l-.34.35a1.49,1.49,0,0,0,.05,2.1,1.49,1.49,0,0,0,2.09-.06l.34-.35a1.49,1.49,0,0,0,0-2.1A1.47,1.47,0,0,0,347.89,215.78Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M319.94,252.46h-.34a1.48,1.48,0,0,0,0,3h.34a1.48,1.48,0,1,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M283.82,315.3h-.21a1.48,1.48,0,0,0,0,3h.21a1.48,1.48,0,1,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M279.62,240.79a1.48,1.48,0,0,0-1.48,1.48v.35a1.48,1.48,0,1,0,3,0v-.35A1.48,1.48,0,0,0,279.62,240.79Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M485.22,235.75l-.18.18h0a1.48,1.48,0,0,0,.05,2.09,1.51,1.51,0,0,0,1,.41,1.49,1.49,0,0,0,1.07-.46l.19-.2a1.48,1.48,0,0,0-2.15-2Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M493.19,306.59l-.34.36a1.48,1.48,0,0,0,1.07,2.5A1.49,1.49,0,0,0,495,309l.34-.36a1.48,1.48,0,0,0-.05-2.09A1.46,1.46,0,0,0,493.19,306.59Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M248.35,281.5a1.48,1.48,0,0,0-1.48,1.48v.35a1.48,1.48,0,1,0,3,0V283A1.48,1.48,0,0,0,248.35,281.5Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M248,204.39a1.48,1.48,0,0,0,1.48-1.48v-.22a1.48,1.48,0,0,0-3,0v.22A1.48,1.48,0,0,0,248,204.39Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M562.23,224.13a1.51,1.51,0,0,0-1.49,1.23,1.56,1.56,0,1,0,3.06-.09A1.46,1.46,0,0,0,562.23,224.13Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M606.21,254.61a1.48,1.48,0,0,0-1.48,1.48v.36a1.48,1.48,0,1,0,3,0v-.36A1.48,1.48,0,0,0,606.21,254.61Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M582.34,315.19H582a1.48,1.48,0,0,0,0,3h.34a1.48,1.48,0,0,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M539,342.52h-.68a1.48,1.48,0,0,0,0,3H539a1.48,1.48,0,0,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M534.26,277.16h-.34a1.48,1.48,0,0,0,0,3h.34a1.48,1.48,0,1,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M258.16,352h-.34a1.48,1.48,0,0,0,0,3h.34a1.48,1.48,0,0,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M155.39,345.45a1.48,1.48,0,0,0-1.48,1.48v.36a1.48,1.48,0,1,0,3,0v-.36A1.48,1.48,0,0,0,155.39,345.45Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M172.48,315.31a1.48,1.48,0,1,0-2.14,2l.34.35a1.45,1.45,0,0,0,1.07.46,1.47,1.47,0,0,0,1.07-2.5Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M144.13,285.49h-.34a1.48,1.48,0,0,0,0,3h.34a1.48,1.48,0,1,0,0-3Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-9"
                    d="M171.4,259.75a1.48,1.48,0,0,0-1.48-1.48h-.34a1.48,1.48,0,0,0,0,3h.34A1.48,1.48,0,0,0,171.4,259.75Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <path
                  className="cls-1"
                  d="M156.22,224.18h.34"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-1"
                  d="M146.37,172.41H146"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-10"
                  d="M158.27,379.32c13.35,19.07,31.7,29.95,51.06,35.78,47.46,14.31,100.86-1.74,100.86-1.74s57.15-22,91.62-16.18,78.78,37.26,115.35,23.74c26.35-9.74,26.43-26,29.41-41.6Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-2"
                  d="M551,365.27c.34-.66.71-1.32,1.1-2l-4.33,1.62Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-10"
                  d="M400.6,379.32H158.27c13.35,19.07,31.7,29.95,51.06,35.78,47.46,14.31,100.86-1.74,100.86-1.74s57.15-22,91.62-16.18,78.78,37.26,115.35,23.74c26.35-9.74,26.43-26,29.41-41.6Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-10"
                  d="M542.43,375l-15,2.69L509,375H480.84L464,378.27l-24.9-.37-20.37,1.42H381.35l-18.81-2-43.44-2-42.21,1.93-29.52.08-16.79-4.06-22.06,2.07-7-1.49-14.13-.62-25.5-3.77-9.44.45c13.51,25,34.51,38.47,56.85,45.19,47.46,14.31,100.86-1.74,100.86-1.74s57.15-22,91.62-16.18,78.78,37.26,115.35,23.74c28.53-10.55,26.26-28.7,30.22-45.48Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <g>
                  <path
                    className="cls-11"
                    d="M242.3,257.85c-5.2-1-13.73-1.46-22.71-1.51,6.09,25.57,14.87,46.69,24,58.72A433.39,433.39,0,0,0,242.3,257.85Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-11"
                    d="M215.59,160.89c-.38.5-.77,1-1.17,1.61-3.89,18-3.76,45.05,1.28,74.7q1.61,9.47,3.7,18.33c8.93.05,17.44.53,22.82,1.53C237.4,205.43,223.72,173.55,215.59,160.89Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                </g>
                <path
                  className="cls-12"
                  d="M215.61,159.83c-17.46,22.08-53.46,116.49-10.77,274.1h21.55C266.89,273.66,231.29,183.43,215.61,159.83Zm.09,77.37c-5-29.65-5.17-56.7-1.28-74.7.4-.57.79-1.11,1.17-1.61,8.13,12.66,21.81,44.54,26.63,96.17-5.38-1-13.89-1.48-22.82-1.53Q217.32,246.68,215.7,237.2Zm3.89,19.14c9,.05,17.51.52,22.71,1.51a433.39,433.39,0,0,1,1.27,57.21C234.46,303,225.68,281.91,219.59,256.34Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-13"
                  d="M201.83,422.38h0c-.06-.25-.12-.51-.19-.76h0c-13.45-53.74-17.67-99.79-16.75-137.75-6.25,29.66-5.79,68.69-4,98.87,4.91,19.63,12.45,34.5,21.33,41.33Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-6"
                  d="M243.43,258.08h0l-.06-.58c-4.91-52.49-19.26-84.87-27.76-97.67-9.45,12-24.34,45.14-29.23,98.06,0,.17,0,.34,0,.51h0c-3.83,42.17-1.31,96.82,15.31,163.22h0c.07.25.13.51.19.76h0c1,3.82,1.95,7.66,3,11.55h.37c-1-3.89-2-7.74-2.91-11.55h26.38c-.87,3.81-1.76,7.65-2.72,11.55h.43C244.85,360.86,247.5,302.35,243.43,258.08Zm-1.21-1c-12.68-2.37-42.66-1.8-54.69.33,4.82-52,19-84.68,28.06-96.5C223.72,173.55,237.4,205.43,242.22,257.06Zm-54.76,1.12A108,108,0,0,1,198.77,257c0,65.55,2.29,103.83,10,164.67h-6.66C186,355,183.65,300.3,187.46,258.18ZM209.6,421.62c-7.72-60.87-10-99.14-10-164.73,14.15-.91,33.56-.78,42.71,1,3.81,41.7,1.83,96.2-13.45,163.77Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <path
                  className="cls-13"
                  d="M244.92,283.7c1,36.85-2.68,81.64-14.76,134.5,9.34-11.75,16.56-32.95,19.77-58.77C250.57,334.08,249.69,306.18,244.92,283.7Z"
                  transform="translate(-137.35 -159.83)"
                ></path>
                <g>
                  <path
                    className="cls-14"
                    d="M180.93,382.74c-1.83-30.18-2.29-69.21,4-98.87v0c-11.94,56-2.1,145.53-2.1,145.53h14.32l-4.79,22.08h1.23l8.52-22.08h1.56c-.47-1.77-.92-3.53-1.37-5.29C193.38,417.24,185.84,402.37,180.93,382.74Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <path
                    className="cls-14"
                    d="M244.92,283.69h0c4.77,22.48,5.65,50.38,5,75.73-3.21,25.82-10.43,47-19.77,58.77,0-.15.07-.3.1-.46-.87,3.84-1.78,7.7-2.74,11.62h1.78l8.53,22.08h1.22l-4.79-22.08h12.79S256.91,339.7,244.92,283.69Z"
                    transform="translate(-137.35 -159.83)"
                  ></path>
                  <polygon
                    className="cls-14"
                    points="68.6 274.1 67.2 284 88.93 284 87.54 274.1 68.6 274.1"
                  ></polygon>
                </g>
              </svg>
            </div>
          </div>
          <div
            style={{
              position: "absolute",
              left: "50%",
              transform: "translateX(-50%)",
            }}
          >
            <Button
              type="button"
              onClick={() => this.setState({ hasError: false })}
            >
              Try again
            </Button>
          </div>
        </div>
      );
    }

    // Return children components in case of no error

    return this.props.children;
  }
}

export default ErrorBoundary;
