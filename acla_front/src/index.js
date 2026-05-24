import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import './styles/design-tokens.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";

//The purpose of the function is to define the HTML element where a React component should be displayed.
const root = ReactDOM.createRoot(document.getElementById('root'));

//define the React component that should be rendered.
//  In "public" folder, there is an index.html file. This is where our React application will be rendered.
//Display inside an element with the id of "root":
// The Electron always-on-top overlay window loads the same bundle under
// hash route #/floating-chat. Radix's <Theme> paints a full-viewport dark
// background that would defeat the transparent BrowserWindow, so we skip
// the Theme wrapper for that route. The overlay also doesn't need any of
// Radix's tokens — its styles are self-contained.
const isFloatingChat =
  typeof window !== 'undefined' && window.location.hash.startsWith('#/floating-chat');

root.render(
  <React.StrictMode>
    {isFloatingChat ? (
      <App />
    ) : (
      <Theme appearance="dark" accentColor="green" grayColor="mauve" radius="medium">
        <App />
      </Theme>
    )}
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
