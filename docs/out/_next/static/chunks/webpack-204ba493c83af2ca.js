!function(){"use strict";var e,t,r,n,c,o,f,i={},u={};function a(e){var t=u[e];if(void 0!==t)return t.exports;var r=u[e]={id:e,loaded:!1,exports:{}},n=!0;try{i[e].call(r.exports,r,r.exports,a),n=!1}finally{n&&delete u[e]}return r.loaded=!0,r.exports}a.m=i,e=[],a.O=function(t,r,n,c){if(r){c=c||0;for(var o=e.length;o>0&&e[o-1][2]>c;o--)e[o]=e[o-1];e[o]=[r,n,c];return}for(var f=1/0,o=0;o<e.length;o++){for(var r=e[o][0],n=e[o][1],c=e[o][2],i=!0,u=0;u<r.length;u++)f>=c&&Object.keys(a.O).every(function(e){return a.O[e](r[u])})?r.splice(u--,1):(i=!1,c<f&&(f=c));if(i){e.splice(o--,1);var d=n();void 0!==d&&(t=d)}}return t},a.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return a.d(t,{a:t}),t},a.d=function(e,t){for(var r in t)a.o(t,r)&&!a.o(e,r)&&Object.defineProperty(e,r,{enumerable:!0,get:t[r]})},a.f={},a.e=function(e){return Promise.all(Object.keys(a.f).reduce(function(t,r){return a.f[r](e,t),t},[]))},a.u=function(e){return"static/chunks/"+(({471:"d488728f",987:"72fdc299"})[e]||e)+"."+({18:"2f2d350ce98e731b",46:"7f825bcd22a832b3",154:"512653003c4fe647",161:"99eff4830d44b530",223:"3ae3c49b02b70b52",224:"5ea9101bad35e25d",244:"fe83820362427c4f",301:"10a8fb7436b6db9c",306:"b9c5293f3182721e",356:"b80003e9ef4def29",451:"83e5f24d8d20280e",471:"c98fb9a49c301665",494:"b19f430bbd1907c3",524:"2236aeb73a3a448d",525:"db13950cbc2b367b",529:"630f41a48e30c9ac",541:"7e68dc58f7b304ab",584:"f2d7dee766279e82",606:"6dc713679b1b4128",626:"8ab70ffaf60a6bb9",657:"9dd61f9245135373",769:"577de36efd46715c",783:"a628064dd3c0d669",802:"697a7eedd80665db",838:"61f5cae07a97b6c3",924:"b0ff9072929f227e",929:"9e0498eb4fc27ee4",987:"045cdaa3b929cc2b"})[e]+".js"},a.miniCssF=function(e){return"static/css/3735b0f78292ebc9.css"},a.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||Function("return this")()}catch(e){if("object"==typeof window)return window}}(),a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},t={},r="_N_E:",a.l=function(e,n,c,o){if(t[e]){t[e].push(n);return}if(void 0!==c)for(var f,i,u=document.getElementsByTagName("script"),d=0;d<u.length;d++){var l=u[d];if(l.getAttribute("src")==e||l.getAttribute("data-webpack")==r+c){f=l;break}}f||(i=!0,(f=document.createElement("script")).charset="utf-8",f.timeout=120,a.nc&&f.setAttribute("nonce",a.nc),f.setAttribute("data-webpack",r+c),f.src=a.tu(e)),t[e]=[n];var s=function(r,n){f.onerror=f.onload=null,clearTimeout(b);var c=t[e];if(delete t[e],f.parentNode&&f.parentNode.removeChild(f),c&&c.forEach(function(e){return e(n)}),r)return r(n)},b=setTimeout(s.bind(null,void 0,{type:"timeout",target:f}),12e4);f.onerror=s.bind(null,f.onerror),f.onload=s.bind(null,f.onload),i&&document.head.appendChild(f)},a.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},a.nmd=function(e){return e.paths=[],e.children||(e.children=[]),e},a.tt=function(){return void 0===n&&(n={createScriptURL:function(e){return e}},"undefined"!=typeof trustedTypes&&trustedTypes.createPolicy&&(n=trustedTypes.createPolicy("nextjs#bundler",n))),n},a.tu=function(e){return a.tt().createScriptURL(e)},a.p="/bachelorarbeit/_next/",c={272:0},a.f.j=function(e,t){var r=a.o(c,e)?c[e]:void 0;if(0!==r){if(r)t.push(r[2]);else if(272!=e){var n=new Promise(function(t,n){r=c[e]=[t,n]});t.push(r[2]=n);var o=a.p+a.u(e),f=Error();a.l(o,function(t){if(a.o(c,e)&&(0!==(r=c[e])&&(c[e]=void 0),r)){var n=t&&("load"===t.type?"missing":t.type),o=t&&t.target&&t.target.src;f.message="Loading chunk "+e+" failed.\n("+n+": "+o+")",f.name="ChunkLoadError",f.type=n,f.request=o,r[1](f)}},"chunk-"+e,e)}else c[e]=0}},a.O.j=function(e){return 0===c[e]},o=function(e,t){var r,n,o=t[0],f=t[1],i=t[2],u=0;if(o.some(function(e){return 0!==c[e]})){for(r in f)a.o(f,r)&&(a.m[r]=f[r]);if(i)var d=i(a)}for(e&&e(t);u<o.length;u++)n=o[u],a.o(c,n)&&c[n]&&c[n][0](),c[n]=0;return a.O(d)},(f=self.webpackChunk_N_E=self.webpackChunk_N_E||[]).forEach(o.bind(null,0)),f.push=o.bind(null,f.push.bind(f))}();