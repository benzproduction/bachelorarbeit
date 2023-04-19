const colors = require('@appkit4/styles/design-tokens/tokens/colors.json');
const tokens = require('@appkit4/styles/design-tokens/tokens/tokens.json');

module.exports = {
  content: ['./src/pages/**/*.{js,ts,jsx,tsx}', './src/components/**/*.{js,ts,jsx,tsx}', './src/hooks.ts'],
  theme: {
    extend: {
      colors,
      tokens
    },
  },
  plugins: [
    require('@tailwindcss/line-clamp'),
  ]
}
