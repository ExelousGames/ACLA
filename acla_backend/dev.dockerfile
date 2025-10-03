FROM node:24

WORKDIR /app

# Set Node.js memory limit to 2GB
ENV NODE_OPTIONS="--max-old-space-size=2048"

COPY package.json .

RUN npm install

COPY . .

CMD [ "npm", "run", "start:dev" ]