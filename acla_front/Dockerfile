FROM node:24-alpine

WORKDIR /docker-image

COPY package.json .

RUN npm install

COPY . .

EXPOSE 3000 

CMD [ "npm", "start" ]