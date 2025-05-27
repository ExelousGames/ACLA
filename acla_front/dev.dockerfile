FROM node:24-alpine AS build

#set the work directory inside the container
WORKDIR /app

#Copy package json and package-lock.json to work directory first
COPY package*.json ./

ENV PATH /app/node_modules/.bin:$PATH

#install dependencies
RUN npm install

#Copy Source Code: Copy the remaining application code into the container.
COPY . .

RUN npm run build

# Define the command to run your app
CMD [ "npm", "start" ]