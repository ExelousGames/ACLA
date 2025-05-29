FROM node:24-alpine AS build

#this app is wrapped with a nginx image. this image inject environment variable at runtime, 
#but node image requires the variable at build time. we must provide the info at build time
ARG BACKEND_SERVER_IP
ARG BACKEND_PROXY_PORT


#must put REACT_APP at the front for react js
ENV REACT_APP_BACKEND_SERVER_IP $BACKEND_SERVER_IP
ENV REACT_APP_BACKEND_PROXY_PORT $BACKEND_PROXY_PORT

#set the work directory inside the container
WORKDIR /app

#Copy package json and package-lock.json to work directory first
COPY package*.json ./

#install dependencies
RUN npm install

#Copy Source Code: Copy the remaining application code into the container.
COPY . .

RUN npm run build

# Define the command to run your app
CMD [ "npm", "start" ]