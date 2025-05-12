FROM node:24-alpine

RUN mkdir -p /app

#set the work directory inside the container
WORKDIR /app

#Copy package json and package-lock.json to work directory first
COPY package*.json .

#install dependencies
RUN npm install

#copy rest of the application
COPY . .

EXPOSE 3000 

# Define the command to run your app
CMD [ "npm", "start" ]