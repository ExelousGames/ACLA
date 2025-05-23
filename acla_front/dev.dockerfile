FROM node:24-alpine AS build

#set the work directory inside the container
WORKDIR /app

#Copy package json and package-lock.json to work directory first
COPY package*.json ./

#install dependencies
RUN npm install

#Copy Source Code: Copy the remaining application code into the container.
COPY . .

#Build Application: Build the React application
RUN npm run build

# Stage  2
FROM node:24-alpine


COPY --from=build /app/src /app/src

# Define the command to run your app
CMD [ "npm", "start" ]