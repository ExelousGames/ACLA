FROM node:24-alpine AS build

WORKDIR /app

COPY package*.json .

RUN npm install

COPY . .

#build nest js for all the necessary files will remain
RUN npm run build

# Stage 2
FROM node:24-alpine AS production

WORKDIR /app

ENV NODE_ENV production

COPY package*.json ./

RUN npm install --only production

#copy from necessary files over  to the image
COPY --from=build /app/dist ./dist
COPY --chown=node:node --from=build /app/node_modules ./node_modules

CMD [ "node", "dist/main.js" ]