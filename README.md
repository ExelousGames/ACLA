# ACLA


## Installation


### For Mac:


#### Install Mongo DB server
For more detail about [Install MongoDB]

[Install MongoDB]: https://www.mongodb.com/docs/v7.0/tutorial/install-mongodb-on-os-x/ 

1. Install the Xcode command-line tools by running the following command in your macOS Terminal:
      ```
      xcode-select --install
      ```

2. Install Homebrew
    macOS does not include the Homebrew brew package by default.
    Install brew using the official [Homebrew installation instructions]. 

[Homebrew installation instructions]: https://brew.sh/#install

3. Tap the MongoDB Homebrew Tap to download the official Homebrew formula for MongoDB and the Database Tools, by running the following command in your macOS Terminal:
   ```
   brew tap mongodb/brew
   ```
    
4. install MongoDB, run the following command in your macOS Terminal application:
   ```
   brew install mongodb-community@7.0
   ```
   
#### Install MongoDB Compas
Download [MongoDB Compass] for easy database editing:

[MongoDB Compass]: https://www.mongodb.com/products/tools/compass

#### Start Backend
      cd acla_backend
      npm install
      npm run start
      
#### Start Frontend
      cd acla_front
      npm install
      npm start


### For Windows:

