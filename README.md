# ACLA


Installation 


For Mac:

Initial Start:

  Install Mongo DB server(https://www.mongodb.com/docs/v7.0/tutorial/install-mongodb-on-os-x/):
    Install the Xcode command-line tools by running the following command in your macOS Terminal:
      xcode-select --install
  Install Homebrew
    macOS does not include the Homebrew brew package by default.
    Install brew using the official Homebrew installation instructions. https://brew.sh/#install

  Tap the MongoDB Homebrew Tap to download the official Homebrew formula for MongoDB and the Database Tools, by running the following command in your macOS Terminal:
    brew tap mongodb/brew
    
  install MongoDB, run the following command in your macOS Terminal application:
    brew install mongodb-community@7.0

  Download MongoDB Compass for easy database editing:
    https://www.mongodb.com/products/tools/compass
    
  Open Command line:
    cd acla_backend
    npm install
    npm run start

    cd ..
    cd acla_front
    npm install
    npm start

    
