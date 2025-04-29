# ACLA


## Installation frontend and backend enviornments


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






## Design Desktop Application Prerequisite
https://github.com/ParthJadhav/Tkinter-Designer/blob/master/docs/instructions.md
### 1. Install Python

Before using Tkinter Designer, you'll need to install Python.  
- [Here is a link to the Python downloads page.](https://www.python.org/downloads)  
- [Here is a helpful guide to installing Python on various operating systems.](https://wiki.python.org/moin/BeginnersGuide/Download)

*Later in this guide, you will use the Package Installer for Python (pip), which may require you to add Python to your system PATH.*

___
<br>

<a id="getting-started-2"></a>

### 2. Install Tkinter Designer

https://github.com/ParthJadhav/Tkinter-Designer/blob/master/docs/instructions.md

*Three options:*

1. `pip install tkdesigner`


2. To run Tkinter Designer from the source code, follow the instructions below.

   1. Download the source files for Tkinter Designer by downloading it manually or using GIT.

      ` git clone https://github.com/ParthJadhav/Tkinter-Designer.git `

   2. Change your working directory to Tkinter Designer.

      `cd Tkinter-Designer`

   3. Install the necessary dependencies by running

      - `pip install -r requirements.txt`
         - In the event that pip doesn't work, also try the following commands:
         - `pip3 install -r requirements.txt`
         - `python -m pip install -r requirements.txt`
         - `python3 -m pip install -r requirements.txt`
         - If this still doesn't work, ensure that Python is added to the PATH.

   This will install all requirements and Tkinter Designer. Before you use Tkinter Designer you need to create a Figma File with the below instructions.

   If you already have created a file then skip to [**Using Tkinter Designer**](#Using-Tkinter-Designer) Section.

___
<br>

<a id="getting-started-3"></a>

### 3. Make a Figma Account

1. In a web browser, navigate to [figma.com](https://www.figma.com/) and click 'Sign up'
2. Enter your information, then verify your email
3. Create a new Figma Design file
4. Get started making your GUI
   - The next section covers required formatting for Tkinter Designer input.
     - [Here is the official Figma tutorial series for beginners.](https://www.youtube.com/watch?v=Cx2dkpBxst8&list=PLXDU_eVOJTx7QHLShNqIXL1Cgbxj7HlN4)
     - [Here is the official Figma YouTube channel.](https://www.youtube.com/c/Figmadesign/featured)
     - [Here is the Figma Help Center.](https://help.figma.com/hc/en-us)

<br><br>

<a id="formatting-1"></a>

### Using Tkinter Designer <small>[[Top](#table-of-contents)]</small>

<a id="using-1"></a>

#### 1. Personal Access Token

1. Log into your Figma account
2. Navigate to Settings
3. In the **Account** tab, scroll down to **Personal access tokens**
4. Enter the name of your access token in the entry form and press <kbd>Enter</kbd>
5. Your personal access token will be created.
   - Copy this token and keep it somewhere safe.
   - **You will not get another chance to copy this token.**

<a id="using-2"></a>

#### 2. Getting your File URL

1. In your Figma design file, click the **Share** button in the top bar, then click on **&#x1f517; Copy link**

<a id="using-cli"></a>

