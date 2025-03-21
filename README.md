# Vessel Game

## Setup & Installation

Follow these steps to set up and run the **Vessel Game** application.

### **1. Install Node.js and npm**
Ensure you have **Node.js** and **npm** installed. You can verify this by running:

```sh
node -v
npm -v
```

If not installed, download and install Node.js from [nodejs.org](https://nodejs.org/).

---
### **2. Install Dependencies**
Run the following command in the project root directory (where `package.json` is located):

```sh
npm install
```

This installs all required dependencies.

---
### **3. Set Up Python Virtual Environment (Preinstall Script)**
Since the **preinstall script** sets up a Python environment inside the `analytic/` directory, ensure Python 3 is installed.

Manually set up the virtual environment by running:

#### **For Mac/Linux**:
```sh
cd analytic
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

#### **For Windows**:
```sh
cd analytic
python3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

(The preinstall script should do this automatically, but you can manually verify it.)

---
### **4. Start the Server**
Run the following command to start the Express server (`server.js`):

```sh
npm start
```

The server should now be running at `http://localhost:3000`.

---
### **5. Running the Analytics Script (Optional)**
To manually execute the Python analytics script:

```sh
npm run analytic
```

This runs `main.py` inside the `analytic/` directory while activating the virtual environment.

---
## **Troubleshooting**

### **Permission Errors on Linux/Mac**
If you get a permission error, try:
```sh
chmod +x analytic/venv/bin/activate
```

### **Server Fails to Start**
If `npm start` fails, check if `server.js` is properly configured to run an Express app.

### **Dependency Issues**
If `npm install` doesn’t work, delete `package-lock.json` and `node_modules/`, then reinstall:

```sh
rm -rf node_modules package-lock.json
npm install
```

---

### **Project Structure**
```
/vessel-game
│── analytic/        # Analytics module with Python scripts
│── data/            # Stores game-related data
│── public/          # Frontend assets
│── server.js        # Express backend server
│── package.json     # Project dependencies and scripts
│── README.md        # Setup instructions
```

Now you’re ready to run and explore the Vessel Game! 🚀

