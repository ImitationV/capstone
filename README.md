# Budget Buddy

_The Easy Financial Planner_

Budget Buddy aims to create a user-friendly web application to simplify personal financial management and analysis.
With an intuitive, minimalistic UI, users will easily be able to start and track saving goals, input their personal information, and have real-time tracking and analytics.

### MVP Pages

- Homepage/Landing Page
- Login Page/Space
- User Welcome Page/Overview/Dashboard
- Financial Planning + Spending Goal Input Page (Transactions)
- Sidebar + Full Screen Chatbot (Financial Recommendation)

  OR

- Stock Recommendation (using ML)
- Terms and Services Page

### Extra Features

- Settings
- Receipt Scanning

## Database Integration

Option 1:
PostgreSQL for Core Financial Data

- Userid
- Monthly Income
- Expense Categories (Maybe into different columns or a separate table)
- Transactions (Separate Table)

Firebase Data Connect: NOSQL

-- Data Connect is a cloud-based SQL Postgres Database --

- Chat History
- Real-time Data through API calls

## Run it locally

#### Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js:** (Recommended version: Latest LTS) - You can download it from [nodejs.org](https://nodejs.org/).
- **npm** (Node Package Manager) or **yarn** (alternative package manager) - npm comes with Node.js, and you can install yarn separately from [yarnpkg.com](https://yarnpkg.com/).

#### Installation

1.  **Clone the Repository (If you haven't already):**

    If you downloaded the project as a ZIP file, extract it to your desired location. If you cloned it using Git, skip this step.

2.  **Navigate to the Project Directory:**

    Open your terminal or command prompt and navigate to the project's root directory:

    ```bash
    cd your-project-directory
    ```

    Replace `your-project-directory` with the actual name of your project folder.

3.  **Install Dependencies:**

    **Front-end**

    **Using npm:**

    ```bash
    npm install
    ```

    **Using yarn:**

    ```bash
    yarn install
    ```

    This command will read the `package.json` file and install the necessary packages listed in the `dependencies` and `devDependencies` sections.

    **Backend**

    Check Python and PIP:

    ```bash
    python --version 
    pip --version
    ```

    Go to Backend Folder and Install Requirements
    ```bash
    cd /backend
    pip install -r requirements.txt
    ```

#### Running the Application

Once the dependencies are installed, you can start the development server:

**Using npm:**

```bash
npm run dev
```
