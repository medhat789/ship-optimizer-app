# Ship Route & Fuel Optimization Flask App - Deployment Guide (Render.com)

This guide provides step-by-step instructions for deploying the Flask-based ship route and fuel optimization application to Render.com using a GitHub repository.

**Project Structure:**

```
/
|-- src/                  # Main application code
|   |-- __init__.py
|   |-- main.py           # Flask app entry point
|   |-- models/           # Data models (if any)
|   |-- routes/           # Route definitions (if any)
|   |-- static/           # CSS, JS files
|   |-- templates/        # HTML templates
|   |-- model_files/      # Machine learning models and related data
|-- venv/                 # Virtual environment (used locally, not deployed)
|-- requirements.txt      # Python dependencies
|-- runtime.txt           # Specifies Python version for Render
|-- render.yaml           # Render deployment configuration
|-- README.md             # This deployment guide
```

**Prerequisites:**

1.  **Git:** Ensure Git is installed on your local machine.
2.  **GitHub Account:** You need a GitHub account to host the code repository.
3.  **Render Account:** You need a Render.com account to deploy the application.

**Deployment Steps:**

1.  **Create a GitHub Repository:**
    *   Go to GitHub and create a new repository (e.g., `ship-optimizer-app`). It can be public or private.
    *   Do **not** initialize it with a README, .gitignore, or license yet.

2.  **Initialize Git Locally and Push Code:**
    *   Open a terminal or command prompt on your local machine.
    *   Navigate to the project directory (`/home/ubuntu/ship_optimizer_app_deploy` in the context where this was prepared, or wherever you have extracted the final ZIP file).
    *   Initialize a Git repository:
        ```bash
        git init -b main
        ```
    *   Add all the project files:
        ```bash
        git add .
        ```
    *   Commit the files:
        ```bash
        git commit -m "Initial commit of ship optimizer app"
        ```
    *   Add the GitHub repository as a remote origin (replace `<YourGitHubUsername>` and `<YourRepoName>`):
        ```bash
        git remote add origin https://github.com/<YourGitHubUsername>/<YourRepoName>.git
        ```
    *   Push the code to GitHub:
        ```bash
        git push -u origin main
        ```

3.  **Deploy on Render.com:**
    *   Log in to your Render account.
    *   Click **New +** and select **Web Service**.
    *   Connect your GitHub account if you haven't already.
    *   Choose the GitHub repository you just created (`ship-optimizer-app` or your chosen name).
    *   Render will detect the `render.yaml` file. Click **Approve** to use its settings.
    *   Alternatively, if you don't use `render.yaml` or want to configure manually:
        *   **Name:** Give your service a name (e.g., `ship-optimizer`).
        *   **Region:** Choose a region.
        *   **Branch:** Select `main` (or your default branch).
        *   **Runtime:** Select `Python 3`.
        *   **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
        *   **Start Command:** `gunicorn --chdir src main:app --bind 0.0.0.0:$PORT`
        *   **Instance Type:** Choose a plan (Free tier might work for testing, but consider paid plans for production due to resource needs).
        *   Under **Advanced**, add an Environment Variable:
            *   **Key:** `PYTHON_VERSION`
            *   **Value:** `3.11.9` (or the version specified in `runtime.txt`)
    *   Click **Create Web Service**.

4.  **Wait for Deployment:**
    *   Render will clone your repository, install dependencies, and start the application.
    *   Monitor the deployment logs in the Render dashboard.
    *   Once deployed, Render will provide a public URL (e.g., `https://ship-optimizer.onrender.com`).

**Important Notes & Known Issues:**

*   **Model Loading Warning:** During local testing, a warning `Error loading model: STACK_GLOBAL requires str` appeared, followed by `Using fallback prediction model`. This suggests an issue with loading the primary `.joblib` or `.pkl` model file, potentially related to serialization or library versions between model training and application runtime. The application might function with the fallback, but its accuracy or performance could be affected. Investigate `src/main.py` around the model loading logic to ensure the correct model is loaded and the serialization method is compatible.
*   **Datetime Handling Error:** An error `TypeError: can't subtract offset-naive and offset-aware datetimes` occurred when clicking the "Optimize Voyage" button. This typically happens when mixing timezone-aware datetime objects (e.g., from user input or external APIs) with timezone-naive ones (e.g., `datetime.now()` without timezone info). Review the date/time handling logic in `src/main.py`, specifically where `required_arrival_time` and `now` are calculated or compared within the `optimize_route` function. Ensure all datetime objects involved in calculations have consistent timezone information (either both naive or both aware, preferably UTC aware).
*   **Dependencies:** This application relies on libraries with native C extensions (`numpy`, `pandas`, `scikit-learn`, `scipy`). Render.com supports these, unlike serverless platforms. Ensure the build process completes successfully on Render.
*   **Resource Usage:** Machine learning models can be memory-intensive. Monitor resource usage on Render and upgrade your instance type if necessary.

**Updating the Application:**

1.  Make changes to your code locally.
2.  Commit the changes to Git:
    ```bash
    git add .
    git commit -m "Your update description"
    ```
3.  Push the changes to GitHub:
    ```bash
    git push origin main
    ```
4.  Render will automatically detect the push (if auto-deploy is enabled, which is the default) and trigger a new deployment.


