# Rice Yield Prediction

I developed this system by training a machine learning model on agricultural crop data and deploying it as a Flask web application. I tested the model locally, added form validation, and built an easy-to-use interface for interacting with the trained model.

This project implements an end-to-end machine learning workflow including:
- **Data preprocessing** — cleaning, imputing missing values, numeric/categorical handling.
- **Model Training (Random Forest Regressor)** — trained using rainfall, pesticide usage, average temperature, and other crop-related variables.
- **Deployment** — serving predictions through a REST + HTML interface powered by Flask.

## Features
- Accepts user inputs such as Area, Year, average_rain_fall_mm_per_year, pesticides_tonnes, and avg_temp.
- Returns a numerical predicted rice yield using the trained Random Forest model.
- Web-based UI with clean input handling and real-time displayed predictions.
- Includes exploratory data visualizations (histograms, boxplots, correlation matrix).
- Supports both HTML form submission and JSON-based API requests.

## Web Service Deployment
https://rice-yield-flask.onrender.com/predict

<img width="684" height="466" alt="image" src="https://github.com/user-attachments/assets/8cb3a6ff-b9c5-41a0-96c9-141c38db4446" />

Deployed the project to Render (free tier) since it provides a straightforward zero-devops workflow for Python/Flask web services. The service automatically rebuilds and redeploys on every Git push to the main branch.

## How to run locally
1. Ensure Python (>=3.10) is installed.  
2. Create and activate a virtual environment:

---

## Additional Resources

<details>
<summary><strong>Show Resources (Click to Expand)</strong></summary>
<br>

**Google Colab Notebook:**  
https://colab.research.google.com/drive/1z4Pacfr5SbvZOpXBF2gPcFuCzNL05RjU?usp=sharing

**Saved Images (EDA & Plots):**  
https://drive.google.com/drive/folders/1XF76db60F2K3DJDYokLjylCKBy9HUpUa?usp=drive_link

**Presentation (PPT):**  
https://docs.google.com/presentation/d/1Bd4zmlSXm5n8gnYN5mrROXOaWD4BKxxFyfndV0YxAmc/edit?usp=sharing

</details>
