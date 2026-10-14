# Interview Q&A (specific to this project)

**1. What does the project do?** It analyses network-flow records, classifies attacks (DoS, brute force, port scan, data exfiltration), flags unusual behaviour the classifier doesn't know, and raises alerts with severity and a reason. A Flask API serves predictions.

**2. Why simulated data?** I had no access to real corporate logs, so I built a generator modelled on NSL-KDD/CICIDS features, with realistic noise (missing values, duplicates, stealthy attacks). The pipeline accepts any CSV with the same columns, so real datasets can be swapped in.

**3. Why Random Forest + Isolation Forest?** Random Forest gives high-precision, named detections of known attacks. Isolation Forest is trained only on normal traffic, so it can flag new attack types — the two cover each other's weaknesses.

**4. Which metrics matter and why?** Recall (don't miss attacks) and precision (avoid alert fatigue). Accuracy alone misleads because 70% of traffic is normal. I report precision/recall/F1 and a confusion matrix.

**5. How did you handle imbalance?** `class_weight="balanced"`, stratified train/test split, and macro/weighted F1.

**6. How did you avoid data leakage?** Imputation and scaling live inside the sklearn pipeline and are fit only on training data; test rows are never used for fitting. The Isolation Forest only sees normal training rows.

**7. What feature engineering did you do?** Upload/download byte ratio (exfiltration), failed-login ratio (brute force), error-rate sum (scans/floods), night-time flag, log of bytes.

**8. What are the limitations?** Results come from simulated data and are optimistic; real traffic is noisier and attackers adapt. Next steps: real datasets, streaming ingestion, SIEM integration, autoencoders, model drift monitoring.

**9. How would you deploy it?** Wrap the saved pipeline in the Flask API (already done), containerise with Docker, and put it behind a queue/stream consumer; retrain regularly on fresh labelled data.

**10. Explain it to a non-technical person.** It's like a smart security guard that has learned what normal activity looks like and what break-ins look like, watches every connection, and rings an alarm — saying how serious and why — when something looks wrong.
