import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Phase 2: Data Collection ---
df = pd.read_csv("employee_data.csv")
print(f"Shape: {df.shape}")
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# --- Phase 3: Data Preparation & Storage ---
df["HireDate"] = pd.to_datetime(df["HireDate"])
df["TenureYears"] = (pd.Timestamp("2024-01-01") - df["HireDate"]).dt.days / 365.25

Q1, Q3 = df["Salary"].quantile([0.25, 0.75])
IQR = Q3 - Q1
df["Salary"] = df["Salary"].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

edu_order = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "PhD": 5}
df["EducationLevel"] = df["Education"].map(edu_order)

# --- Phase 4: Data Processing & Analysis ---
dept_salary = df.groupby("Department")["Salary"].mean().sort_values(ascending=True)
edu_perf = df.groupby("Education")["PerformanceScore"].mean()
status_dist = df["Status"].value_counts()

print("\nAverage Salary by Department:\n", dept_salary.sort_values(ascending=False).round(2))
print("\nPerformance by Education:\n", edu_perf.round(2))
print("\nCorrelation Matrix:\n", df[["Age", "Salary", "PerformanceScore", "YearsExperience", "TrainingHours", "EducationLevel"]].corr().round(3))

# --- Phase 5: Visualisation ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Lab 1 — Employee Data Analytics Dashboard", fontsize=16, fontweight="bold")

# 1. Bar chart: Average Salary by Department
axes[0, 0].barh(dept_salary.index, dept_salary.values, color="steelblue", edgecolor="black")
axes[0, 0].set_title("Average Salary by Department")
axes[0, 0].set_xlabel("Salary (USD)")

# 2. Histogram: Distribution of Performance Scores
axes[0, 1].hist(df["PerformanceScore"], bins=15, color="coral", edgecolor="black")
axes[0, 1].set_title("Distribution of Performance Scores")
axes[0, 1].set_xlabel("Performance Score")
axes[0, 1].set_ylabel("Frequency")

# 3. Scatter plot: Salary vs Years of Experience
axes[1, 0].scatter(df["YearsExperience"], df["Salary"], color="seagreen", alpha=0.45)
axes[1, 0].set_title("Salary vs Years of Experience")
axes[1, 0].set_xlabel("Years of Experience")
axes[1, 0].set_ylabel("Salary (USD)")

# 4. Pie chart: Employee Status Distribution
axes[1, 1].pie(status_dist.values, labels=status_dist.index, autopct="%1.1f%%", startangle=90)
axes[1, 1].set_title("Employee Status Distribution")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("lab1_dashboard.png", dpi=150)
plt.show()