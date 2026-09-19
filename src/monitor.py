import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift():
    print("🔍 Inspecting Data Drift via Evidently AI...")
    reference_data = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [0.1, 0.2, 0.3, 0.4]})
    current_data = pd.DataFrame({"feat1": [8, 9, 8, 9], "feat2": [0.9, 0.8, 0.9, 0.9]}) 

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    os.makedirs("monitoring_reports", exist_ok=True)
    report_path = "monitoring_reports/drift_report.html"
    report.save_html(report_path)

    drift_result = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    if drift_result:
        print("⚠️ CRITICAL ALERT: Data Drift Detected!")
        return True
    
    print("✅ Data Distributions Stable.")
    return False

if __name__ == "__main__":
    check_drift()
  
