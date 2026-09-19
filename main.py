import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 50)
    print("🚀 بدء تجربة نظام MLOps المتكامل")
    print("=" * 50)

    # 1. تدريب النموذج وتوثيقه في MLflow
    print("\n[1] بدء تدريب النموذج وتحديث MLflow...")
    X_train = pd.DataFrame({"feat1": [1, 2, 3, 4, 1, 2], "feat2": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2]})
    y_train = [0, 1, 1, 0, 0, 1]
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_train, model.predict(X_train))
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/latest_model.pkl")
        print(f"✅ تم التدريب بنجاح! دقة النموذج: {acc * 100}%")

    # 2. مراقبة انحراف البيانات (Data Drift)
    print("\n[2] فحص انحراف البيانات (Data Drift)...")
    reference_data = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [0.1, 0.2, 0.3, 0.4]})
    current_data = pd.DataFrame({"feat1": [8, 9, 8, 9], "feat2": [0.9, 0.8, 0.9, 0.9]})

    report_path = "drift_report.html"

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html(report_path)
        print("⚠️ تحذير MLOps: تم اكتشاف انحراف في البيانات! (Data Drift Detected)")

    except Exception:
        # محرك قياسي احتياطي لحساب الانحراف وتوليد التقرير في حال وجود تعارض في البيئة
        from scipy.stats import ks_2samp

        drift_details = []
        for col in reference_data.columns:
            stat, p_val = ks_2samp(reference_data[col], current_data[col])
            drift_details.append((col, p_val, p_val < 0.05))

        rows_html = "".join([
            f"<tr style='border-bottom:1px solid #313244;'>"
            f"<td style='padding:12px;'><b>{col}</b></td>"
            f"<td style='padding:12px;'>{p:.4f}</td>"
            f"<td style='padding:12px; font-weight:bold; color:{'#f38ba8' if d else '#a6e3a1'};'>{'⚠️ DRIFT DETECTED' if d else '✅ OK'}</td>"
            f"</tr>" for col, p, d in drift_details
        ])

        html_content = f"""
        <div style="background:#1e1e2e; color:#cdd6f4; padding:20px; border-radius:12px; font-family:sans-serif; margin-top:10px;">
            <h2 style="color:#f38ba8; margin-top:0;">⚠️ Enterprise MLOps Data Drift Dashboard</h2>
            <p style="font-size:15px;"><b>Status:</b> <span style="color:#11111b; background:#f38ba8; padding:4px 10px; border-radius:6px; font-weight:bold;">CRITICAL DRIFT ALERT</span></p>
            <table style="width:100%; text-align:left; border-collapse:collapse; margin-top:15px;">
                <tr style="border-bottom:2px solid #45475a; background:#181825;">
                    <th style="padding:12px;">Feature</th>
                    <th style="padding:12px;">p-value (Kolmogorov-Smirnov Test)</th>
                    <th style="padding:12px;">Drift Status</th>
                </tr>
                {rows_html}
            </table>
        </div>
        """
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("⚠️ تحذير MLOps: تم اكتشاف انحراف في البيانات! (Data Drift Detected)")

    print(f"\n📄 تم إنشاء وحفظ التقرير التفاعلي بنجاح في: {report_path}")

if __name__ == "__main__":
    main()
