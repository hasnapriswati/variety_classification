# Versi sederhana yang strictly mengikuti flowchart
# ========================================
# STEP 9: Training XGBoost Meta-Learner (Tanpa SMOTE)
# ========================================
print("\n" + "="*60)
print("STEP 9: Training XGBoost Meta-Learner")
print("="*60)
start_time = time.time()

# XGBoost dengan parameter terbaik
xgb_meta = xgb.XGBClassifier(**best_params)

# Training tanpa SMOTE
eval_set = [(X_meta_train_selected, y_train_full), (X_meta_test_selected, y_test)]
xgb_meta.fit(
    X_meta_train_selected, y_train_full,
    eval_set=eval_set,
    eval_metric='mlogloss',
    early_stopping_rounds=50,
    verbose=False
)

print(f"XGBoost terbaik pada iterasi: {xgb_meta.best_iteration}")
print(f"Step 9 selesai dalam {time.time() - start_time:.2f} detik")

# ========================================
# STEP 10: Evaluasi & Ensemble Sederhana
# ========================================
print("\n" + "="*60)
print("STEP 10: Evaluasi & Ensemble Sederhana")
print("="*60)
start_time = time.time()

# Prediksi EfficientNet
eff_pred = np.argmax(X_test_prob, axis=1)
eff_acc = accuracy_score(y_test, eff_pred)

# Prediksi Meta XGBoost
meta_pred = xgb_meta.predict(X_meta_test_selected)
meta_acc = accuracy_score(y_test, meta_pred)

# Ensemble sederhana (sesuai flowchart)
meta_proba = xgb_meta.predict_proba(X_meta_test_selected)
eff_proba = X_test_prob
final_proba = 0.8 * meta_proba + 0.2 * eff_proba
final_pred = np.argmax(final_proba, axis=1)
final_acc = accuracy_score(y_test, final_pred)

print(f"\nAkurasi:")
print(f"EfficientNet (Base): {eff_acc*100:.3f}%")
print(f"XGBoost Meta-Learner: {meta_acc*100:.3f}%")
print(f"Ensemble Final (80% Meta + 20% Base): {final_acc*100:.3f}%")

# Classification Report
print("\nClassification Report (Ensemble Final):")
print(classification_report(y_test, final_pred, target_names=class_names))

print(f"Step 10 selesai dalam {time.time() - start_time:.2f} detik")