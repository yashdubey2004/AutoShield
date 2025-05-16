import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import tensorflow as tf
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # change this in production
CORS(app)

# -------------------- Database Configuration --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///accounts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------- Models --------------------
class FakeAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account_data = db.Column(db.Text, nullable=False)
    fake_percentage = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detection_time = db.Column(db.Float, nullable=False)
    accounts_processed = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# -------------------- ML Model Pipeline (your new code) --------------------
dataset_path = os.path.join(os.path.dirname(__file__), "social_media_accounts_dataset.csv")
df = pd.read_csv(dataset_path)

def engineer_features(df):
    numeric_cols = ['followers', 'following', 'engagement_rate', 'post_frequency', 'posts', 'account_age_days', 'bio_length']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna({col: 0 for col in numeric_cols}, inplace=True)
    df['has_external_link'] = df['has_external_link'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    df['has_profile_picture'] = df['has_profile_picture'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    df['is_verified'] = df['is_verified'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    df['suspicious_activity'] = pd.to_numeric(df['suspicious_activity'], errors='coerce').fillna(0)
    df['profile_score'] = df['has_profile_picture'].astype(int) + (df['bio_length'] > 0).astype(int) + df['is_verified'].astype(int)
    df['engagement_quality'] = np.log1p(df['engagement_rate'] * df['post_frequency'])
    df['follower_trust'] = np.where(df['followers'] > 0, df['followers'] / (df['following'] + df['followers'] + 1), 0.0)
    df['activity_score'] = np.sqrt(df['post_frequency']) * np.log1p(df['account_age_days'])
    df['post_to_follower_ratio'] = df['posts'] / (df['followers'] + 1)
    df['suspicious_activity_score'] = (df['suspicious_activity'].astype(int) +
                                       df['has_external_link'].astype(int) +
                                       (df['profile_score'] < 2).astype(int))
    df['follower_following_ratio'] = df['followers'] / (df['following'] + 1)
    if 'account_type' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['account_type_encoded'] = le.fit_transform(df['account_type'])
    return df

df = engineer_features(df)
df.fillna({
    'bio_length': 0,
    'engagement_rate': df['engagement_rate'].median(),
    'post_frequency': df['post_frequency'].median(skipna=True),
    'posts': 0
}, inplace=True)
if 'is_fake' not in df.columns:
    df['is_fake'] = np.where(
        (df.get('suspicious_activity', 0) == 1) |
        ((df.get('followers', 0) > 1000) & (df.get('engagement_rate', 1) < 0.01)) |
        ((df.get('has_profile_picture', 1) == 0) & (df.get('bio_length', 0) < 10)),
        1, 0
    )

all_features = [
    'followers', 'following', 'bio_length', 'has_profile_picture',
    'is_verified', 'engagement_rate', 'post_frequency', 'account_age_days',
    'profile_score', 'engagement_quality', 'follower_trust', 'activity_score',
    'post_to_follower_ratio', 'external_link_indicator', 'suspicious_activity_score',
    'bot_like_behavior', 'follower_following_ratio', 'bio_length_norm',
    'engagement_anomaly', 'post_frequency_anomaly', 'account_age_norm'
]
if 'account_type_encoded' in df.columns:
    all_features.append('account_type_encoded')
features = [col for col in all_features if col in df.columns]

X = df[features]
y = df['is_fake']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Selected Features: {features}")

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)

def add_noise(X, y, noise_level=0.3, label_noise_ratio=0.3):
    np.random.seed(42)
    noise = np.random.normal(loc=0, scale=noise_level * X.std(axis=0), size=X.shape)
    X_noisy = X + noise
    flip_indices = np.random.choice(len(y), int(len(y) * label_noise_ratio), replace=False)
    y_noisy = y.copy()
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    return X_noisy, y_noisy

X_resampled, y_resampled = add_noise(X_resampled, y_resampled, 0.3, 0.3)

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_resampled_pca = pca.fit_transform(X_resampled)
X_test_pca = pca.transform(X_test)
print(f"Total PCA components selected: {X_resampled_pca.shape[1]}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def create_nn():
    model = Sequential([
        Dense(16, activation='relu', kernel_regularizer=l2(0.02), input_shape=(X_resampled_pca.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(8, activation='relu', kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
nn_model = create_nn()
history = nn_model.fit(
    X_resampled_pca, y_resampled,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    # Simple dummy login logic
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        # Here you should validate credentials; for now, we assume any input is valid.
        session["user"] = email
        return redirect(url_for("profile"))
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    # Simple dummy signup logic
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")
        # In production, create the user account.
        session["user"] = email
        return redirect(url_for("profile"))
    return render_template("signup.html")

@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))
    # In production, fetch real stats; here we use dummy data.
    stats = {
        "flaggedAccounts": 100,
        "totalChecked": 1000,
        "dangerousAccounts": 30,
        "timeRequired": "0:45"
    }
    return render_template("profile.html", stats=stats)

@app.route("/input", methods=["GET"])
def input_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("inputpage.html")

@app.route("/detect", methods=["POST"])
def detect_accounts():
    # This endpoint processes the account data using the ML model
    # Expecting JSON payload from a fetch/AJAX call.
    data = request.get_json()
    accounts = data.get("accounts", [])
    if not accounts:
        return jsonify({"error": "No account data provided."}), 400

    start_time = time.time()
    df_input = pd.DataFrame(accounts)
    required_cols = [
        "followers", "following", "engagement_rate", "post_frequency",
        "posts", "account_age_days", "bio_length", "has_external_link",
        "has_profile_picture", "is_verified", "suspicious_activity"
    ]
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    if missing_cols:
        return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

    df_fe = engineer_features(df_input.copy())
    for feat in features:
        if feat not in df_fe.columns:
            df_fe[feat] = 0
        df_fe[feat] = pd.to_numeric(df_fe[feat], errors="coerce").fillna(0)

    X_pred = df_fe[features].values
    X_pred_scaled = scaler.transform(X_pred)
    X_pred_pca = pca.transform(X_pred_scaled)

    try:
        preds = nn_model.predict(X_pred_pca)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    preds = np.squeeze(preds)
    if np.ndim(preds) == 0:
        preds = np.array([preds])
    
    results = []
    for idx, pred in enumerate(preds):
        account_result = accounts[idx].copy()
        fake_percentage = float(pred * 100)
        account_result["fake_percentage"] = fake_percentage
        account_result["is_fake"] = fake_percentage >= 50.0
        results.append(account_result)
        if fake_percentage >= 50.0:
            account_json = json.dumps(accounts[idx])
            fake_acc = FakeAccount(account_data=account_json, fake_percentage=fake_percentage)
            db.session.add(fake_acc)
    
    end_time = time.time()
    detection_time = end_time - start_time
    log = DetectionLog(detection_time=detection_time, accounts_processed=len(accounts))
    db.session.add(log)
    db.session.commit()

    # Save results in session so flaggedaccounts page can render them
    session["results"] = results
    return jsonify({"results": results, "detection_time": detection_time})

@app.route("/flagged", methods=["GET"])
def flagged_accounts():
    if "user" not in session:
        return redirect(url_for("login"))
    results = session.get("results", [])
    return render_template("flaggedaccounts.html", results=results)

@app.route("/metrics", methods=["GET"])
def get_metrics():
    total_flagged = FakeAccount.query.count()
    total_reports = db.session.query(db.func.sum(DetectionLog.accounts_processed)).scalar() or 0
    high_risk = FakeAccount.query.filter(FakeAccount.fake_percentage >= 75).count()
    avg_detection_time = db.session.query(db.func.avg(DetectionLog.detection_time)).scalar() or 0
    metrics = {
        "total_flagged": total_flagged,
        "total_reports": total_reports,
        "high_risk": high_risk,
        "avg_detection_time": avg_detection_time
    }
    return jsonify(metrics)

if __name__ == "__main__":
    app.secret_key = "supersecretkey"  # set a secure key in production
    with app.app_context():
        db.create_all()
    app.run(debug=True)
