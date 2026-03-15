import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === 1️⃣ Define paths ===
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset', 'news_dataset.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 Reading data from: {DATA_PATH}")

# === 2️⃣ Load CSV ===
try:
    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading dataset: {e}")

print(f"🧾 Columns available: {list(df.columns)}")

# === 3️⃣ Standardize columns ===
# Handle both possible formats: ['label', 'text'] OR ['Headline', 'Content', 'Category']
if 'label' in df.columns and 'text' in df.columns:
    text_col = 'text'
    label_col = 'label'
elif 'Headline' in df.columns and 'Category' in df.columns:
    df = df.rename(columns={'Headline': 'text', 'Category': 'label'})
    text_col = 'text'
    label_col = 'label'
else:
    raise ValueError("❌ Could not find suitable text and label columns.")

# === 4️⃣ Drop missing values ===
df = df.dropna(subset=[text_col, label_col])
print(f"✅ Cleaned dataset shape: {df.shape}")

# === 5️⃣ Show unique categories ===
unique_labels = df[label_col].unique()
print(f"📊 Unique categories found: {unique_labels}")

# === 6️⃣ Encode labels ===
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[label_col])

# Build label mapping
label_mapping = {label: int(idx) for label, idx in zip(le.classes_, le.transform(le.classes_))}

print("\n🔢 Label mapping:")
for label, idx in label_mapping.items():
    print(f"  {label} → {idx}")

# === 7️⃣ Split into train/test ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])
print(f"\n✅ Train size: {len(train_df)} | Test size: {len(test_df)}")

# === 8️⃣ Save processed data ===
train_path = os.path.join(OUTPUT_DIR, 'train_prepared.csv')
test_path = os.path.join(OUTPUT_DIR, 'test_prepared.csv')
mapping_path = os.path.join(OUTPUT_DIR, 'label_mapping.json')

train_df[[text_col, 'label_encoded']].to_csv(train_path, index=False)
test_df[[text_col, 'label_encoded']].to_csv(test_path, index=False)

# Convert NumPy int64 → Python int for JSON serialization
label_mapping_serializable = {k: int(v) for k, v in label_mapping.items()}

with open(mapping_path, 'w', encoding='utf-8') as f:
    json.dump(label_mapping_serializable, f, ensure_ascii=False, indent=2)

print("\n💾 Saved processed data:")
print(f"  📁 Train: {train_path}")
print(f"  📁 Test:  {test_path}")
print(f"  📁 Label mapping: {mapping_path}")

print("\n🎉 Preprocessing complete! Ready for model training 🚀")
