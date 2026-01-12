# run_experiment.py
import argparse
import sys
import numpy as np
from pathlib import Path

# Đảm bảo python tìm thấy src
sys.path.append(str(Path(__file__).parent))

from src.auto_cluster import AutoClusterPipeline
from src.data_loader import DataLoader
from src.classifier import ClusterClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import os

def main():
    parser = argparse.ArgumentParser(description="HAC Auto-Cluster Experiment Runner")
    
    # 1. Dataset & Mode
    # [STYLE] Bỏ path mặc định hardcode để tránh lỗi trên máy khác
    parser.add_argument("--file", type=str, default="/home/nhannv/Hello/ICN/data_process/dataset_5percent_33classes_no_normalize.csv", help="Path to input CSV")
    parser.add_argument("--label", type=str, default="label", help="Label column name")
    parser.add_argument("--init", type=str, default="ver6", choices=["ver6", "legacy"], help="Center init mode")
    
    # 2. Search Range (Internal AutoCluster tìm K tốt nhất trong khoảng này)
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=10)
    
    # 3. Environment Config
    parser.add_argument("--use_ig", action="store_true", help="Enable Information Gain Feature Selection")
    parser.add_argument("--ig_k", type=int, default=40, help="Number of features to keep if IG is on")
    parser.add_argument("--norm", type=str, default="minmax", choices=["minmax", "zscore"], help="Normalization method")
    
    # 4. Classification Model
    parser.add_argument("--classifier", type=str, default="gb", 
                        choices=["gb", "rf", "dt", "svc", "lr"],
                        help="Classification model: gb=GradientBoosting, rf=RandomForest, dt=DecisionTree, svc=SVC, lr=LogisticRegression")
    
    # 5. System
    parser.add_argument("--max_mem", type=float, default=4.0, help="Max memory in GB")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 = all CPUs, 1 = sequential)")

    args = parser.parse_args()

    # Xác định số CPU cores để sử dụng
    n_cpus = os.cpu_count() or 1
    n_jobs_actual = n_cpus if args.n_jobs == -1 else max(1, args.n_jobs)
    
    # Map classifier name to model (sử dụng tham số mặc định của sklearn + n_jobs)
    classifier_map = {
        "gb": ("GradientBoosting", GradientBoostingClassifier(random_state=42)),  # GB không có n_jobs
        "rf": ("RandomForest", RandomForestClassifier(random_state=42, n_jobs=n_jobs_actual)),
        "dt": ("DecisionTree", DecisionTreeClassifier(random_state=42)),  # DT không có n_jobs
        "svc": ("SVC", SVC(random_state=42, probability=True)),  # SVC không có n_jobs
        "lr": ("LogisticRegression", LogisticRegression(random_state=42, n_jobs=n_jobs_actual))
    }
    classifier_name, base_classifier = classifier_map[args.classifier]
    
    print("\n" + "="*60)
    print(f"🧪 EXPERIMENT START")
    print(f"📂 File:      {args.file}")
    print(f"⚙️  Mode:      {args.init.upper()} (Range K={args.min_k}..{args.max_k})")
    print(f"🔍 Feature:   {'IG Enabled (Top ' + str(args.ig_k) + ')' if args.use_ig else 'Original Features'}")
    print(f"📏 Norm:      {args.norm} (trước khi split)")
    print(f"🤖 Classifier: {classifier_name}")
    print(f"💻 CPUs:       {n_cpus} cores (using {n_jobs_actual} jobs)")
    print("="*60 + "\n")
    
    try:
        # --- BƯỚC 1: Load dữ liệu ---
        # [NEW] Đổi logic: Normalize TRƯỚC → Split SAU (cả 2 luồng đều như vậy)
        print("⏳ Loading & Preprocessing data...")
        
        # Sử dụng DataLoader cho cả 2 trường hợp để đảm bảo consistency
        # Logic: Load → Remove constant → Normalize → Calculate IG → Split
        loader = DataLoader(log_level="INFO")
        X_train, X_test, y_train, y_test, ig_weights = loader.load_and_preprocess(
            file_path=args.file,
            label_column=args.label,
            normalize_method=args.norm,
            remove_constant=True,
            calculate_ig=args.use_ig,
            test_size=0.2,
            random_state=42
        )
        
        # Vì đây là experiment tìm số cụm (Unsupervised), ta nên gộp lại để chạy trên toàn bộ dữ liệu
        X_full = np.concatenate((X_train, X_test), axis=0)
        
        # Áp dụng IG weights nếu có
        if args.use_ig and ig_weights is not None:
             # Lấy top K features tốt nhất dựa trên weights (logic đơn giản hóa)
             # Hoặc nhân weights vào X như trong logic clustering cũ
             # Ở đây ta nhân weights trực tiếp để AutoCluster dùng
             # Ghép weight của train/test lại (lưu ý ig_weights trả về vector features, dùng chung cho cả tập)
             X_full = X_full * ig_weights
             X_train = X_train * ig_weights  # [NEW] Cũng áp dụng cho train/test
             X_test = X_test * ig_weights
             print(f"✅ Applied Information Gain weights")

        print(f"✅ Data ready: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")

        # --- BƯỚC 2: Khởi tạo Auto Pipeline ---
        auto = AutoClusterPipeline(
            min_clusters=args.min_k,
            max_clusters=args.max_k,
            center_init=args.init,
            optimize_params=True, # Luôn tối ưu theta/alpha
            max_memory_gb=args.max_mem,
            n_jobs=n_jobs_actual,  # [NEW] Thêm đa luồng
            log_level="INFO"
        )

        # --- BƯỚC 3: Chạy Experiment ---
        # [FIX] Truyền numpy array X vào thay vì đường dẫn file
        # Result trả về là object AutoClusterResult, KHÔNG phải tuple
        result = auto.run(X_full, selection_metric="silhouette")

        # --- BƯỚC 4: Hiển thị kết quả clustering ---
        print("\n" + "="*60)
        print(f"🏆 CLUSTERING RESULT")
        print(f"   Best K:         {result.best_n_clusters}")
        # Truy cập vào best_evaluation để lấy metrics
        print(f"   Best Silhouette:{result.best_evaluation.silhouette_score:.4f}")
        print(f"   Best Params:    θ={result.best_evaluation.theta:.4f}, α={result.best_evaluation.alpha:.4f}")
        print(f"   Total Time:     {result.total_time:.2f}s")
        print("="*60 + "\n")
        
        # In bảng chi tiết từ phương thức có sẵn
        print(result.summary())
        
        # --- BƯỚC 5: Training và đánh giá model classification ---
        print("\n" + "="*60)
        print("🤖 TRAINING CLASSIFICATION MODEL")
        print("="*60)
        
        # Khởi tạo classifier với số cụm tốt nhất
        classifier = ClusterClassifier(
            n_clusters=result.best_n_clusters,
            theta=result.best_evaluation.theta,
            alpha=result.best_evaluation.alpha,
            center_init=args.init,
            base_classifier=base_classifier,  # [NEW] Sử dụng model được chọn
            log_level="INFO"
        )
        
        # Train model
        print(f"⏳ Training model với {result.best_n_clusters} cụm...")
        classifier.fit(
            X_train, 
            y_train,
            information_gain_weights=ig_weights if args.use_ig else None
        )
        training_time = classifier.training_time
        print(f"✅ Training completed in {training_time:.2f}s")
        
        # Test và đánh giá
        print(f"⏳ Testing model...")
        prediction_result = classifier.predict(X_test, y_test)
        
        # --- BƯỚC 6: Hiển thị kết quả classification ---
        print("\n" + "="*60)
        print(f"📊 CLASSIFICATION RESULTS")
        print("="*60)
        print(f"   Accuracy:       {prediction_result.accuracy:.4f} ({prediction_result.accuracy*100:.2f}%)")
        print(f"   Precision:      {prediction_result.precision:.4f}")
        print(f"   Recall:         {prediction_result.recall:.4f}")
        print(f"   F1-Score:       {prediction_result.f1:.4f}")
        print(f"   Training Time:  {training_time:.2f}s")
        print(f"   Testing Time:   {prediction_result.total_time:.4f}s")
        print("="*60)
        
        # In classification report chi tiết
        print("\n📋 Classification Report:")
        print(prediction_result.classification_report)
        
        # Tóm tắt cuối cùng
        print("\n" + "="*60)
        print("🎯 FINAL SUMMARY")
        print("="*60)
        print(f"   Classifier:     {classifier_name}")
        print(f"   Best Clusters:  {result.best_n_clusters}")
        print(f"   Best Params:    θ={result.best_evaluation.theta:.4f}, α={result.best_evaluation.alpha:.4f}")
        print(f"   Silhouette:     {result.best_evaluation.silhouette_score:.4f}")
        print(f"   Accuracy:       {prediction_result.accuracy:.4f} ({prediction_result.accuracy*100:.2f}%)")
        print(f"   F1-Score:       {prediction_result.f1:.4f}")
        print(f"   Total Time:     {result.total_time + training_time + prediction_result.total_time:.2f}s")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()