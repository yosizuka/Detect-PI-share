import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset
from tqdm import tqdm
import argparse
from typing import Tuple, List

# 共通ユーティリティをインポート
from utils import (
    get_embedding,
    get_llm_outputs_embeddings,
    save_embeddings_cache,
    load_embeddings_cache,
    save_model,
    load_model,
    create_metadata,
    TRAIN_CACHE_FILE,
    MODEL_CACHE_FILE,
    EMBEDDING_DIM,
    NUM_LLMS
)

def generate_embeddings(prompts: List[str], labels: List[int], num_samples: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    埋め込みを生成
    
    Args:
        prompts: プロンプトのリスト
        labels: ラベルのリスト
        num_samples: 処理するサンプル数
    
    Returns:
        (特徴量行列, ラベル配列, 使用したプロンプトのリスト)のタプル
    """
    print(f"\n{'='*70}")
    print(f"Generating embeddings for {num_samples} samples...")
    print(f"{'='*70}\n")
    
    X_train_list = []
    y_train = np.array(labels[:num_samples])
    prompts_used = []

    for i in tqdm(range(num_samples), desc="Processing"):
        prompt = prompts[i]
        prompts_used.append(prompt)
        
        try:
            # プロンプトの埋め込み
            prompt_embedding = get_embedding(prompt)
            
            # LLM出力の埋め込み
            output_embedding, _ = get_llm_outputs_embeddings(prompt)
            
            # 結合
            combined_embedding = np.concatenate([prompt_embedding, output_embedding])
            X_train_list.append(combined_embedding)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process sample {i}: {e}")
            # エラー時はゼロベクトルで埋める
            zero_embedding = np.zeros(EMBEDDING_DIM * (1 + NUM_LLMS))
            X_train_list.append(zero_embedding)

    X_train = np.vstack(X_train_list)
    
    print("\nEmbeddings generated successfully")
    print(f"  - Shape: {X_train.shape}")
    print(f"  - Memory: {X_train.nbytes / 1024 / 1024:.2f} MB")
    
    return X_train, y_train, prompts_used

def train_classifier_from_embeddings(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    埋め込みから分類器を学習
    
    Args:
        X_train: 特徴量行列
        y_train: ラベル配列
    
    Returns:
        学習済みRandomForest分類器
    """
    print(f"\n{'='*70}")
    print("Training Random Forest Classifier...")
    print(f"{'='*70}\n")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    # 学習データでの精度を表示
    train_score = clf.score(X_train, y_train)
    print("\nTraining completed")
    print(f"  - Training accuracy: {train_score:.3f}")
    
    return clf

def main(num_samples: int = 100, force_refresh: bool = False) -> None:
    """
    メイン処理
    
    Args:
        num_samples: 学習サンプル数
        force_refresh: キャッシュを無視して再計算するか
    """
    print(f"\n{'='*70}")
    print("Adversarial Prompt Detection - Training Script")
    print(f"{'='*70}")
    print(f"Samples: {num_samples}")
    print(f"Force refresh: {force_refresh}")
    print(f"{'='*70}\n")
    
    # データセットの読み込み
    print("Loading dataset...")
    dataset = load_dataset("imoxto/prompt_injection_cleaned_dataset-v2")
    train_data = dataset["train"]
    prompts = train_data["text"]
    labels = train_data["labels"]
    print(f"Dataset loaded: {len(prompts)} samples\n")
    
    # 埋め込みの準備
    if not force_refresh:
        cached_data = load_embeddings_cache(TRAIN_CACHE_FILE)
        if cached_data is not None:
            X_train = cached_data['X_train']
            y_train = cached_data['y_train']
            print("\nUsing cached embeddings\n")
        else:
            X_train, y_train, prompts_used = generate_embeddings(prompts, labels, num_samples)
            
            # キャッシュに保存
            embeddings_data = {
                'X_train': X_train,
                'y_train': y_train,
                'prompts': prompts_used,
                'metadata': create_metadata(num_samples, X_train.shape[1])
            }
            save_embeddings_cache(embeddings_data, TRAIN_CACHE_FILE)
    else:
        print("Force refresh: Regenerating embeddings...\n")
        X_train, y_train, prompts_used = generate_embeddings(prompts, labels, num_samples)
        
        # キャッシュに保存
        embeddings_data = {
            'X_train': X_train,
            'y_train': y_train,
            'prompts': prompts_used,
            'metadata': create_metadata(num_samples, X_train.shape[1])
        }
        save_embeddings_cache(embeddings_data, TRAIN_CACHE_FILE)
    
    # モデルの学習
    if not force_refresh:
        clf = load_model(MODEL_CACHE_FILE)
        if clf is not None:
            print("\nUsing cached model\n")
        else:
            clf = train_classifier_from_embeddings(X_train, y_train)
            save_model(clf, MODEL_CACHE_FILE)
    else:
        print("\nForce refresh: Retraining model...\n")
        clf = train_classifier_from_embeddings(X_train, y_train)
        save_model(clf, MODEL_CACHE_FILE)
    
    print(f"\n{'='*70}")
    print("Training completed successfully!")
    print(f"{'='*70}")
    print("Cache files:")
    print(f"  - Embeddings: {TRAIN_CACHE_FILE}")
    print(f"  - Model: {MODEL_CACHE_FILE}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adversarial prompt classifier")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of training samples (default: 100)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regenerate embeddings and retrain model"
    )
    
    args = parser.parse_args()
    
    main(num_samples=args.num_samples, force_refresh=args.force_refresh)