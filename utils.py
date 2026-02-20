from openai import OpenAI
import os
import numpy as np
import ollama
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

# 設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# キャッシュディレクトリの設定
CACHE_DIR = Path("embeddings_cache")
CACHE_DIR.mkdir(exist_ok=True)

TRAIN_CACHE_FILE = CACHE_DIR / "train_embeddings.pkl"
MODEL_CACHE_FILE = CACHE_DIR / "trained_classifier.pkl"

# 定数
EMBEDDING_DIM = 1536  # text-embedding-3-smallの次元数
NUM_LLMS = 3  # 使用するLLMの数


def get_embedding(text: str) -> np.ndarray:
    """
    テキストの埋め込みを取得
    
    Args:
        text: 埋め込みを取得するテキスト
    
    Returns:
        埋め込みベクトル（numpy配列）
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


def get_llm_output(prompt: str) -> str:
    """
    GPT-4から出力を取得
    
    Args:
        prompt: LLMに渡すプロンプト
    
    Returns:
        LLMからの応答テキスト
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content


def get_local_llm_output(prompt: str, model: str) -> str:
    """
    ローカルLLMから出力を取得
    
    Args:
        prompt: LLMに渡すプロンプト
        model: 使用するモデル名
    
    Returns:
        LLMからの応答テキスト
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': prompt
            },
        ],
        think=False
    )
    return response['message']['content']


def get_llm_outputs_embeddings(prompt: str, models: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    プロンプトを複数のLLMに入力し、その出力の埋め込みを取得
    
    Args:
        prompt: 評価するプロンプト
        models: 使用するモデル名のリスト。None の場合はデフォルトで
                ["gpt-4", "deepseek-r1:7b", "llama3"] を使用する。
                "gpt-*" や "openai" で始まる識別子は OpenAI API（get_llm_output）を呼び、
                それ以外はローカル ollama モデル（get_local_llm_output）として扱います。
    
    Returns:
        (結合された埋め込み, LLM出力のリスト)のタプル
    """
    if models is None:
        models_list = ["gpt-4", "deepseek-r1:7b", "llama3"]
    else:
        models_list = models

    outputs = []

    for model in models_list:
        try:
            # OpenAI系モデルは get_llm_output を使う
            if model.lower().startswith("gpt") or model.lower().startswith("openai"):
                out = get_llm_output(prompt)
            else:
                # それ以外はローカル ollama モデルとして扱う
                out = get_local_llm_output(prompt, model)
            outputs.append(out)
        except Exception as e:
            print(f"[WARN] {model} error: {e}")
            outputs.append("")

    # 各出力の埋め込みを取得して結合
    output_embeddings = []
    for output in outputs:
        if output.strip():
            emb = get_embedding(output)
            output_embeddings.append(emb)
        else:
            # 空の場合はゼロベクトル
            emb = np.zeros(EMBEDDING_DIM)
            output_embeddings.append(emb)

    # すべての埋め込みを結合
    combined_embedding = np.concatenate(output_embeddings)

    return combined_embedding, outputs


def save_embeddings_cache(embeddings_data: Dict[str, Any], filepath: Path) -> None:
    """
    埋め込みデータをキャッシュファイルに保存
    
    Args:
        embeddings_data: 保存する埋め込みデータ
        filepath: 保存先のファイルパス
    """
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"Embeddings saved to {filepath}")
    print(f"  - Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def load_embeddings_cache(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    埋め込みデータをキャッシュファイルから読み込み
    
    Args:
        filepath: 読み込むファイルパス
    
    Returns:
        埋め込みデータの辞書、ファイルが存在しない場合はNone
    """
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Embeddings loaded from {filepath}")
        print(f"  - Training samples: {len(data['y_train'])}")
        print(f"  - Feature dimensions: {data['X_train'].shape[1]}")
        print(f"  - Created at: {data['metadata']['created_at']}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load cache: {e}")
        return None


def save_model(clf: Any, filepath: Path) -> None:
    """
    学習済みモデルを保存
    
    Args:
        clf: 学習済み分類器
        filepath: 保存先のファイルパス
    """
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {filepath}")
    print(f"  - Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def load_model(filepath: Path) -> Optional[Any]:
    """
    学習済みモデルを読み込み
    
    Args:
        filepath: 読み込むファイルパス
    
    Returns:
        学習済み分類器、ファイルが存在しない場合はNone
    """
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'rb') as f:
            clf = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return clf
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def get_classifier_prediction_and_llm_outputs(prompt: str, clf: Any, models: Optional[List[str]] = None) -> Tuple[int, List[str]]:
    """
    プロンプトとLLM出力を含めた分類器の予測を取得（使用するモデルを指定可能）
    
    Args:
        prompt: 評価するプロンプト
        clf: 学習済み分類器
        models: 使用するモデル名のリスト（Noneならデフォルトのモデルリストを使用）
    
    Returns:
        (予測クラス(0 or 1), LLM出力のリスト)のタプル
    """
    # プロンプトの埋め込み
    prompt_embedding = get_embedding(prompt)
    
    # LLM出力の埋め込み（models を渡す）
    output_embedding, outputs = get_llm_outputs_embeddings(prompt, models=models)
    
    # 結合
    combined_embedding = np.concatenate([prompt_embedding, output_embedding]).reshape(1, -1)
    
    # 予測クラスを取得
    prediction = int(clf.predict(combined_embedding)[0])
    
    return prediction, outputs


def create_metadata(num_samples: int, feature_dims: int) -> Dict[str, Any]:
    """
    メタデータを作成
    
    Args:
        num_samples: サンプル数
        feature_dims: 特徴量の次元数
    
    Returns:
        メタデータの辞書
    """
    return {
        'created_at': datetime.now().isoformat(),
        'num_samples': num_samples,
        'feature_dims': feature_dims
    }