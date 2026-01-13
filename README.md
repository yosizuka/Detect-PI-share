# 実行環境

```
uv 0.9.7
```

## 実行準備

```
uv venv -p 3.11
```

```
uv pip sync requirements.txt
```

## .env

`.env`ファイルを作成してOpenAIのAPI keyを設定

```
OPENAI_API_KEY="your_api_key"
```

# 分類機の学習(時間がかかります)

```
uv run --env-file=.env train_clf.py
```

# 実験スクリプトの実行

```
uv run --env-file=.env evaluate.py
```

## 結果ファイルの出力

```
./run.sh
```