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