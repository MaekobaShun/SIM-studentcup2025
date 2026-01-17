# kabetani-yusei README
# はじめに
- ベースラインコードです
- どれくらいスコアが出るのかの確認です
# 環境構築

# やりたいこと
- 生成させるllmのプロンプトチューニングするのはあり

# メモ
Qwen/Qwen3-Embedding-0.6Bで埋め込み表現にして近いやつを引っ張ってくる
```
TOP2:
  Both correct rate: 0.1000 (2/20)
  Individual correct rate: 0.4500 (18/40)
  AUC: 0.8915

TOP5:
  Both correct rate: 0.3000 (6/20)
  Individual correct rate: 0.6000 (24/40)
  AUC: 0.8915

TOP10:
  Both correct rate: 0.7000 (14/20)
  Individual correct rate: 0.8500 (34/40)
  AUC: 0.8915
```

TOP5までに2つが含まれているかを見ても，精度が悪いため，
埋め込み表現でそのままやるのは効果が薄いのかも