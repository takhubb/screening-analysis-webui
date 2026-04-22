# J-Quants Screening Analysis WebUI

`references` 配下のスクリーニング / リターン分析ロジックを参考にしつつ、条件指定から将来リターン分析までをブラウザで実行できる WebUI です。

## できること

- 出来高急増、高値更新、時価総額、PBR、自己資本比率、売上成長率、営業利益成長率、営業利益率などを条件指定
- 指定期間内のシグナルを抽出
- スクリーニング銘柄の 3 / 6 / 9 / 12 か月などの将来リターンを分析
- 市場別 / 業種別 / 年別の集計表示
- 最新シグナル一覧と全シグナル CSV のダウンロード

## 前提

- ルートの `.env` に `JQUANTS_API_KEY=...` が入っていること
- `references` は参照専用で、このアプリは変更しません

## 起動方法

```bash
docker compose up --build
```

起動後、ブラウザで `http://localhost:8501` を開いてください。

## 実装メモ

- 価格・銘柄マスタ・財務サマリは J-Quants の bulk を使って `.cache/jquants` にキャッシュします
- bulk の価格 CSV は調整後価格を持たないため、`AdjFactor` から `AdjC` / `AdjVo` を復元します
- 将来リターンは `references/stock-analysis` に合わせ、暦月ベースで算出し、非営業日は直後営業日に寄せます
- 財務条件は `references/ten-bagger-screening` の考え方をベースに、シグナル日時点で利用可能な最新開示を `merge_asof` で結合します
