####################
#	1/4
# date_reading: 
# thought: 
# words: 
# reference: 

####################

import warnings
warnings.filterwarnings("ignore")



####################
#	2/4
# date_reading: 
# thought: 
# words: 
# reference: 

####################

import matplotlib.pyplot as plt
import japanize_matplotlib

from prophet import Prophet
import pandas as pd
from io import BytesIO

import sys
import os

class SuppressOutput:
    """標準出力と標準エラー出力を抑制するコンテキストマネージャ"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr



####################
#	3/4
# date_reading: 
# thought: 
# words: 
# reference: 

####################

def forecast_and_save_all_plots(data_path, city_names, output_dir):
    # データの読み込み
    df = pd.read_csv(data_path, encoding="shift_jis")
    
    # 数値データの変換
    def _convert_to_float(x):
        try:
            return float(x.replace(",", ""))
        except:
            return None
    df["value"] = df["value"].apply(_convert_to_float)

    # 保存先ディレクトリを作成（存在しない場合）
    os.makedirs(output_dir, exist_ok=True)

    for city_name in city_names:
        try:
            # 指定された市区町村のデータを抽出
            df_city = df[df["地域"] == city_name].copy()

            # 型変換
            df_city["value"] = df_city["value"].astype(float)
            df_city = df_city.rename(columns={"時間軸（月）": "ds", "value": "y"})
            
            # 日付形式の変換
            df_city["ds"] = pd.to_datetime(df_city["ds"].apply(
                lambda x: x.replace("年", "-").replace("月", "-01") 
                if len(x) == 8
                else x.replace("年", "-0").replace("月", "-01")
            ))
            
            # Prophetモデルの作成と学習
            with SuppressOutput():
                model = Prophet(
                    seasonality_mode="multiplicative",
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    mcmc_samples=300,
                )
                model.fit(df_city)

            # 予測
            future = model.make_future_dataframe(periods=100, freq='M')
            forecast = model.predict(future)

            # 横一列のプロット作成
            fig_row, axes_row = plt.subplots(1, 3, figsize=(20, 5))  # 1行3列

            # 左列: 予測プロット
            model.plot(forecast, ax=axes_row[0])
            axes_row[0].set_title(f'{city_name}の予測結果')

            # 中央列: コンポーネントプロットを画像として描画
            fig_components = model.plot_components(forecast)
            buf = BytesIO()  # メモリ内に保存
            fig_components.savefig(buf, format='png')
            buf.seek(0)
            img = plt.imread(buf)
            axes_row[1].imshow(img)
            axes_row[1].axis('off')  # 軸を非表示にする

            # 中央プロットは閉じる
            plt.close(fig_components)

            # 右列: 各年ごとの月別データプロット
            df_city["year"] = pd.to_datetime(df_city["ds"]).dt.year
            df_city["month"] = pd.to_datetime(df_city["ds"]).dt.month
            pivot_df = df_city.pivot(index="year", columns="month", values="y")

            for year in pivot_df.index:
                axes_row[2].plot(pivot_df.columns, pivot_df.loc[year], label=f"{year}年", marker='o')

            axes_row[2].set_title(f"{city_name}の各年ごとの値段（月別）")
            axes_row[2].set_xlabel("月")
            axes_row[2].set_ylabel("値段")
            axes_row[2].set_xticks(range(1, 13))
            axes_row[2].grid(True)

            legend = axes_row[2].legend(title="年", fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1))

            # 横一列のプロットを保存
            row_file = os.path.join(output_dir, f"{city_name}_row_plot.png")
            plt.tight_layout()
            plt.savefig(row_file, dpi=300)
            print(f"{city_name}の横一列のプロットを保存しました: {row_file}")

            # Notebook上に表示
            plt.show(fig_row)
            plt.close(fig_row)

        except Exception as e:
            # エラーをキャッチし、市名とエラー内容を表示
            print(f"市: {city_name} でエラーが発生しました: {e}")



####################
#	4/4
# date_reading: 
# thought: 
# words: 
# reference: 

####################

data_path = "../datasets/FEH_00200571_241119204010.csv"

city_names = [
    '札幌市', '青森市', '盛岡市', '仙台市', '秋田市', '山形市', '福島市', '水戸市', '宇都宮市', '前橋市', 
    '千葉市', '特別区部', '横浜市', '富山市', '金沢市', '福井市', '甲府市', '長野市', '岐阜市', '名古屋市', 
    '津市', '大津市', '京都市', '大阪市', '神戸市', '奈良市', '和歌山市', '鳥取市', '松江市', '広島市', 
    '山口市', '徳島市', '高松市', '松山市', '高知市', '福岡市', '佐賀市', '長崎市', '大分市', '宮崎市',
    '鹿児島市', '那覇市'
]
output_dir = "./output"
forecast_and_save_all_plots(data_path, city_names, output_dir)