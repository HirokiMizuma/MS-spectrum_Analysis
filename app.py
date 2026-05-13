import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io

# ==========================================
# 1. ページ設定とタイトル
# ==========================================
st.set_page_config(page_title="MS Polymer Fitting Tool", layout="wide")
st.title("MSスペクトル フィッティング解析")
st.markdown("""
データの読み込み、ベースライン（オフセット）の設定、およびポリマー質量の微調整を行い、
18成分のガウス分布を用いた自動フィッティングを実行します。
""")

# ==========================================
# 2. サイドバー設定（基本パラメータ）
# ==========================================
with st.sidebar:
    st.header("パラメータ設定")
    psa_mass = st.number_input("PSA 分子量", value=66800.0)
    smp_mass = st.number_input("SMP 1単位分子量（架橋部分）", value=152.12748, format="%.5f")
    smp_count = st.number_input("SMP 結合数 (n)", value=10)
    
    # 理論的なベース質量 (mu0)
    mu_0 = psa_mass + (smp_mass * smp_count)
    st.info(f"Polymer0-PSAの分子量: {mu_0:.2f}")
    
    st.divider()
    threshold_pct = st.slider("オフセット (%)", 0.0, 5.0, 1.0, 0.1)
    resolution_r = 22.0
    sigma_factor = 51.81 # R=22 -> FWHM=m/22 -> sigma = m/(22*2.355) = m/51.81

# ==========================================
# 3. データの読み込み（キャッシュ機能付き）
# ==========================================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # テキストファイルを読み込み
            df = pd.read_csv(uploaded_file, sep=r'\s+', comment='#', header=None, names=['mz', 'intensity'], on_bad_lines='skip')
            df['mz'] = pd.to_numeric(df['mz'], errors='coerce')
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
            df = df.dropna()
            # 解析範囲に絞る
            df = df[(df['mz'] >= 60000) & (df['mz'] <= 150000)]
            return df
        except Exception as e:
            st.error(f"データの読み込みに失敗しました: {e}")
            return None
    return None

uploaded_file = st.file_uploader("MSスペクトルデータ (TXTファイル) をアップロードしてください", type=['txt'])
df = load_data(uploaded_file)

if df is not None:
    # 全データ
    mz_fit = df['mz'].values
    int_fit = df['intensity'].values
    max_int = np.max(int_fit)
    
    # 描画用（軽量化のため5点に1点間引き）
    df_view = df.iloc[::5]
    mz_view = df_view['mz'].values
    int_view = df_view['intensity'].values

    # ==========================================
    # 4. インタラクティブ設定（オフセット & 質量初期値）
    # ==========================================
    st.divider()
    st.subheader("フィッティング初期値の調整")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("オフセットを設定し、ベースラインを作成してください。また、生データの各ピークにラインが合うようにポリマーの分子量を調整してください。")
        offset_pct = st.slider("オフセット (%)", 0.0, 15.0, 1.0, 0.1)
        init_poly_mass = st.slider("ポリマーの分子量 (Da) ", 4500.0, 6500.0, 5300.0, 1.0)
        
        st.warning("調整後に下の『フィッティング解析』を押してください。")
        run_fit = st.button("フィッティング解析", use_container_width=True, type="primary")

    with col2:
        # 初期調整用グラフ
        fig_ui, ax_ui = plt.subplots(figsize=(10, 5))
        ax_ui.plot(mz_view, int_view, color='black', alpha=0.6, linewidth=0.8, label="Line")
        # オフセットライン
        offset_val = max_int * (offset_pct / 100)
        ax_ui.axhline(offset_val, color='red', linestyle='--', alpha=0.7, label=f"Offset ({offset_pct}%)")
        # 質量ガイドライン
        for n in range(1, 15):
            ax_ui.axvline(mu_0 + n * init_poly_mass, color='blue', linestyle=':', alpha=0.4)
        
        ax_ui.set_xlim(mz_view.min(), mz_view.max())
        ax_ui.set_xlabel("m/z")
        ax_ui.set_ylabel("Intensity")
        ax_ui.legend(loc='upper right')
        st.pyplot(fig_ui)

    # ==========================================
    # 5. フィッティング実行
    # ==========================================
    if run_fit:
        with st.spinner("解析中..."):
            
            # 定数設定
            OFFSET_INT = max_int * (offset_pct / 100)
            COMPONENTS = 18

            # モデル関数
            def multi_gaussian(x, poly_mass, *amplitudes):
                y = np.full_like(x, OFFSET_INT)
                for n, A in enumerate(amplitudes, start=1):
                    mu_n = mu_0 + n * poly_mass
                    sigma_n = mu_n / sigma_factor
                    y += A * np.exp(-((x - mu_n)**2) / (2 * sigma_n**2))
                return y

            # 初期値と範囲
            p0 = [init_poly_mass] + [max(0, int_fit[np.abs(mz_fit - (mu_0 + n * init_poly_mass)).argmin()] - OFFSET_INT) for n in range(1, COMPONENTS + 1)]
            lb = [init_poly_mass - 100.0] + [0.0] * COMPONENTS
            ub = [init_poly_mass + 100.0] + [np.inf] * COMPONENTS

            # 最適化
            popt, _ = curve_fit(multi_gaussian, mz_fit, int_fit, p0=p0, bounds=(lb, ub))
            opt_poly_mass = popt[0]
            opt_amps = popt[1:]

            # 面積と平均結合数算出
            areas = []
            for n, A in enumerate(opt_amps, start=1):
                mu_n = mu_0 + n * opt_poly_mass
                sigma_n = mu_n / sigma_factor
                area = A * sigma_n * np.sqrt(2 * np.pi)
                areas.append(area)

            max_area = max(areas)
            ratios = [(a / max_area) * 100 for a in areas]
            
            valid_area_sum = 0
            weighted_n_sum = 0
            final_amps = []
            
            for n, ratio in enumerate(ratios, start=1):
                if ratio >= threshold_pct:
                    valid_area_sum += areas[n-1]
                    weighted_n_sum += n * areas[n-1]
                    final_amps.append(opt_amps[n-1])
                else:
                    final_amps.append(0.0)
            
            mean_n = weighted_n_sum / valid_area_sum if valid_area_sum > 0 else 0
            
            # 誤差算出（裏側でのみ保持）
            sim_total = multi_gaussian(mz_fit, opt_poly_mass, *final_amps)
            residuals = int_fit - sim_total
            ssr = np.sum(residuals**2)
            rmse = np.sqrt(np.mean(residuals**2))

            # ==========================================
            # 6. 結果表示（レイアウト分割）
            # ==========================================
            st.success("解析が完了しました！")
            
            # 左側にグラフ、右側に結果と表を配置
            res_col_left, res_col_right = st.columns([2, 1])

            with res_col_left:
                st.markdown("**フィッティング波形**")
                # グラフ描画
                fig_res, ax_res = plt.subplots(figsize=(10, 6))
                ax_res.plot(mz_fit, int_fit, color='black', alpha=0.4, label="Raw Data")
                ax_res.plot(mz_fit, sim_total, color='red', linestyle='--', label="Simulated Total")
                ax_res.axhline(OFFSET_INT, color='gray', linestyle=':', label="Baseline")
                
                colors = plt.cm.tab20(np.linspace(0, 1, COMPONENTS))
                for n, A in enumerate(final_amps, start=1):
                    if A > 0:
                        mu_n = mu_0 + n * opt_poly_mass
                        sigma_n = mu_n / sigma_factor
                        y_comp = A * np.exp(-((mz_fit - mu_n)**2) / (2 * sigma_n**2))
                        ax_res.plot(mz_fit, y_comp + OFFSET_INT, color=colors[n-1], alpha=0.7)
                        ax_res.text(mu_n, (A + OFFSET_INT)*1.02, f"P{n}", color=colors[n-1], ha='center', fontsize=8)

                ax_res.set_xlim(70000, 135000)
                ax_res.set_xlabel("m/z")
                ax_res.set_ylabel("Intensity")
                ax_res.legend()
                st.pyplot(fig_res)

            with res_col_right:
                st.markdown("**算出結果**")
                st.metric("平均結合数 (Avg n)", f"{mean_n:.3f}")
                st.metric("最適ポリマー質量", f"{opt_poly_mass:.2f} Da")
                
                st.markdown("**各成分の相対割合**")
                # 割合一覧表の作成
                ratio_df_display = pd.DataFrame({
                    "成分": [f"P{i}" for i in range(1, 19)],
                    "割合 (%)": [f"{r:.2f}" for r in ratios]
                })
                # Streamlitのdataframe機能で綺麗に表示
                st.dataframe(ratio_df_display, use_container_width=True, height=350)

            # ==========================================
            # 7. Excelダウンロード機能
            # ==========================================
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary Sheet (ここにSSRとRMSEはしっかり残します)
                sum_df = pd.DataFrame({
                    "項目": ["平均結合数", "ポリマー質量", "オフセット(%)", "SSR", "RMSE", "閾値(%)"],
                    "値": [mean_n, opt_poly_mass, offset_pct, ssr, rmse, threshold_pct]
                })
                sum_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Ratios Sheet
                ratio_df = pd.DataFrame({"結合数": [f"P{i}" for i in range(1, 19)], "相対割合(%)": ratios})
                ratio_df.to_excel(writer, sheet_name='Summary', startrow=8, index=False)
                
                # Data Sheet
                spectral_df = pd.DataFrame({'mz': mz_fit, 'Raw': int_fit, 'Simulated': sim_total})
                spectral_df.to_excel(writer, sheet_name='Spectral_Data', index=False)

            st.download_button(
                label="ダウンロード（Excelファイル）",
                data=output.getvalue(),
                file_name=f"MS_Analysis_{uploaded_file.name.split('.')[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("↑ Uploadから、解析したいスペクトルファイル（TXT）を選択してください。")

# ==========================================
# 💡 使い方ガイド
# ==========================================
with st.expander("ガイドライン"):
    st.write("""
    1. **ファイルのアップロード**: MALDI-TOF MSから出力されたTXTファイルをアップロードします。
    2. **パラメータ設定**: PSA、SMPの質量およびSMP結合数を入力すると、Polymerが結合していないPolymer0-PSAの理論的な質量が自動計算されます。
    3. **目視調整**: スライダーを動かして、青い点線が実際のピークの頂点に、赤い線がベースラインの底に重なるように調整します。
    4. **解析実行**: ボタンを押すと、±100Daの範囲で自動フィッティングを行い、平均結合数が算出されます。
    5. **ダウンロード**: 計算結果とスペクトルデータは、そのままExcelファイルとして保存可能です。
    """)
