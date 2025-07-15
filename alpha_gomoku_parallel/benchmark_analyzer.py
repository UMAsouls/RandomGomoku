"""
ベンチマーク結果の詳細分析ユーティリティ
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# オプションのインポート
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class BenchmarkAnalyzer:
    def __init__(self, results_file='benchmark_results.json'):
        """
        ベンチマーク結果を読み込んで分析
        """
        self.results_file = results_file
        self.results = self.load_results()
    
    def load_results(self):
        """ベンチマーク結果を読み込み"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"ベンチマーク結果を {self.results_file} から読み込みました。")
            return results
        except FileNotFoundError:
            print(f"結果ファイル {self.results_file} が見つかりません。")
            return None
        except Exception as e:
            print(f"結果ファイルの読み込み中にエラーが発生しました: {e}")
            return None
    
    def analyze_performance_trends(self):
        """パフォーマンスの傾向を分析"""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("パフォーマンス傾向分析")
        print("="*60)
        
        # 有効な結果のみを抽出
        valid_results = {k: v for k, v in self.results.items() 
                        if v is not None and isinstance(v, dict) and 'times' in v}
        
        for impl_name, results in valid_results.items():
            times = results['times']
            print(f"\n{impl_name} の詳細:")
            print(f"  実行回数: {len(times)}")
            print(f"  平均時間: {np.mean(times):.2f}秒")
            print(f"  標準偏差: {np.std(times):.2f}秒")
            print(f"  最小時間: {np.min(times):.2f}秒")
            print(f"  最大時間: {np.max(times):.2f}秒")
            print(f"  変動係数: {np.std(times)/np.mean(times)*100:.1f}%")
            
            # 一貫性の評価
            consistency = "高" if np.std(times)/np.mean(times) < 0.1 else "中" if np.std(times)/np.mean(times) < 0.2 else "低"
            print(f"  一貫性: {consistency}")
    
    def analyze_resource_utilization(self):
        """リソース使用率の分析"""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("リソース使用率分析")
        print("="*60)
        
        system_info = self.results.get('system_info', {})
        print(f"システム構成:")
        print(f"  CPU: {system_info.get('cpu_count', 'N/A')} cores")
        print(f"  メモリ: {system_info.get('memory_total', 'N/A')} GB")
        print(f"  GPU: {system_info.get('gpu_name', 'N/A')}")
        
        # 各実装のリソース使用効率
        valid_results = {k: v for k, v in self.results.items() 
                        if v is not None and isinstance(v, dict) and 'avg_memory' in v}
        
        for impl_name, results in valid_results.items():
            print(f"\n{impl_name} のリソース使用率:")
            print(f"  平均メモリ使用量: {results['avg_memory']:.1f} MB")
            
            if 'avg_cpu' in results:
                print(f"  平均CPU使用率: {results['avg_cpu']:.1f}%")
                cpu_efficiency = results['avg_cpu'] / system_info.get('cpu_count', 1)
                print(f"  CPU効率: {cpu_efficiency:.1f}%/core")
            
            if 'parallel_efficiency' in results:
                print(f"  並列効率: {results['parallel_efficiency']:.2f}")
    
    def create_detailed_performance_charts(self):
        """詳細なパフォーマンスチャートを作成"""
        if not self.results:
            return
        
        valid_results = {k: v for k, v in self.results.items() 
                        if v is not None and isinstance(v, dict) and 'times' in v}
        
        if len(valid_results) < 2:
            print("詳細チャートを作成するには少なくとも2つの実装が必要です。")
            return
        
        # 詳細チャートを作成
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        implementations = list(valid_results.keys())
        colors = ['blue', 'green', 'red', 'orange', 'purple'][:len(implementations)]
        
        # 1. 実行時間の分布
        ax1 = axes[0, 0]
        for i, (impl, results) in enumerate(valid_results.items()):
            ax1.hist(results['times'], alpha=0.7, label=impl, color=colors[i], bins=10)
        ax1.set_title('実行時間の分布')
        ax1.set_xlabel('実行時間 (秒)')
        ax1.set_ylabel('頻度')
        ax1.legend()
        
        # 2. ボックスプロット
        ax2 = axes[0, 1]
        data_for_box = [valid_results[impl]['times'] for impl in implementations]
        ax2.boxplot(data_for_box, labels=implementations)
        ax2.set_title('実行時間のボックスプロット')
        ax2.set_ylabel('実行時間 (秒)')
        
        # 3. 平均時間の比較
        ax3 = axes[0, 2]
        avg_times = [valid_results[impl]['avg_time'] for impl in implementations]
        std_times = [valid_results[impl]['std_time'] for impl in implementations]
        ax3.bar(implementations, avg_times, yerr=std_times, color=colors, alpha=0.7)
        ax3.set_title('平均実行時間（エラーバー付き）')
        ax3.set_ylabel('実行時間 (秒)')
        
        # 4. メモリ使用量の比較
        ax4 = axes[1, 0]
        memory_usage = [valid_results[impl]['avg_memory'] for impl in implementations]
        ax4.bar(implementations, memory_usage, color=colors, alpha=0.7)
        ax4.set_title('平均メモリ使用量')
        ax4.set_ylabel('メモリ使用量 (MB)')
        
        # 5. スループット比較
        ax5 = axes[1, 1]
        throughput = [valid_results[impl]['games_per_second'] for impl in implementations]
        ax5.bar(implementations, throughput, color=colors, alpha=0.7)
        ax5.set_title('処理スループット')
        ax5.set_ylabel('ゲーム/秒')
        
        # 6. 効率性マトリックス
        ax6 = axes[1, 2]
        # 基準を設定
        baseline_time = max(avg_times)
        speedups = [baseline_time / time for time in avg_times]
        efficiency_scores = [speedup / memory for speedup, memory in zip(speedups, memory_usage)]
        
        scatter = ax6.scatter(speedups, efficiency_scores, s=100, c=colors[:len(implementations)], alpha=0.7)
        for i, impl in enumerate(implementations):
            ax6.annotate(impl, (speedups[i], efficiency_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax6.set_title('効率性マトリックス')
        ax6.set_xlabel('スピードアップ')
        ax6.set_ylabel('効率性スコア')
        
        plt.tight_layout()
        plt.savefig('detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("詳細パフォーマンスチャートを 'detailed_performance_analysis.png' に保存しました。")
    
    def generate_recommendation_report(self):
        """推奨事項レポートを生成"""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("推奨事項レポート")
        print("="*60)
        
        valid_results = {k: v for k, v in self.results.items() 
                        if v is not None and isinstance(v, dict) and 'avg_time' in v}
        
        if len(valid_results) < 2:
            print("推奨事項を生成するには少なくとも2つの実装が必要です。")
            return
        
        # 最高性能の実装を特定
        best_impl = min(valid_results.items(), key=lambda x: x[1]['avg_time'])
        best_name, best_results = best_impl
        
        # 各指標での評価
        print(f"【総合評価】")
        print(f"最高性能実装: {best_name}")
        print(f"  平均実行時間: {best_results['avg_time']:.2f}秒")
        print(f"  処理速度: {best_results['games_per_second']:.2f}ゲーム/秒")
        
        # シナリオ別推奨事項
        print(f"\n【シナリオ別推奨事項】")
        
        # メモリ制約のある環境
        memory_efficient = min(valid_results.items(), key=lambda x: x[1]['avg_memory'])
        print(f"メモリ制約のある環境: {memory_efficient[0]}")
        print(f"  メモリ使用量: {memory_efficient[1]['avg_memory']:.1f} MB")
        
        # 高速処理が必要な環境
        fastest = min(valid_results.items(), key=lambda x: x[1]['avg_time'])
        print(f"高速処理が必要な環境: {fastest[0]}")
        print(f"  実行時間: {fastest[1]['avg_time']:.2f}秒")
        
        # 一貫性が重要な環境
        most_consistent = min(valid_results.items(), 
                            key=lambda x: np.std(x[1]['times'])/np.mean(x[1]['times']))
        consistency_score = np.std(most_consistent[1]['times'])/np.mean(most_consistent[1]['times'])
        print(f"一貫性が重要な環境: {most_consistent[0]}")
        print(f"  変動係数: {consistency_score*100:.1f}%")
        
        # 具体的な推奨事項
        print(f"\n【具体的な推奨事項】")
        
        system_info = self.results.get('system_info', {})
        cpu_count = system_info.get('cpu_count', 1)
        
        if cpu_count >= 8:
            print("- 8コア以上のシステムでは並列処理版が推奨されます")
        elif cpu_count >= 4:
            print("- 4-7コアのシステムでは並列処理版が有効ですが、効果は限定的です")
        else:
            print("- 4コア未満のシステムでは元の実装が適している可能性があります")
        
        if system_info.get('gpu_available', False):
            print("- GPU利用可能環境では並列処理版でGPU加速が期待できます")
        
        memory_total = system_info.get('memory_total', 0)
        if memory_total >= 16:
            print("- 16GB以上のメモリ環境では並列処理の恩恵を最大限活用できます")
        elif memory_total >= 8:
            print("- 8-15GBのメモリ環境では並列度を調整することを推奨します")
        else:
            print("- 8GB未満のメモリ環境では逐次処理版を推奨します")
        
    def export_to_excel(self, filename='benchmark_analysis.xlsx'):
        """結果をExcelファイルにエクスポート"""
        if not self.results:
            return
        
        if not HAS_PANDAS:
            print("Excelエクスポートにはpandasが必要です。pip install pandasを実行してください。")
            return
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # システム情報
                system_df = pd.DataFrame([self.results.get('system_info', {})])
                system_df.to_excel(writer, sheet_name='System_Info', index=False)
                
                # 実装別結果
                valid_results = {k: v for k, v in self.results.items() 
                                if v is not None and isinstance(v, dict) and 'times' in v}
                
                for impl_name, results in valid_results.items():
                    # 基本統計
                    stats_data = {
                        'Metric': ['Average Time', 'Standard Deviation', 'Min Time', 'Max Time', 
                                  'Games per Second', 'Memory Usage', 'CPU Usage'],
                        'Value': [
                            results['avg_time'],
                            results['std_time'],
                            min(results['times']),
                            max(results['times']),
                            results['games_per_second'],
                            results['avg_memory'],
                            results.get('avg_cpu', 'N/A')
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name=f'{impl_name}_Stats', index=False)
                    
                    # 詳細実行時間
                    times_df = pd.DataFrame({'Execution_Time': results['times']})
                    times_df.to_excel(writer, sheet_name=f'{impl_name}_Times', index=False)
                
            print(f"詳細分析結果を {filename} にエクスポートしました。")
            
        except Exception as e:
            print(f"Excelエクスポート中にエラーが発生しました: {e}")
            print("openpyxlがインストールされていない可能性があります。pip install openpyxlを実行してください。")

def main():
    """メイン実行関数"""
    analyzer = BenchmarkAnalyzer()
    
    if analyzer.results:
        print("ベンチマーク結果の詳細分析を開始します...")
        
        # 各種分析を実行
        analyzer.analyze_performance_trends()
        analyzer.analyze_resource_utilization()
        analyzer.create_detailed_performance_charts()
        analyzer.generate_recommendation_report()
        
        # Excelにエクスポート
        analyzer.export_to_excel()
        
        print("\n詳細分析が完了しました。")
    else:
        print("ベンチマーク結果が見つかりません。先にcomprehensive_benchmark.pyを実行してください。")

if __name__ == "__main__":
    main()
