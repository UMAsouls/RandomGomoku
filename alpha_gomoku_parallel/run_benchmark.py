"""
AlphaZero実装間の包括的ベンチマーク実行スクリプト
"""
import sys
import os
import multiprocessing
import time
from datetime import datetime

def run_comprehensive_benchmark():
    """包括的ベンチマークを実行"""
    print("=" * 80)
    print("AlphaZero実装間の包括的ベンチマーク")
    print("=" * 80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # マルチプロセス設定
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    try:
        # 包括的ベンチマークを実行
        print("1. 包括的ベンチマークを実行しています...")
        from comprehensive_benchmark import ComprehensiveBenchmark
        
        benchmark = ComprehensiveBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        if results:
            print("✓ ベンチマーク実行完了")
            
            # 比較レポートを生成
            print("\n2. 比較レポートを生成しています...")
            comparison_data = benchmark.generate_comparison_report()
            print("✓ 比較レポート生成完了")
            
            # 結果を保存
            print("\n3. 結果を保存しています...")
            benchmark.save_results()
            print("✓ 結果保存完了")
            
            # パフォーマンスグラフを作成
            print("\n4. パフォーマンスグラフを作成しています...")
            benchmark.create_performance_graph()
            print("✓ グラフ作成完了")
            
            # 詳細分析を実行
            print("\n5. 詳細分析を実行しています...")
            try:
                from benchmark_analyzer import BenchmarkAnalyzer
                analyzer = BenchmarkAnalyzer()
                analyzer.analyze_performance_trends()
                analyzer.analyze_resource_utilization()
                analyzer.create_detailed_performance_charts()
                analyzer.generate_recommendation_report()
                analyzer.export_to_excel()
                print("✓ 詳細分析完了")
            except Exception as e:
                print(f"⚠ 詳細分析中にエラーが発生しました: {e}")
            
        else:
            print("✗ ベンチマーク実行に失敗しました")
            return False
            
    except Exception as e:
        print(f"✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def print_usage():
    """使用方法を表示"""
    print("使用方法:")
    print("  python run_benchmark.py [オプション]")
    print()
    print("オプション:")
    print("  -h, --help     このヘルプを表示")
    print("  -q, --quick    クイックベンチマーク（少ないゲーム数）")
    print("  -f, --full     フルベンチマーク（多いゲーム数）")
    print("  -a, --analyze  既存結果の分析のみ実行")
    print()
    print("例:")
    print("  python run_benchmark.py          # 標準ベンチマーク")
    print("  python run_benchmark.py -q       # クイックベンチマーク")
    print("  python run_benchmark.py -f       # フルベンチマーク")
    print("  python run_benchmark.py -a       # 分析のみ")

def run_analysis_only():
    """既存結果の分析のみを実行"""
    print("=" * 80)
    print("既存ベンチマーク結果の分析")
    print("=" * 80)
    
    try:
        from benchmark_analyzer import BenchmarkAnalyzer
        analyzer = BenchmarkAnalyzer()
        
        if analyzer.results:
            print("既存の結果を分析しています...")
            analyzer.analyze_performance_trends()
            analyzer.analyze_resource_utilization()
            analyzer.create_detailed_performance_charts()
            analyzer.generate_recommendation_report()
            analyzer.export_to_excel()
            print("✓ 分析完了")
            return True
        else:
            print("✗ 分析する結果が見つかりません")
            print("先にベンチマークを実行してください: python run_benchmark.py")
            return False
            
    except Exception as e:
        print(f"✗ 分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
            return
        elif sys.argv[1] in ['-a', '--analyze']:
            success = run_analysis_only()
        elif sys.argv[1] in ['-q', '--quick']:
            print("クイックベンチマークモードは現在未実装です。標準ベンチマークを実行します。")
            success = run_comprehensive_benchmark()
        elif sys.argv[1] in ['-f', '--full']:
            print("フルベンチマークモードは現在未実装です。標準ベンチマークを実行します。")
            success = run_comprehensive_benchmark()
        else:
            print(f"不明なオプション: {sys.argv[1]}")
            print_usage()
            return
    else:
        success = run_comprehensive_benchmark()
    
    # 結果サマリーを表示
    print("\n" + "=" * 80)
    if success:
        print("ベンチマーク処理が正常に完了しました！")
        print("\n生成されたファイル:")
        files = [
            'benchmark_results.json',
            'alphazero_benchmark_comparison.png',
            'detailed_performance_analysis.png',
            'benchmark_analysis.xlsx'
        ]
        
        for file in files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ⚠ {file} (生成されませんでした)")
        
        print("\n結果の確認方法:")
        print("  1. benchmark_results.json: 詳細な数値データ")
        print("  2. alphazero_benchmark_comparison.png: 基本的な比較グラフ")
        print("  3. detailed_performance_analysis.png: 詳細分析グラフ")
        print("  4. benchmark_analysis.xlsx: Excel形式の詳細レポート")
        
    else:
        print("ベンチマーク処理中にエラーが発生しました。")
        print("エラーメッセージを確認し、必要に応じて修正してください。")
    
    print("=" * 80)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
