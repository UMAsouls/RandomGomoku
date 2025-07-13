import torch
from alphazero import AlphaZero
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    print(f"使用デバイス: {device}")
    try:
        # モデルの初期化やトレーニングコードをここに追加
        print("モデルの初期化とトレーニングを開始します...")
        alapha_zero = AlphaZero()
        alapha_zero.train()
        print("トレーニングが完了しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print("トレーニング中にエラーが発生しました。")

    except KeyboardInterrupt:
        print("トレーニングが中断されました。")
    print("テストが完了しました。")
        
    
    

    
    
if __name__ == "__main__":
    main()