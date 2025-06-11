import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ユーザー提供のGomokuEnvクラスがgomoku_env.pyファイルにあると仮定
from GomokuEnv import GomokuEnv, Stone

# --- ヘルパー関数 ---

def get_gomoku_env_board_and_player(env):
    """
    GomokuEnvから生の盤面と現在のプレイヤーを抽出します。
    盤面: HxW numpy配列 (0=空, 1=黒, 2=白)
    プレイヤー: 1 (黒), 2 (白)
    """
    raw_board = env.board.GetBoardInt()
    current_player_id = env.current_player # 1 (黒), 2 (白)
    return raw_board, current_player_id

def board_to_mixnet_input(raw_board, current_player_id, device='cpu'):
    """
    GomokuEnvからの生の盤面状態を、MixNetが期待する $2 \times H \times W$ のテンソルに変換します。
    チャネル0: 現在のプレイヤーの石
    チャネル1: 相手プレイヤーの石
    """
    # テンソルと NumPy 配列の両方に対応
    if isinstance(raw_board, torch.Tensor):
        H, W = raw_board.shape
        board_tensor = raw_board.long()
    else:
        H, W = raw_board.shape
        board_tensor = torch.from_numpy(raw_board).long()

    player_channel = torch.zeros((H, W), dtype=torch.float32)
    opponent_channel = torch.zeros((H, W), dtype=torch.float32)

    if current_player_id == 1: # 黒が現在のプレイヤー
        player_channel[board_tensor == 1] = 1.0
        opponent_channel[board_tensor == 2] = 1.0
    elif current_player_id == 2: # 白が現在のプレイヤー
        player_channel[board_tensor == 2] = 1.0
        opponent_channel[board_tensor == 1] = 1.0
    else:
        raise ValueError(f"無効なplayer_id: {current_player_id}")

    # スタックして 2 x H x W を形成
    mixnet_input = torch.stack([player_channel, opponent_channel], dim=0).to(device)
    return mixnet_input, H, W  # H, Wも返すように変更


def extract_line_patterns(board_state_S, R, C, line_length=11):
    """
    board_state_S (HxW, 0=空, 1=プレイヤー1/黒, 2=プレイヤー2/白) から、
    (R, C) を中心とする `line_length` の4方向のラインパターンを抽出します。
    それぞれがパターンを表す4つのリストのリストを返します。
    範囲外の場合は特別な値（例: -1）でパディングします。
    """
    H, W = board_state_S.shape
    patterns = []
    # プレイヤー1/黒は1、プレイヤー2/白は2、空は0。後で一貫したインデックス付けのために、
    # この生の形式で問題ありません。MyStone/OpponentStoneへの変換はコードブックのインデックス付けの前に行われます。

    # kが-5から5までのオフセット（長さ11の場合）
    k_offsets = list(range(-(line_length // 2), (line_length // 2) + 1))

    # 水平、垂直、主対角線、反対角線の(dx, dy)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        line = []
        for k in k_offsets:
            r, c = R + dr * k, C + dc * k
            if 0 <= r < H and 0 <= c < W:
                line.append(board_state_S[r, c])
            else:
                line.append(-1) # 範囲外のパディング値
        patterns.append(line)
    return patterns


def pattern_to_index(pattern_list, current_player_id):
    """
    単一のラインパターン（石の状態0,1,2とパディング用の-1のリスト）を、
    コードブック検索のための一意の整数インデックスに変換します。
    石は次のようにマッピングされます: 0 (パディング), 1 (空), 2 (現在のプレイヤー), 3 (相手)。
    インデックスには4進数エンコーディングを使用します。
    """
    idx = 0
    base = 4 # 0:pad, 1:empty, 2:my_stone, 3:opp_stone

    # 物理的な石（0,1,2）を相対的な石にマッピング
    # Pad = -1 -> 0
    # Empty = 0 -> 1
    # MyStone -> 2
    # OpponentStone -> 3
    
    opponent_player_id = 3 - current_player_id

    for i, stone_code in enumerate(pattern_list):
        value = 0
        if stone_code == -1: # パッド
            value = 0
        elif stone_code == 0: # 空
            value = 1
        elif stone_code == current_player_id: # 自分の石
            value = 2
        elif stone_code == opponent_player_id: # 相手の石
            value = 3
        else:
            raise ValueError(f"パターン内の未知のstone_code {stone_code}。")
        
        idx += value * (base ** i)
    return idx


# --- MixNet モジュール ---

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class MappingNetworkBlock(nn.Module):
    """ マッピングネットワークの簡略化されたブロック（図3左のResBlock / OutBlock）"""
    def __init__(self, M_channels):
        super().__init__()
        # 「3x3 Dir Conv」はラインパターンに沿った1D畳み込みとして解釈されます
        # このブロックへの入力は (Batch, M_channels, PatternLength) と仮定
        self.dir_conv = nn.Conv1d(M_channels, M_channels, kernel_size=3, padding=1, bias=False)
        self.silu1 = SiLU()
        self.point_conv = nn.Conv1d(M_channels, M_channels, kernel_size=1, bias=False) # 1x1 conv
        self.silu2 = SiLU()

    def forward(self, x_in): # x_in は残差接続への入力
        x = self.dir_conv(x_in)
        x = self.silu1(x)
        x = self.point_conv(x)
        x = self.silu2(x)
        return x + x_in # スキップ接続

class MappingNetwork(nn.Module):
    """
    ラインパターンを処理してコードブック用の特徴ベクトルを生成します。
    入力: ラインパターンのバッチ、(Batch, NumChannels=2 (one-hot player/opponent), PatternLength=11)
    出力: 各パターンの (Batch, C_channels) 特徴ベクトル。
    これは、(訓練後の)出力がコードブックを構成するネットワークです。
    最小MixNet: M=64 (内部), C=32 (コードブック出力用)
    5つの実効的な「DirConv」レイヤーを使用します。
    ここでの「DirConv」レイヤーは、Conv1d + PointwiseConv + Activationで構成されます。
    最初の「3x3 Dir Conv」は入力チャネル（one-hot石タイプ用に2）をMにマッピングします。
    """
    def __init__(self, M_channels=64, C_out_channels=32, pattern_length=11, input_pattern_channels=2):
        super().__init__()
        self.M = M_channels
        self.C_out = C_out_channels
        self.pattern_length = pattern_length

        # Mチャネルへの初期畳み込み
        # 図3左の「3x3 Dir Conv」。ラインパターン用の1D畳み込みと仮定。
        self.initial_conv = nn.Conv1d(input_pattern_channels, M_channels, kernel_size=3, padding=1, bias=False)
        
        # 図3に従い: 4つのResBlock + 1つのOutBlock。それぞれがDirConv相当 + 1x1を含む。合計5つ。
        self.res_blocks = nn.Sequential(
            MappingNetworkBlock(M_channels), # 実効的にDirConvレイヤー1 (initialの後)
            MappingNetworkBlock(M_channels), # DirConvレイヤー2
            MappingNetworkBlock(M_channels), # DirConvレイヤー3
            MappingNetworkBlock(M_channels)  # DirConvレイヤー4
        )
        self.out_block = MappingNetworkBlock(M_channels) # DirConvレイヤー5 (構造はResBlockに類似)

        # 最終1x1 ConvでC_out_channelsへ
        self.final_point_conv = nn.Conv1d(M_channels, C_out_channels, kernel_size=1)
        
        # パターンの出力特徴は通常、処理されたパターン内の特定の位置の特徴、
        # または集約（例: 平均）です。
        # 「(1,i+1)での特徴出力を記録する」というのは、処理後に中央の特徴を取ることを示唆しています。
        # ここでは簡単のため、パターン長にわたって平均します。

    def forward(self, x_pattern_batch): # x_pattern_batch: (Batch, 2, PatternLength)
        x = self.initial_conv(x_pattern_batch) # (Batch, M, PatternLength)
        
        x = self.res_blocks(x) # (Batch, M, PatternLength)
        x = self.out_block(x)  # (Batch, M, PatternLength)
        
        x = self.final_point_conv(x) # (Batch, C_out, PatternLength)
        
        # pattern_lengthに沿って集約して (Batch, C_out) を得る
        # 論文では「(1,i+1)での特徴出力を記録する」とあり、
        # これは中央の特徴を取ることを意味します。長さ11のパターンの場合、インデックス5。
        # 例: features = x[:, :, self.pattern_length // 2]
        # または、平均プーリング:
        features = x.mean(dim=2) # (Batch, C_out)
        return features

class StarBlock(nn.Module):
    """ 図4bのスターブロックを実装 """
    def __init__(self, C_in, V_out, C_internal_factor=1): # C_internal_factor は図の「2C」用
        super().__init__()
        C_hidden = C_in * C_internal_factor # これは図の「Split C C」の「C」
                                         # または、図の表記が2*C_hiddenの場合「2C」
        self.V_out = V_out  # V_outをインスタンス変数として保存

        # 図では入力x_inが2つの線形レイヤーに分岐しているように見えます。
        # それぞれが2Cチャネルを生成します（Cは内部次元、ここではC_star_internalと呼びます）。
        # 図の「Linear 2C」をLinear(C_in -> 2*C_star_internal)と解釈します。
        # V_outはこのブロックの最終チャネル次元です。
        # 入力がC_inの場合、図の「Linear 2C」はLinear(C_in, 2 * V_out_intermediate)になります。
        # 簡単のため、最終レイヤーの前にV_out_intermediate = V_outとします。
        
        self.linear1_lhs = nn.Linear(C_in, 2 * V_out) # 出力 2V'
        self.linear2_rhs = nn.Linear(C_in, 2 * V_out) # 出力 2V'
        self.relu_rhs = nn.ReLU()

        # 乗算とsplit+sumの後、V_outチャネルになります。
        # その後、V_outへの最終線形レイヤー。
        self.final_linear = nn.Linear(V_out, V_out)
        self.final_relu = nn.ReLU()

    def forward(self, x): # x: (Batch, C_in)
        x1 = self.linear1_lhs(x)    # (Batch, 2*V_out)
        x2 = self.linear2_rhs(x)    # (Batch, 2*V_out)
        
        y1 = x1
        y2 = self.relu_rhs(x2)
        
        z = y1 * y2                 # 要素ごとの積 (Batch, 2*V_out)
        
        # Split C, C して合計 (または図に従って加算)
        # z_a = z[:, :V_out]
        # z_b = z[:, V_out:]
        # s = z_a + z_b               # (Batch, V_out)
        
        # 「ペアごとのドット積...チャネルを半減」の代替解釈:
        # zが2Kチャネルを持つ場合、(B, K, 2)にリシェイプし、最後の次元に沿って積を取る？
        # または、より単純に「チャネルを半減」するために線形レイヤーを使用する:
        # s = F.adaptive_avg_pool1d(z.unsqueeze(1), V_out).squeeze(1) # チャネル削減のハックな方法、またはLinear(2*V_out, V_out)
        # 直接的な線形レイヤーが、学習された変換に対して「ペアごとのドット積で半減」が実質的に意味するものでしょう。
        # 今のところ、可能であれば図のsplit & sum解釈に従います。
        # zが(Batch, 2*V_out)の場合、z_aは(Batch, V_out)、z_bは(Batch, V_out)
        s = z[:, :self.V_out] + z[:, self.V_out:] # (Batch, V_out)

        out = self.final_linear(s)  # (Batch, V_out)
        out = self.final_relu(out)  # (Batch, V_out)
        return out


class PolicyHead(nn.Module):
    """ 図4aのポリシーヘッドを実装 """
    def __init__(self, C_channels, H, W, P_channels=16): # smallの場合 C=32, P=16
        super().__init__()
        self.P_channels = P_channels
        self.H, self.W = H, W

        # グローバル特徴平均計算
        self.avg_pool_global = nn.AdaptiveAvgPool2d(1) # 出力 (Batch, C, 1, 1)

        # 動的畳み込みの重みとバイアスを生成するための線形レイヤー
        # 入力 C_channels、重み(16*P)とバイアス(P)の出力。論文では16Pと16。P_channels = 16。
        # 動的point-wise畳み込みのカーネル: P_in_dyn_conv (P) x P_out_dyn_conv (16)
        # 動的point-wise畳み込みのバイアス: P_out_dyn_conv (16)
        self.P_out_dyn_conv = 16 # 動的畳み込みの固定出力チャネル数
        
        self.fc1_dyn = nn.Linear(C_channels, C_channels * 2) # 中間サイズ
        self.relu_dyn = nn.ReLU()
        # P_channels -> P_out_dyn_conv の重み、および P_out_dyn_conv のバイアス
        self.fc2_dyn_weights = nn.Linear(C_channels * 2, self.P_channels * self.P_out_dyn_conv)
        self.fc2_dyn_bias = nn.Linear(C_channels * 2, self.P_out_dyn_conv)

        # 最終1x1畳み込み
        self.final_conv = nn.Conv2d(self.P_out_dyn_conv, 1, kernel_size=1) # 出力 (Batch, 1, H, W)

    def forward(self, F_prime): # F_prime: (Batch, C, H, W)
        batch_size = F_prime.size(0)
        # 実際の入力サイズを取得
        _, _, current_H, current_W = F_prime.shape
        
        # グローバル特徴平均
        global_feat_mean = self.avg_pool_global(F_prime) # (B, C, 1, 1)
        global_feat_mean_flat = global_feat_mean.view(batch_size, -1) # (B, C)

        # 動的畳み込みパラメータの生成
        dyn_params = self.relu_dyn(self.fc1_dyn(global_feat_mean_flat)) # (B, C*2)
        
        # 動的重み: (B, P_channels * P_out_dyn_conv) -> (B, P_out_dyn_conv, P_channels, 1, 1) (畳み込み用)
        dynamic_weights = self.fc2_dyn_weights(dyn_params).view(batch_size, self.P_out_dyn_conv, self.P_channels, 1, 1)
        # 動的バイアス: (B, P_out_dyn_conv)
        dynamic_bias = self.fc2_dyn_bias(dyn_params) # (B, P_out_dyn_conv)

        # F_primeの最初のPチャネルに動的pointwise畳み込みを適用
        # F_prime_P: (B, P_channels, H, W)
        F_prime_P = F_prime[:, :self.P_channels, :, :]
        
        # バッチアイテムごとの動的重みのための手動グループ畳込み
        policy_features_list = []
        for i in range(batch_size):
            # weights_i: (P_out_dyn_conv, P_channels, 1, 1)
            # bias_i: (P_out_dyn_conv)
            # F_prime_P_i: (1, P_channels, H, W)
            conv_out = F.conv2d(F_prime_P[i:i+1], dynamic_weights[i], bias=dynamic_bias[i], padding=0)
            policy_features_list.append(conv_out)
        policy_features_intermediate = torch.cat(policy_features_list, dim=0) # (B, P_out_dyn_conv, H, W)

        policy_features_intermediate = F.relu(policy_features_intermediate) # 動的畳み込み後のReLU (図4a)
        
        # 最終1x1畳み込み
        raw_policy = self.final_conv(policy_features_intermediate) # (B, 1, H, W)
        return raw_policy.view(batch_size, current_H * current_W) # 実際の入力サイズに基づいてフラット化


class ValueHead(nn.Module):
    """ 図4bのバリューヘッドを実装 """
    def __init__(self, C_channels, H, W, V_channels=32): # smallの場合 C=32, V=32
        super().__init__()
        self.C_channels = C_channels
        self.V_channels = V_channels
        self.H, self.W = H, W

        # グローバル特徴平均パス
        self.avg_pool_global = nn.AdaptiveAvgPool2d(1) # (B, C, 1, 1)

        # バリューグルーピングパス
        self.num_group_regions_side = 3
        self.num_group_regions_total = self.num_group_regions_side ** 2 # 9領域

        # チャンクからの9つの平均特徴それぞれに対するスターブロック
        self.star_blocks_local = nn.ModuleList(
            [StarBlock(C_channels, V_channels) for _ in range(self.num_group_regions_total)]
        )
        
        # 論文では「連結され、平均化されて2x2のグループを形成する」(Eq 5)とあります。
        # これにより4つのグループができ、それぞれV_channels。合計4*V_channels。

        # 最終的な価値予測のためのMLP
        # 入力: グローバル特徴平均(C)と4つのグループ特徴(4*V)の連結
        mlp_input_size = C_channels + (4 * V_channels)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_input_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_input_size // 2, mlp_input_size // 4),
            nn.ReLU(),
            nn.Linear(mlp_input_size // 4, 3) # Win, Loss, Draw
        )

    def forward(self, F_prime): # F_prime: (Batch, C, H, W)
        batch_size = F_prime.size(0)

        # 1. グローバル特徴平均
        global_feat_mean = self.avg_pool_global(F_prime).view(batch_size, -1) # (B, C)

        # 2. バリューグルーピングパス
        #    a. F'を3x3の領域（チャンク）に分割
        #       各チャンクは (B, C, H/3, W/3)
        chunk_h, chunk_w = self.H // self.num_group_regions_side, self.W // self.num_group_regions_side
        
        # 簡単のため、HとWがnum_group_regions_sideで割り切れることを確認
        if self.H % self.num_group_regions_side != 0 or self.W % self.num_group_regions_side != 0:
            # 割り切れない場合はパディングまたはアダプティブプーリングを適用
            # ここでは、チャンクのアダプティブプーリングを使用
             F_prime_padded = F.adaptive_avg_pool2d(F_prime, 
                                (self.num_group_regions_side * (self.H // self.num_group_regions_side), 
                                 self.num_group_regions_side * (self.W // self.num_group_regions_side)))
             chunk_h = F_prime_padded.size(2) // self.num_group_regions_side
             chunk_w = F_prime_padded.size(3) // self.num_group_regions_side
        else:
            F_prime_padded = F_prime

        local_group_G_ij = [] # 9つのV次元特徴を保持するリスト
        for i in range(self.num_group_regions_side):
            for j in range(self.num_group_regions_side):
                chunk = F_prime_padded[:, :, i*chunk_h:(i+1)*chunk_h, j*chunk_w:(j+1)*chunk_w]
                # b. 各チャンクを平均プーリング: (B, C, 1, 1) -> (B, C)
                chunk_mean_feat = F.adaptive_avg_pool2d(chunk, 1).view(batch_size, -1)
                # c. スターブロックを適用: (B, C) -> (B, V)
                star_block_module = self.star_blocks_local[i * self.num_group_regions_side + j]
                local_group_G_ij.append(star_block_module(chunk_mean_feat))
        
        # local_group_G_ij には (B, V) 形状のテンソルが9つ含まれる
        # d. Eq. 5を使用して2x2のグループ G'_ij を形成
        # G_00, G_01, G_02
        # G_10, G_11, G_12
        # G_20, G_21, G_22
        # G'_00 = (G00+G01+G10+G11)/4
        # G'_01 = (G01+G02+G11+G12)/4
        # G'_10 = (G10+G11+G20+G21)/4
        # G'_11 = (G11+G12+G21+G22)/4
        
        G_prime_ij_list = []
        for r_idx in range(self.num_group_regions_side -1): # 0, 1
            for c_idx in range(self.num_group_regions_side -1): # 0, 1
                g_sum = (local_group_G_ij[r_idx * self.num_group_regions_side + c_idx] +        # G_i,j
                         local_group_G_ij[r_idx * self.num_group_regions_side + (c_idx + 1)] +  # G_i,j+1
                         local_group_G_ij[(r_idx + 1) * self.num_group_regions_side + c_idx] +  # G_i+1,j
                         local_group_G_ij[(r_idx + 1) * self.num_group_regions_side + (c_idx + 1)]) / 4 # G_i+1,j+1
                G_prime_ij_list.append(g_sum)
        
        # G_prime_ij_list には (B, V) 形状のテンソルが4つ含まれる
        grouped_features_4V = torch.cat(G_prime_ij_list, dim=1) # (B, 4*V)

        # 3. グローバル特徴とグループ化された特徴を連結
        combined_features = torch.cat([global_feat_mean, grouped_features_4V], dim=1) # (B, C + 4V)

        # 4. 最終予測のためのMLP
        raw_value = self.mlp(combined_features) # (B, 3)
        return raw_value


class MixNet(nn.Module):
    """
    RapfiのMixNetモデル。
    """
    def __init__(self, H, W, # 盤面の次元
                 C_channels=32, M_channels_mapping=64, # MixNet-Small用
                 P_channels_policy=16, V_channels_value=32,
                 line_length=11, num_possible_patterns_per_type=None, device='cpu'):
        super().__init__()
        self.H, self.W = H, W
        self.C = C_channels
        self.line_length = line_length
        self.device = device

        # コードブック: 実際のシナリオでは、これは事前計算されます。
        # ここでは、プレースホルダーとしてnn.Embeddingを使用します。
        # 2種類のマッピング関数M_hvとM_diがあります。
        # そのため、潜在的に2つのコードブック、または区別するロジックを持つ1つのコードブックが存在します。
        # 簡単のため、すべてのパターンに対して1つのコードブックを仮定します。
        # 一意なパターンの数（長さ11の4進数エンコーディング）: 4^11 = 4,194,304
        # これは大きいです。論文のN=397488は慎重な導出が必要です。
        # このプレースホルダーには、妥当なnum_possible_patternsを指定します。
        if num_possible_patterns_per_type is None:
             # 石の状態: pad, empty, my, opp (4状態)
            num_possible_patterns_per_type = 4**line_length 
        
        # 水平/垂直パターン(M_hv)と対角線パターン(M_di)の特徴が必要です。
        # これらは別々の埋め込みであるか、事前計算されていない場合は個別のマッピングネットワークによって処理される可能性があります。
        # この構造では、コードブック検索自体がこの区別を処理するか、
        # M_hvとM_diが1つの大きなコードブックへの異なるインデックス方法であるか、
        # または、最も忠実に、M_hvとM_diが2つの異なる訓練済みマッピングネットワークからの特徴であると仮定します。
        # ここでは、M_hvとM_diによって入力されたかのように2つのコードブックをシミュレートします。
        self.codebook_hv = nn.Embedding(num_possible_patterns_per_type, C_channels).to(device)
        self.codebook_di = nn.Embedding(num_possible_patterns_per_type, C_channels).to(device)
        # 実際には: コードブック特徴はMixNet訓練中に学習されるのではなく、ロードされます。

        # Fの最初の半分のチャネルに対する深さ方向3x3畳み込み
        self.depthwise_conv_F = nn.Conv2d(C_channels // 2, C_channels // 2, 
                                          kernel_size=3, padding=1, groups=C_channels // 2, bias=False)
        
        self.policy_head = PolicyHead(C_channels, H, W, P_channels_policy)
        self.value_head = ValueHead(C_channels, H, W, V_channels_value)

        self.relu_agg = nn.ReLU()

    def get_one_hot_pattern_representation(self, pattern_list, current_player_id):
        """
        パターンリスト（0空、1黒、2白、-1パッド）を、
        (プレースホルダーの)マッピングネットワーク用の2xL one-hotテンソルに変換します。
        Ch0: 現在のプレイヤー、Ch1: 相手。
        """
        L = len(pattern_list)
        one_hot = torch.zeros((2, L), dtype=torch.float32, device=self.device)
        opponent_player_id = 3 - current_player_id
        for i, stone_code in enumerate(pattern_list):
            if stone_code == current_player_id:
                one_hot[0, i] = 1.0
            elif stone_code == opponent_player_id:
                one_hot[1, i] = 1.0
            # 空とパッドは暗黙的にゼロ
        return one_hot

    def forward(self, board_S_numpy, current_player_id):
        """
        引数:
            board_S_numpy (np.array): HxW盤面 (0=空, 1=黒, 2=白)
            current_player_id (int): 1 (黒), 2 (白)
        """
        batch_size = 1 # 今のところ単一盤面評価を仮定。バッチ処理可能。
        
        # 集約された特徴マップF (B, C, H, W) を作成
        # これには、パターンの抽出、コードブックのインデックス付け、集約が含まれます。
        # 今のところ、これは事前計算されたコードブックなしでシミュレートするのが最も複雑な部分です。
        
        # Fのプレースホルダー - 実際のシナリオでは、パターンの抽出とコードブック検索が含まれます。
        # 盤面の各(r, c)に対して:
        #   patterns_at_rc = extract_line_patterns(board_S_numpy, r, c, self.line_length)
        #   idx_horiz = pattern_to_index(patterns_at_rc[0], current_player_id)
        #   idx_vert  = pattern_to_index(patterns_at_rc[1], current_player_id)
        #   idx_maind = pattern_to_index(patterns_at_rc[2], current_player_id)
        #   idx_antid = pattern_to_index(patterns_at_rc[3], current_player_id)
        #
        #   feat_horiz = self.codebook_hv(torch.tensor(idx_horiz, device=self.device))
        #   feat_vert  = self.codebook_hv(torch.tensor(idx_vert, device=self.device))
        #   feat_maind = self.codebook_di(torch.tensor(idx_maind, device=self.device))
        #   feat_antid = self.codebook_di(torch.tensor(idx_antid, device=self.device))
        #
        #   F_aggregated_rc = self.relu_agg(feat_horiz + feat_vert + feat_maind + feat_antid)
        #   F_map[batch_idx, :, r, c] = F_aggregated_rc
        
        # デモンストレーションのため、完全な検索パスを実装しない場合はFにランダムテンソルを使用
        # または、$2 \times H \times W$ 入力をいくつかの初期畳み込みに通したものをFの代わりとして使用
        initial_mixnet_input, current_H, current_W = board_to_mixnet_input(board_S_numpy, current_player_id, self.device)
        initial_mixnet_input = initial_mixnet_input.unsqueeze(0) # (1, 2, H, W)
        
        # --- パターン抽出とコードブックの 大幅な簡略化 ---
        # 完全なパターンロジックの代わりに、単純な畳み込みを使用してCチャネルを取得し、
        # 真の「F」特徴マップのプレースホルダーとして機能させます。
        if not hasattr(self, 'placeholder_F_conv'):
            self.placeholder_F_conv = nn.Conv2d(2, self.C, kernel_size=3, padding=1).to(self.device)
            # ここで学習する場合（Rapfiの動作方法ではない）、コードブック埋め込みを小さなランダム値で初期化
            nn.init.xavier_uniform_(self.codebook_hv.weight.data, gain=0.1)
            nn.init.xavier_uniform_(self.codebook_di.weight.data, gain=0.1)

        F_map_placeholder = self.placeholder_F_conv(initial_mixnet_input) # (B, C, H, W)
        # --- F_mapの簡略化終了 ---
        
        # FからF_primeを作成 (最初のC/2チャネルに深さ方向畳み込みを適用)
        F_first_half = F_map_placeholder[:, :self.C//2, :, :]
        F_second_half = F_map_placeholder[:, self.C//2:, :, :]
        
        F_prime_first_half = self.depthwise_conv_F(F_first_half)
        F_prime = torch.cat([F_prime_first_half, F_second_half], dim=1) # (B, C, H, W)

        # ポリシーヘッドとバリューヘッド
        raw_policy = self.policy_head(F_prime) # (B, H*W)
        raw_value = self.value_head(F_prime)   # (B, 3)
        
        # 訓練/推論に必要なsoftmax/log_softmaxを適用
        policy_dist = F.softmax(raw_policy, dim=1)
        value_dist = F.softmax(raw_value, dim=1) # 勝率、敗率、引き分け率

        return policy_dist, value_dist


# --- 使用例（概念的）---
if __name__ == '__main__':
    # これは概念的な例です。
    # ユーザーのファイルから適切なGomokuEnvインスタンスが必要になります。
    
    # --- デモンストレーション用のモックGomokuEnv（ユーザーのenvが直接実行可能でない場合）---
    class MockGomokuBoard:
        def __init__(self, H, W):
            self.board = np.zeros((H,W), dtype=int)
            self.H = H
            self.W = W
        def GetBoardInt(self):
            return self.board
        def MakeBoard(self, H, W): # GomokuEnvの構造に合わせるために追加
            self.board = np.zeros((H,W), dtype=int)
        def SetStone(self, r, c, stone_val): # 基本的なインタラクションのために追加
            if self.board[r,c] == 0:
                self.board[r,c] = stone_val
                return False # ゲーム終了せず
            return True # すでに占有されているかゲーム終了

    class MockGomokuEnv:
        def __init__(self, board_size=15, device='cpu'):
            self.board_size = board_size
            self.board = MockGomokuBoard(board_size, board_size)
            self.current_player = 1 # 黒が先手
            self.device = device
        
        def reset(self):
            self.board.MakeBoard(self.board_size, self.board_size)
            self.current_player = 1
            return self.board.GetBoardInt(), self.current_player

        def step(self, action_tuple_rc): # actionは(r,c)
            r, c = action_tuple_rc
            # 基本的な石の配置、モックでは勝利判定なし
            if self.board.GetBoardInt()[r,c] == 0 :
                self.board.SetStone(r,c,self.current_player)
                self.current_player = 3 - self.current_player # プレイヤー交代
                return self.board.GetBoardInt(), 0.0, False, {"which_player": 3 - self.current_player} # 報酬、終了フラグ、情報
            else:
                return self.board.GetBoardInt(), -1.0, True, {"which_player": self.current_player, "invalid_action":True} # 無効な手に対するペナルティ

    # --- モックGomokuEnv終了 ---

    print("概念的なMixNet使用例:")
    BOARD_HEIGHT = 15
    BOARD_WIDTH = 15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ユーザー提供の実際のGomokuEnvを使用する場合:
    # from gomoku_env import GomokuEnv 
    # env = GomokuEnv(board_size=BOARD_HEIGHT, device=DEVICE)
    # current_board_numpy, current_player = env.board.GetBoardInt(), env.current_player
    
    # この例では自己完結させるためにMockGomokuEnvを使用:
    env = MockGomokuEnv(board_size=BOARD_HEIGHT, device=DEVICE)
    current_board_numpy, current_player = env.reset()
    
    # 空でない盤面にするためにいくつかの石を配置
    env.step((7,7)) # 黒
    env.step((7,8)) # 白
    env.step((8,7)) # 黒
    current_board_numpy, current_player = env.board.GetBoardInt(), env.current_player


    print(f"盤面の次元: {BOARD_HEIGHT}x{BOARD_WIDTH}")
    print(f"現在のプレイヤーID: {current_player} ({'黒' if current_player == 1 else '白'})")
    print("盤面状態 (numpy):\n", current_board_numpy)

    # MixNetモデルの初期化
    # num_possible_patterns_per_typeには、デモのメモリ上の理由から小さなプレースホルダーを使用
    # 真の値は4**11。
    # この例では埋め込みを学習しますが、これはRapfiの動作方法ではありません（事前計算されたものを使用します）。
    mix_net_model = MixNet(H=BOARD_HEIGHT, W=BOARD_WIDTH, device=DEVICE,
                           num_possible_patterns_per_type=1024 # デモ用の小さなプレースホルダー
                           ).to(DEVICE)
    mix_net_model.eval() # 評価モードに設定

    with torch.no_grad():
        policy_output, value_output = mix_net_model(current_board_numpy, current_player)

    print("\nMixNet出力:")
    print("ポリシー (形状):", policy_output.shape) # (1, H*W) になるはず
    print("バリュー (形状):", value_output.shape)   # (1, 3) になるはず
    
    print("ポリシー (サンプル、最初の10アクション):", policy_output[0, :10].cpu().numpy())
    print("バリュー (勝率、敗率、引き分け率):", value_output[0].cpu().numpy())

    # マッピングネットワークの使用方法の例（概念的、コードブックを生成するため）
    # これは別の訓練プロセスになります。
    print("\n概念的なマッピングネットワーク使用例:")
    # one-hotエンコードされたパターンのダミーバッチを作成（プレイヤー/相手用に2チャネル、長さ11）
    # (Batch, 2, PatternLength)
    dummy_patterns_one_hot = torch.rand(4, 2, 11).to(DEVICE) # 4パターンのバッチ
    mapping_net = MappingNetwork(M_channels=64, C_out_channels=32, pattern_length=11).to(DEVICE)
    # これらの特徴がコードブックを構成します（例: MixNetのself.codebook_hv, self.codebook_di）
    codebook_features = mapping_net(dummy_patterns_one_hot)
    print("マッピングネットワーク出力形状 (Batch, C_out_channels):", codebook_features.shape)


    # --- パターン抽出とインデックス付けのデモンストレーション（Rapfiの中核部分）---
    print("\nパターン抽出とインデックス付けのデモンストレーション:")
    test_board = np.array([ # 簡単なテスト用の小さな5x5盤面
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], # (1,1)に黒
        [0, 0, 2, 0, 0], # (2,2)に白
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)
    test_player = 1 # 現在のプレイヤーは黒

    # test_boardの(1,1)におけるパターン、簡単のためline_length 5
    center_r, center_c = 1, 1
    line_len_test = 5 
    print(f"プレイヤー {test_player} (黒) のために ({center_r},{center_c}) で長さ {line_len_test} のパターンを抽出")
    raw_patterns = extract_line_patterns(test_board, center_r, center_c, line_length=line_len_test)
    
    # 方向: 水平、垂直、主対角線、反対角線
    dir_names = ["水平      ", "垂直      ", "主対角線  ", "反対角線"]
    for i, p_list in enumerate(raw_patterns):
        p_idx = pattern_to_index(p_list, test_player)
        print(f"  {dir_names[i]}: {p_list} -> インデックス: {p_idx}")

        # 概念的なコードブック検索
        # codebook_entry_hv = mix_net_model.codebook_hv(torch.tensor(p_idx, device=DEVICE))
        # codebook_entry_di = mix_net_model.codebook_di(torch.tensor(p_idx, device=DEVICE))
        # print(f"    コードブック_hv検索形状: {codebook_entry_hv.shape}")