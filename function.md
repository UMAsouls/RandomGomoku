

## Board クラス

### `__init__(self, headmass: IHeadMass) -> None`
**説明**  
`Board` クラスのコンストラクタ。依存性注入により `IHeadMass` オブジェクトを受け取ります。

**引数**  
- `headmass: IHeadMass`  
  ボード生成やマスの管理に必要なヘッドマスのインターフェース。

---

### `SetStone(self, x: int, y: int, stone: Stone) -> bool`
**説明**  
指定した座標 `(x, y)` に石を置きます。

**引数**  
- `x: int`  
  石を置く x 座標。
- `y: int`  
  石を置く y 座標。
- `stone: Stone`  
  石の種類（黒または白）。

**戻り値**  
- `bool`  
  石を正常に置けた場合は `True`、失敗した場合は `False`。

---

### `RandomSet(self)`
**説明**  
初期配置としてランダムに石を配置します。白石を 4 つのランダムエリアに、黒石を 2 つのランダムラインに配置します。

**引数**  
なし。

**戻り値**  
なし。

---

### `MakeBoard(self, w: int, h: int)`
**説明**  
指定された幅 `w` と高さ `h` のボードを作成します。

**引数**  
- `w: int`  
  ボードの幅。
- `h: int`  
  ボードの高さ。

**戻り値**  
なし。

---

### `GetBoardInt(self) -> list[list[int]]`
**説明**  
現在のボードの状態を 2 次元配列として返します。各要素は以下の値を持ちます：
- `0`: 空のマス
- `1`: 黒石
- `2`: 白石

**引数**  
なし。

**戻り値**  
- `list[list[int]]`  
  ボードの状態を表す 2 次元配列。

---

### `PrintBoard(self) -> None`
**説明**  
現在のボードをコンソールに表示します。石の状態は以下の絵文字で表現されます：
- `🔳`: 空のマス
- `🔴`: 黒石
- `🔵`: 白石

**引数**  
なし。

**戻り値**  
なし。

---

## 関数 (ゲームロジック)

### `Agent_Random(board: Board, stone: Stone) -> bool`
**説明**  
ランダムな位置に指定された石を置くエージェントの実装です。

**引数**  
- `board: Board`  
  ゲームのボードオブジェクト。
- `stone: Stone`  
  置く石の種類（黒または白）。

**戻り値**  
- `bool`  
  石を正常に置けた場合は `True`、失敗した場合は `False`。

---

### `Main(agent_black: Callable[[Board, Stone], bool], agent_white: Callable[[Board, Stone], bool]) -> None`
**説明**  
2 つのエージェントを対戦させるメインループを実行します。

**引数**  
- `agent_black: Callable[[Board, Stone], bool]`  
  黒石を置くエージェント。
- `agent_white: Callable[[Board, Stone], bool]`  
  白石を置くエージェント。

**戻り値**  
なし。

---

## スクリプト用関数

### `Main() -> None`
**説明**  
2 人のプレイヤーで対戦するテキストベースの Gomoku (五目並べ) ゲームを実行します。

**引数**  
なし。

**戻り値**  
なし。

