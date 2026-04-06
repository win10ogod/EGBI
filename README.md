# EGBI — Entropy-Gated Bit-Vector Indexing

高速高精度模糊匹配 C++ 函式庫。

EGBI 將**訊息熵 (IDF)** 與**位元向量 (Bit-Vector)** 結合，透過「動態位元分配」讓高鑑別力特徵佔據更多索引空間，低價值特徵自動被壓制。查詢時只需一次 `AND` + `OR` + `POPCNT`，即可在毫秒內完成萬級文件的加權 Jaccard 相似度估算。

---

## 特點

| 特性 | 說明 |
|------|------|
| **Header-only** | 只需 `#include <egbi/egbi.hpp>`，零外部依賴 |
| **跨平台** | Windows / Linux / macOS，C++17 |
| **硬體加速** | 自動使用 `__builtin_popcountll` / `__popcnt64`，搭配 `-mpopcnt` 編譯旗標 |
| **動態位元分配** | 高 IDF 特徵分配更多雜湊位元，低 IDF 噪音自動衰減 |
| **序列化** | 內建二進位 `save()` / `load()`，可持久化索引 |
| **token 模式** | 支援自訂分詞器，透過 `build_vector_from_tokens()` 接入 |

---

## 快速開始

### 1. 取得原始碼

```bash
git clone <repo-url> EGBI
cd EGBI
```

### 2. 編譯

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 3. 執行範例

```bash
./build/egbi_example        # Linux / macOS
build\Release\egbi_example  # Windows
```

### 4. 執行測試

```bash
cd build && ctest --output-on-failure
```

---

## EGBI-CLI 命令列工具

編譯後產生 `egbi-cli` 可執行檔，提供五個子命令：

### `build` — 建立索引

從文字檔（每行一筆文件）建立 EGBI 索引：

```bash
egbi-cli build corpus.txt my.egbi -k 2048 -a 2.5
```

### `search` — 模糊搜尋

對已建立的索引進行模糊查詢：

```bash
egbi-cli search my.egbi "quick brown fox" -t 0.15 --corpus corpus.txt
```

輸出範例：
```
Query: "quick brown fox"
Found 4 results in 6.6 us
----------------------------------------------------------------
  [     0]  score=0.3219  | The quick brown fox jumps over the lazy dog
  [     1]  score=0.2994  | The quick brown fox leaps over a sleepy dog
  [     7]  score=0.2456  | Quick brown foxes are faster than lazy dogs
  [    30]  score=0.2000  | The brown fox quickly jumped over the sleeping dog
```

### `eval` — 準確度評估

將 EGBI 近似值與精確 Jaccard 比較，產出完整評估報告：

```bash
egbi-cli eval corpus.txt --pairs 1000 -k 1024
```

報告包含：
- **MAE / RMSE**：近似誤差統計
- **Ranking Concordance**：排序一致性（三元組測試）
- **Precision / Recall / F1**：在指定閾值下的檢索品質
- **Top 5 Worst Approximations**：誤差最大的配對

### `bench` — 擴展性基準測試

使用合成資料測試不同規模下的效能：

```bash
egbi-cli bench --docs 50000 --queries 100 -k 1024
```

報告包含：
- **IDF / Index Build** 時間與吞吐量
- **Search Latency** (Avg / P50 / P95 / P99 / QPS)
- **Serialization** 讀寫速度
- **Bit-Length Scalability**：不同 k 值對建構和搜尋的影響
- **Document Count Scalability**：文件數量增長的效能曲線

### `info` — 索引資訊

```bash
egbi-cli info my.egbi
```

### 所有選項

| 選項 | 短名 | 說明 | 預設 |
|------|------|------|------|
| `--bits` | `-k` | 位元向量長度 | 1024 |
| `--alpha` | `-a` | 動態位元分配縮放 | 2.0 |
| `--ngram` | `-n` | N-gram 大小 | 3 |
| `--min-bits` | | 每特徵最小位元 | 1 |
| `--max-bits` | | 每特徵最大位元 | 32 |
| `--idf-floor` | | IDF 過濾門檻 | 0.0 |
| `--threshold` | `-t` | 搜尋最低分數 | 0.1 |
| `--max` | `-m` | 最大回傳結果 | 10 |
| `--corpus` | | 搜尋時顯示原文 | — |
| `--pairs` | | eval 測試配對數 | 500 |
| `--docs` | | bench 文件數量 | 10000 |
| `--queries` | | bench 查詢數量 | 50 |
| `--doc-len` | | bench 文件平均長度 | 200 |
| `--seed` | | 隨機種子 | 42 |

---

## 最小使用範例

```cpp
#include <egbi/egbi.hpp>
#include <iostream>

int main() {
    std::vector<std::string> docs = {
        "The quick brown fox jumps over the lazy dog",
        "Machine learning and deep neural networks",
        "The brown fox leaps across a tired hound",
    };

    egbi::Config cfg;
    cfg.bit_length = 1024;
    cfg.alpha      = 2.0f;
    cfg.ngram_size = 3;

    egbi::EGBIIndex index(cfg);
    index.add_documents(docs);  // 建立 IDF + 索引

    auto results = index.search("quick brown fox", 0.15f, 5);
    for (auto& r : results)
        std::cout << "DocId=" << r.doc_id
                  << " score=" << r.score << "\n";
}
```

---

## API 參考

### `egbi::Config`

| 欄位 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `bit_length` | `uint32_t` | 1024 | 位元向量長度 $k$。越大越精確，儲存越多 |
| `alpha` | `float` | 2.0 | 動態位元分配縮放常數 $\alpha$ |
| `ngram_size` | `uint32_t` | 3 | 字元級 n-gram 窗口大小 |
| `min_bits` | `uint32_t` | 1 | 每個特徵最少分配位元數 |
| `max_bits` | `uint32_t` | 32 | 每個特徵最多分配位元數（防止極稀有詞過度佔用） |
| `idf_floor` | `float` | 0.0 | IDF 低於此值的特徵直接丟棄 |

### `egbi::EGBIIndex`

#### 建構與資料

| 方法 | 說明 |
|------|------|
| `EGBIIndex(const Config& cfg = {})` | 建構索引 |
| `void add_documents(const vector<string>& docs)` | 批次建立 IDF 並索引所有文件（推薦） |
| `void build_idf(const vector<string>& docs)` | 僅建立 IDF 統計，不索引 |
| `DocId add_document(const string& text)` | 逐篇新增文件（需先呼叫 `build_idf`） |
| `DocId add_vector(BitVector&& v)` | 直接加入預先建好的位元向量 |
| `void clear()` | 清空索引和 IDF 統計 |

#### 搜尋

| 方法 | 說明 |
|------|------|
| `vector<SearchResult> search(query, threshold, max_results)` | 模糊搜尋，回傳分數 ≥ threshold 的結果（降序） |
| `vector<SearchResult> search_by_vector(vq, threshold, max_results)` | 使用預建查詢向量搜尋 |
| `float similarity(DocId a, DocId b)` | 兩個已索引文件的相似度 |
| `float similarity(const string& query, DocId doc)` | 查詢文字與已索引文件的相似度 |

#### 進階

| 方法 | 說明 |
|------|------|
| `BitVector build_vector(const string& text)` | 為任意文字建構位元向量 |
| `BitVector build_vector_from_tokens(const vector<string>& tokens)` | 從自訂 token 列表建構向量 |
| `float get_idf(const string& term)` | 查詢某 term 的 IDF 值 |
| `vector<string> extract_ngrams(const string& text)` | 提取 n-gram 特徵 |

#### 持久化

| 方法 | 說明 |
|------|------|
| `bool save(const string& path)` | 將索引（含 IDF 表、所有向量）寫入二進位檔案 |
| `bool load(const string& path)` | 從二進位檔案載入索引 |

### `egbi::BitVector`

| 方法 | 說明 |
|------|------|
| `BitVector(uint32_t num_bits)` | 建構指定長度的零向量 |
| `void set_bit(uint32_t pos)` | 設定第 pos 位元為 1 |
| `void clear_bit(uint32_t pos)` | 清除第 pos 位元 |
| `bool get_bit(uint32_t pos)` | 讀取第 pos 位元 |
| `uint32_t popcount()` | 位元中 1 的個數 |
| `static float jaccard(a, b)` | 兩個向量的位元 Jaccard 相似度 |
| `static uint32_t and_popcount(a, b)` | `|A ∩ B|` |
| `static uint32_t or_popcount(a, b)` | `|A ∪ B|` |

---

## 演算法原理

### 核心公式

1. **動態位元分配**：每個特徵 $t$ 分配位元數 $b_t = \lceil \alpha \cdot \text{IDF}(t) \rceil$
2. **相似度估算**：$S_{EGBI}(A,B) = \frac{\|V_A \land V_B\|_1}{\|V_A \lor V_B\|_1}$
3. **誤差上界**：$P(|S_{EGBI} - J_w| \ge \epsilon) \le 2\exp(-2\epsilon^2 k)$（Chernoff-Hoeffding 邊界）

### 為什麼有效

- **高 IDF 特徵** → 更多雜湊映射 → 在位元向量中佔據更大「質量」→ 主導相似度計算
- **低 IDF 噪音** → 極少位元 → 碰撞對分數的影響 ≤ $b_L / k$，結構性過濾
- **位元運算** → 單次 AND + OR + POPCNT，CPU 原生支援，無分支

---

## 調參指南

| 場景 | `bit_length` | `alpha` | `ngram_size` |
|------|-------------|---------|-------------|
| 快速原型 / 小語料 | 512 | 1.5 | 3 |
| 生產環境 / 萬級文件 | 1024 | 2.0 | 3 |
| 高精度需求 | 2048–4096 | 2.5 | 3–4 |
| 中文 / CJK 短文本 | 1024 | 2.0 | 2 |

> **CJK 文本建議**：中文字元在 UTF-8 下為 3 bytes，`ngram_size=3` 恰好對應單字級滑動窗口。若需雙字級，設為 `ngram_size=6`。或使用分詞器後呼叫 `build_vector_from_tokens()`。

---

## 專案結構

```
EGBI/
├── CMakeLists.txt            # 跨平台建構
├── README.md                 # 本文件
├── include/
│   └── egbi/
│       ├── egbi.hpp          # 單一 include 入口
│       ├── types.hpp         # Config, SearchResult
│       ├── bitvector.hpp     # BitVector + popcount
│       └── index.hpp         # EGBIIndex 核心
├── cli/
│   └── main.cpp              # EGBI-CLI 命令列工具
├── examples/
│   └── basic_usage.cpp       # 使用範例
├── tests/
│   └── test_egbi.cpp         # 測試套件
└── testdata/
    └── corpus.txt            # 範例語料庫
```

---

## 整合到你的專案

### 方式一：CMake `add_subdirectory`

```cmake
add_subdirectory(EGBI)
target_link_libraries(your_target PRIVATE egbi)
```

### 方式二：直接複製標頭

將 `include/egbi/` 目錄複製到你的 include 路徑即可。無需編譯任何 `.cpp` 檔案。

### 方式三：CMake `FetchContent`

```cmake
include(FetchContent)
FetchContent_Declare(egbi GIT_REPOSITORY <repo-url> GIT_TAG main)
FetchContent_MakeAvailable(egbi)
target_link_libraries(your_target PRIVATE egbi)
```

---

## 授權

MIT License
