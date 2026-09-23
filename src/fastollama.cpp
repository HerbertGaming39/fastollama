#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <limits.h>
#include <cstdint>

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string exe_dir() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = 0;
    std::string p(buf);
    size_t s = p.find_last_of('/');
    return s == std::string::npos ? "." : p.substr(0, s);
}

static bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static uint64_t file_size(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
}

struct Config {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d) const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    int geti(const std::string& k, int d) const {
        auto it = kv.find(k);
        if (it == kv.end()) return d;
        try { return std::stoi(it->second); } catch (...) { return d; }
    }
    float getf(const std::string& k, float d) const {
        auto it = kv.find(k);
        if (it == kv.end()) return d;
        try { return std::stof(it->second); } catch (...) { return d; }
    }
    bool getb(const std::string& k, bool d) const {
        std::string v = get(k, d ? "1" : "0");
        return v == "1" || v == "true" || v == "on" || v == "yes";
    }
};

static Config load_config() {
    Config c;
    std::string base = exe_dir();
    if (base.size() > 3 && base.substr(base.size() - 3) == "bin") base = base.substr(0, base.size() - 4);
    for (const std::string& path : { base + "/settings.txt", std::string("./settings.txt") }) {
        std::ifstream f(path);
        if (!f.is_open()) continue;
        std::string line;
        while (std::getline(f, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            c.kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
        }
        break;
    }
    return c;
}

struct GgufInfo {
    std::string arch;
    int n_layer = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int ctx_train = 0;
    int n_embd = 0;
    int n_vocab = 0;
    bool ok = false;
    bool is_moe = false;
    int n_expert_layers = 0;
    double expert_bytes_total = 0.0;
    int n_kv_layers = 0; // layers with real attention (hybrid archs: linear layers have 0 kv heads)
    struct LayerInfo {
        double total = 0.0;  // all weight bytes in this layer
        double keep = 0.0;   // bytes that must stay on GPU with the layer's KV (attn q/k/v + norms)
        bool has_attn = false;
    };
    std::map<int, LayerInfo> layers;
    double nonblk_bytes = 0.0; // token_embd, output head, etc.
    // cross-shard accumulators (shard N's tensor list only knows about shard N)
    double expert_sum_acc = 0.0;
    std::map<int, double> expert_per_layer_acc;
    std::map<int, int> attn_ids_acc;
    std::map<std::string, double> nonblk_families; // non-blk tensors grouped by name family
    double nonblk_gpu_bytes = 0.0;  // non-blk bytes small enough to keep on GPU
    std::string nonblk_cpu_regex;   // -ot alternation for giant tables (n-gram embeddings etc.)
    bool parsed = false;
    int shards = 0;
};

static double ggml_type_bpw(uint32_t t) {
    switch (t) {
        case 0: return 32.0;
        case 1: return 16.0;
        case 2: return 4.5;
        case 3: return 5.0;
        case 6: return 5.5;
        case 7: return 6.0;
        case 8: return 8.5;
        case 10: return 2.5625;
        case 11: return 3.4375;
        case 12: return 4.5;
        case 13: return 5.5;
        case 14: return 6.5625;
        case 15: return 8.5;
        case 16: return 2.0625;
        case 17: return 2.3125;
        case 18: return 3.0625;
        case 19: return 1.5625;
        case 20: return 4.5;
        case 21: return 3.4375;
        case 22: return 2.5;
        case 23: return 4.25;
        default: return 4.5; // unknown quant: assume ~4.5 bpw. NEVER f16 — an over-estimate
                             // wastes budget, an under-estimate kills the desktop
    }
}

static void gguf_read_str(std::ifstream& f, std::string& out) {
    uint64_t len = 0;
    f.read((char*)&len, 8);
    out.resize(len);
    if (len) f.read(&out[0], len);
}

template <typename T>
static void gguf_read_pod(std::ifstream& f, T& v) {
    f.read((char*)&v, sizeof(T));
}

static void gguf_skip_value(std::ifstream& f, uint32_t type);

static void gguf_skip_array(std::ifstream& f, uint32_t etype, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        if (etype == 8) {
            std::string s;
            gguf_read_str(f, s);
        } else if (etype == 9) {
            uint32_t t2; uint64_t c2;
            gguf_read_pod(f, t2);
            gguf_read_pod(f, c2);
            gguf_skip_array(f, t2, c2);
        } else {
            static const int sizes[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            f.seekg(sizes[etype], std::ios::cur);
        }
    }
}

static void gguf_skip_value(std::ifstream& f, uint32_t type) {
    if (type == 8) {
        std::string s;
        gguf_read_str(f, s);
    } else if (type == 9) {
        uint32_t et; uint64_t c;
        gguf_read_pod(f, et);
        gguf_read_pod(f, c);
        gguf_skip_array(f, et, c);
    } else {
        static const int sizes[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
        f.seekg(sizes[type], std::ios::cur);
    }
}

// Parse ONE gguf file's header + tensor list into g (called once per shard).
static void read_gguf_one(const std::string& path, GgufInfo& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return;
    uint32_t magic, version;
    uint64_t tensor_count, kv_count;
    f.read((char*)&magic, 4);
    if (magic != 0x46554747) return;
    f.read((char*)&version, 4);
    if (version < 2) return;
    gguf_read_pod(f, tensor_count);
    gguf_read_pod(f, kv_count);
    std::string arch_name;
    std::map<std::string, std::string> kvmap;
    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key;
        uint32_t type;
        gguf_read_str(f, key);
        gguf_read_pod(f, type);
        if (type == 8) {
            std::string s;
            gguf_read_str(f, s);
            kvmap[key] = s;
        } else if (type == 4 || type == 10) {
            uint64_t v = 0;
            f.read((char*)&v, type == 4 ? 4 : 8);
            kvmap[key] = std::to_string(v);
        } else if (type == 6 || type == 12) {
            double v = 0;
            f.read((char*)&v, type == 6 ? 4 : 8);
            kvmap[key] = std::to_string(v);
        } else if (type == 7) {
            uint8_t v;
            f.read((char*)&v, 1);
            kvmap[key] = std::to_string((int)v);
        } else if (type == 2 || type == 3) {
            uint16_t v;
            f.read((char*)&v, 2);
            kvmap[key] = std::to_string(v);
        } else if (type == 0 || type == 1) {
            uint8_t v;
            f.read((char*)&v, 1);
            kvmap[key] = std::to_string(v);
        } else if (type == 9) {
            // array: handle numeric arrays (per-layer hybrid archs like qwen35/qwen3next), else skip
            uint32_t et; uint64_t c;
            gguf_read_pod(f, et);
            gguf_read_pod(f, c);
            static const int esz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            if (et < 13 && esz[et] > 0 && c > 0 && c < 1000000) {
                double sum = 0; uint64_t nonzero = 0;
                for (uint64_t i = 0; i < c; i++) {
                    double v = 0;
                    char buf[8];
                    f.read(buf, esz[et]);
                    if (et == 4) { uint32_t x; memcpy(&x, buf, 4); v = x; }
                    else if (et == 5) { int32_t x; memcpy(&x, buf, 4); v = x; }
                    else if (et == 6) { float x; memcpy(&x, buf, 4); v = x; }
                    else if (et == 10) { uint64_t x; memcpy(&x, buf, 8); v = (double)x; }
                    else if (et == 11) { int64_t x; memcpy(&x, buf, 8); v = (double)x; }
                    else if (et == 12) { double x; memcpy(&x, buf, 8); v = x; }
                    else if (et == 2) { uint16_t x; memcpy(&x, buf, 2); v = x; }
                    else if (et == 3) { int16_t x; memcpy(&x, buf, 2); v = x; }
                    else { int8_t x; memcpy(&x, buf, 1); v = x; }
                    if (v != 0) { sum += v; nonzero++; }
                }
                if (nonzero > 0) kvmap[key] = std::to_string(sum / nonzero);
                else kvmap[key] = "0";
                // remember how many layers actually have attention (hybrid archs)
                if (key.find("attention.head_count_kv") != std::string::npos) {
                    kvmap["__kv_layers"] = std::to_string(nonzero);
                }
            } else {
                gguf_skip_array(f, et, c);
            }
        } else {
            gguf_skip_value(f, type);
        }
    }
    auto it = kvmap.find("general.architecture");
    // only arch-carrying shards (normally shard 1) define metadata. Later shards have
    // empty KV maps here — deriving from defaults would zero n_layer and flip g.ok
    // off AFTER earlier shards set it correctly.
    if (it != kvmap.end() && !it->second.empty()) {
        g.arch = it->second;
        auto getu = [&](const std::string& k, int d) {
            auto jt = kvmap.find(k);
            if (jt == kvmap.end()) return d;
            return atoi(jt->second.c_str());
        };
        g.n_layer   = getu(g.arch + ".block_count", 0);
        g.n_head_kv = getu(g.arch + ".attention.head_count_kv", 0);
        g.head_dim  = getu(g.arch + ".attention.key_length", getu(g.arch + ".attention.head_dim", 0));
        g.ctx_train = getu(g.arch + ".context_length", 0);
        g.n_embd    = getu(g.arch + ".embedding_length", 0);
        g.n_vocab   = getu(g.arch + ".vocab_size", 0);
        if (g.head_dim == 0) {
            int n_embd = getu(g.arch + ".embedding_length", 0);
            int n_head  = getu(g.arch + ".attention.head_count", 0);
            if (n_head > 0) g.head_dim = n_embd / n_head;
        }
        g.ok = g.n_layer > 0 && g.n_head_kv > 0 && g.head_dim > 0;
        g.n_kv_layers = getu("__kv_layers", 0);
        if (g.n_kv_layers <= 0 || g.n_kv_layers > g.n_layer) g.n_kv_layers = g.n_layer;
    }
    if (g.ok && tensor_count > 0 && tensor_count < 100000) {
        g.parsed = true;
        g.shards++;
        for (uint64_t i = 0; i < tensor_count; i++) {
            std::string name;
            uint32_t nd;
            uint64_t dims[4] = {1, 1, 1, 1};
            uint32_t ttype;
            uint64_t offset;
            gguf_read_str(f, name);
            f.read((char*)&nd, 4);
            if (nd > 4) break;
            for (uint32_t d = 0; d < nd; d++) f.read((char*)&dims[d], 8);
            f.read((char*)&ttype, 4);
            f.read((char*)&offset, 8);
            double bytes = (double)dims[0] * dims[1] * dims[2] * dims[3] * ggml_type_bpw(ttype) / 8.0;
            bool is_expert = name.find(".ffn_") != std::string::npos &&
                             name.find("exps") != std::string::npos;
            {
                size_t bp = name.find("blk.");
                if (bp == 0) {
                    size_t dp = name.find('.', bp + 4);
                    if (dp != std::string::npos) {
                        int li = atoi(name.substr(bp + 4, dp - bp - 4).c_str());
                        GgufInfo::LayerInfo& L = g.layers[li];
                        L.total += bytes;
                        // tensors that must stay with the layer's KV cache on the GPU
                        bool heavy = name.find("ffn_") != std::string::npos ||
                                     name.find("ssm_") != std::string::npos ||
                                     name.find("attn_qkv") != std::string::npos ||
                                     name.find("attn_gate") != std::string::npos ||
                                     name.find("attn_output") != std::string::npos ||
                                     name.find("nextn") != std::string::npos;
                        if (!heavy) { L.keep += bytes; if (name.find("attn_k") != std::string::npos) L.has_attn = true; }
                    }
                } else {
                    g.nonblk_bytes += bytes;
                    g.nonblk_families[name.substr(0, name.find('.'))] += bytes;
                }
            }
            if (name.find("attn_k") != std::string::npos || name.find("attn_v") != std::string::npos) {
                size_t bp = name.find("blk.");
                if (bp != std::string::npos) {
                    size_t dp = name.find('.', bp + 4);
                    if (dp != std::string::npos) g.attn_ids_acc[atoi(name.substr(bp + 4, dp - bp - 4).c_str())] = 1;
                }
            }
            if (is_expert) {
                g.expert_sum_acc += bytes;
                int li = -1;
                size_t bp = name.find("blk.");
                if (bp != std::string::npos) {
                    size_t dp = name.find('.', bp + 4);
                    if (dp != std::string::npos) li = atoi(name.substr(bp + 4, dp - bp - 4).c_str());
                }
                if (li >= 0) g.expert_per_layer_acc[li] += bytes;
            }
        }
    }
}

// Multi-shard GGUF: every shard carries the KV metadata but lists ONLY ITS OWN
// tensors. Parsing just shard 1 sees a sliver of the model (this exact bug sized a
// 70 GB model as 11 MB, planned it as full-GPU, and OOM-killed the session).
// Shard 1 provides arch metadata; every shard's tensors are folded in.
static GgufInfo read_gguf(const std::string& path) {
    GgufInfo g;
    read_gguf_one(path, g);
    std::string dir, fname;
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) { fname = path; }
    else { dir = path.substr(0, slash + 1); fname = path.substr(slash + 1); }
    const std::string pat = "-00001-of-0000";
    size_t p = fname.find(pat);
    if (p == std::string::npos || fname.size() < p + pat.size() + 1) return g;
    int n = fname[p + pat.size()] - '0';
    if (n < 2 || n > 9) return g;
    for (int i = 2; i <= n; i++) {
        char suf[32];
        snprintf(suf, sizeof(suf), "-000%02d-of-0000%d.gguf", i, n);
        read_gguf_one(dir + fname.substr(0, p) + suf, g);
    }
    // derive MoE / hybrid-attention facts from ALL shards' tensor lists
    if (!g.expert_per_layer_acc.empty()) {
        g.is_moe = true;
        g.n_expert_layers = (int)g.expert_per_layer_acc.size();
        g.expert_bytes_total = g.expert_sum_acc;
    }
    if (!g.attn_ids_acc.empty()) {
        g.n_kv_layers = (int)g.attn_ids_acc.size();
        if (g.n_kv_layers > 0 && g.n_kv_layers <= g.n_layer) {
            // +1 headroom: linear layers keep small fixed recurrent state
            g.n_kv_layers = std::min(g.n_layer, g.n_kv_layers + 1);
        }
    }
    // giant non-block tables (Qwen4-preview per-layer/n-gram embeddings: 28+ GB) cannot
    // live on any GPU — they are row-gathered, so RAM streaming is cheap by design.
    // Small families (output head, plain embeddings) stay on GPU.
    for (auto& fam : g.nonblk_families) {
        if (fam.second > 1.0e9) {
            if (!g.nonblk_cpu_regex.empty()) g.nonblk_cpu_regex += "|";
            g.nonblk_cpu_regex += fam.first;
        } else {
            g.nonblk_gpu_bytes += fam.second;
        }
    }
    return g;
}

static uint64_t model_file_size_all_shards(const std::string& model) {
    uint64_t sz = file_size(model);
    std::string dir, fname;
    size_t slash = model.find_last_of('/');
    if (slash == std::string::npos) { fname = model; }
    else { dir = model.substr(0, slash + 1); fname = model.substr(slash + 1); }
    const std::string pat = "-00001-of-0000";
    size_t p = fname.find(pat);
    if (p == std::string::npos || fname.size() < p + pat.size() + 1) return sz;
    int n = fname[p + pat.size()] - '0';
    if (n < 2 || n > 9) return sz;
    for (int i = 2; i <= n; i++) {
        char suf[32];
        snprintf(suf, sizeof(suf), "-000%02d-of-0000%d.gguf", i, n);
        sz += file_size(dir + fname.substr(0, p) + suf);
    }
    return sz;
}

static double kv_bytes_per_elem(const std::string& q) {
    if (q == "f16" || q == "bf16") return 2.0;
    if (q == "q8_0") return 34.0 / 32.0;
    if (q == "q4_0") return 18.0 / 32.0;
    if (q == "q4_1") return 20.0 / 32.0;
    if (q == "q5_0") return 22.0 / 32.0;
    if (q == "q5_1") return 24.0 / 32.0;
    if (q == "q6_0") return 26.0 / 32.0;
    if (q == "iq4_nl" || q == "q6_k") return 0.8;
    if (q == "q5_k" || q == "q5_k_m" || q == "iq5") return 0.75;
    return 0.5625;
}

static int amd_card_with_largest_vram() {
    int best = -1;
    uint64_t best_total = 0;
    for (int i = 0; i < 8; i++) {
        std::string p = "/sys/class/drm/card" + std::to_string(i) + "/device/";
        std::ifstream vf(p + "vendor");
        std::string v;
        std::getline(vf, v);
        if (v.find("1002") == std::string::npos) continue;
        std::ifstream tf(p + "mem_info_vram_total");
        uint64_t t = 0;
        if (tf.is_open()) {
            tf >> t;
            if (t > best_total) { best_total = t; best = i; }
        }
    }
    return best;
}

static uint64_t amd_vram_total() {
    int c = amd_card_with_largest_vram();
    if (c < 0) return 16304ull * 1024 * 1024;
    std::ifstream tf("/sys/class/drm/card" + std::to_string(c) + "/device/mem_info_vram_total");
    uint64_t t = 0;
    if (tf.is_open()) tf >> t;
    return t ? t : 16304ull * 1024 * 1024;
}

static uint64_t amd_vram_used() {
    int c = amd_card_with_largest_vram();
    if (c < 0) return 0;
    std::ifstream tf("/sys/class/drm/card" + std::to_string(c) + "/device/mem_info_vram_used");
    uint64_t u = 0;
    if (tf.is_open()) tf >> u;
    return u;
}

static uint64_t ram_available_bytes() {
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || psize <= 0) return 0;
    return (uint64_t)pages * (uint64_t)psize;
}


static bool file_exists(const std::string& path);
struct App;

// full model + KV + compute must fit in the VRAM budget with ZERO tensors on CPU:
// on a hybrid arch every CPU-resident tensor (even FFN) costs RAM bandwidth per token,
// so the governor's job in auto mode is to pick the biggest quant that avoids that
struct VramPlan {
    int ngl = 0;
    uint64_t est_bytes = 0;
    uint64_t allowed_bytes = 0;
    int expert_cpu_from = 0;
    int n_expert_layers = 0;
    std::string ot_regex; // smart mode: offload heavy tensors of these layers to CPU
    bool full_gpu = false; // true = nothing streams from RAM at generation time
};

static VramPlan plan_vram(const App& app, const std::string& model);

struct App {
    Config cfg;
    std::string base;
    std::string lbin;
    std::string model_path(const std::string& name) const {
        if (name.find('/') != std::string::npos) return name;
        // values may already carry the .gguf extension (e.g. MTP sidecars)
        std::string raw = base + "/models/" + name;
        if (file_exists(raw)) return raw;
        std::string p = base + "/models/" + name + ".gguf";
        if (file_exists(p)) return p;
        // multi-shard models: the user names the model, we resolve to shard 1
        // (llama.cpp finds the rest automatically). Supports flat layout
        // (models/X-00001-of-0000M.gguf) and dir layout (models/X/X-00001-of-...).
        for (int n = 2; n <= 9; n++) {
            char t[80];
            snprintf(t, sizeof(t), "-00001-of-0000%d.gguf", n);
            std::string flat = base + "/models/" + name + t;
            if (file_exists(flat)) return flat;
            std::string in_dir = base + "/models/" + name + "/" + name + t;
            if (file_exists(in_dir)) return in_dir;
        }
        return p;
    }
    // model = auto: walk the Qwen3.8-27B quant ladder from best quality down and pick
    // the LARGEST quant whose whole model fits on the GPU (plan.full_gpu). A fully-GPU
    // small quant beats a bigger quant that streams half its weights from RAM.
    std::string main_model() const {
        std::string want = cfg.get("model", "Qwen3-8B-Q4_K_M");
        if (want != "auto") return model_path(want);
        static const char* LADDER[] = {
            "Qwen3.8-Flash-Next-UD-IQ1_S", // 177B-class, only if downloaded (governor splits it)
            "Qwen3.8-27B-UD-Q4_K_XL", "Qwen3.8-27B-UD-Q4_K_M", "Qwen3.8-27B-UD-Q4_K_S",
            "Qwen3.8-27B-UD-IQ4_XS",  "Qwen3.8-27B-UD-Q3_K_XL", "Qwen3.8-27B-UD-IQ3_XXS",
            "Qwen3.8-27B-UD-IQ2_S",   "Qwen3.8-27B-UD-IQ2_XXS",
        };
        std::string best_avail;
        for (const char* q : LADDER) {
            std::string cand = model_path(q);
            if (!file_exists(cand)) continue;
            if (best_avail.empty()) best_avail = cand; // biggest downloaded quant
            VramPlan p = plan_vram(*this, cand);
            if (p.full_gpu) {
                std::cerr << "[fastollama] model=auto -> " << q
                          << " (largest quant that fits 100% on GPU)\n";
                return cand;
            }
        }
        if (!best_avail.empty()) {
            std::cerr << "[fastollama] model=auto -> " << best_avail
                      << " (nothing fits fully on GPU; governor will split it)\n";
            return best_avail;
        }
        return model_path("Qwen3-8B-Q4_K_M");
    }
    std::string draft_model() const { return model_path(cfg.get("draft_model", "Qwen3-0.6B-Q4_K_M")); }
    std::string llama_server() const {
        std::string ed = cfg.get("engine_dir", "");
        if (!ed.empty()) {
            if (ed[0] != '/') ed = base + "/" + ed;
            return ed + "/llama-server";
        }
        return base + "/llama.cpp/" + backend_dir() + "/bin/llama-server";
    }
    std::string llama_cli() const { return base + "/llama.cpp/" + backend_dir() + "/bin/llama-cli"; }
    std::string llama_bench() const { return base + "/llama.cpp/" + backend_dir() + "/bin/llama-bench"; }
    std::string llama_quantize() const { return base + "/llama.cpp/" + backend_dir() + "/bin/llama-quantize"; }
    std::string backend_dir() const {
        std::string b = cfg.get("backend", "vulkan");
        return b == "hip" || b == "rocm" ? "build-hip" : b == "cpu" ? "build-cpu" : "build-vk";
    }
    void apply_env() const {
        std::string b = cfg.get("backend", "vulkan");
        if (b == "hip" || b == "rocm") {
            setenv("GGML_ROCM_DEVICES", "0", 1);
        } else if (b == "vulkan") {
            setenv("GGML_VK_VISIBLE_DEVICES", "0", 1);
        }
        setenv("GGML_VK_ALLOW_SYSMEM_FALLBACK", "0", 1);
    }
};

// RAM is the other half of the budget. A model whose CPU-resident share exceeds
// available RAM does not merely run slow — it thrashes disk via mmap and can
// freeze the whole desktop (page allocation failures in unrelated processes).
// Refuse that launch unless the user explicitly asks for disk streaming.
static bool ram_feasible_or_die(const App& app, const std::string& model, uint64_t est_gpu_bytes) {
    uint64_t msz = model_file_size_all_shards(model);
    uint64_t ram_need = msz > est_gpu_bytes ? msz - est_gpu_bytes : 0;
    double reserve = app.cfg.getf("ram_reserve_gb", 4.0) * 1024.0 * 1024.0 * 1024.0;
    uint64_t avail = ram_available_bytes();
    if (ram_need + (uint64_t)reserve <= avail) return true;
    std::cerr << "\n[fastollama] RAM BUDGET: this plan needs " << ram_need / 1024 / 1024
              << " MiB of system RAM (model " << msz / 1024 / 1024 << " MiB minus "
              << est_gpu_bytes / 1024 / 1024 << " MiB on GPU), but only "
              << avail / 1024 / 1024 << " MiB is available (keeping "
              << (uint64_t)reserve / 1024 / 1024 << " MiB reserve for the desktop).\n";
    std::cerr << "[fastollama] Running it anyway would stream weights from disk and can\n"
              << "             freeze the entire system, not just the model.\n";
    if (app.cfg.getb("allow_disk_stream", false)) {
        std::cerr << "[fastollama] allow_disk_stream = 1 -> starting anyway. Expect heavy slowdowns.\n";
        return true;
    }
    std::cerr << "[fastollama] Refusing to start. Use a smaller quant (fastollama pull ...),\n"
              << "             or set allow_disk_stream = 1 in settings.txt to override.\n";
    return false;
}

static VramPlan plan_vram(const App& app, const std::string& model) {
    VramPlan p{};
    GgufInfo g = read_gguf(model);
    uint64_t model_size = model_file_size_all_shards(model);
    if (!g.ok || model_size == 0) {
        // UNKNOWN/unparseable model: never assume it fits. ngl=999 with no plan
        // behind it is exactly how a desktop dies. Fail safe to RAM, tell the user.
        p.ngl = 0;
        p.est_bytes = model_size;
        std::cerr << "[vram-governor] WARNING: cannot parse " << model
                  << " — failing safe to ngl=0 (all weights in RAM).\n"
                  << "[vram-governor] Set gpu_layers manually in settings.txt to override.\n";
        return p;
    }
    uint64_t ctx = (uint64_t)app.cfg.geti("context", 40960);
    std::string kvq = app.cfg.get("kv_cache", "q8_0");
    double per_token = 2.0 * g.n_head_kv * g.head_dim * kv_bytes_per_elem(kvq);
    double kv_bytes = per_token * (double)g.n_kv_layers * (double)ctx * 1.05;
    int ub = app.cfg.geti("ubatch", 512);
    double compute_bytes = 0.45e9 + 0.7e-3 * ub * 1e6 + 0.15e9;
    // MTP sidecar weights live on the GPU too
    std::string spec_type = app.cfg.get("spec_type", "");
    if (spec_type.empty()) spec_type = app.cfg.getb("speculative", false) ? "draft" : "off";
    if (spec_type == "draft-mtp") {
        std::string mtp = app.model_path(app.cfg.get("mtp_model", ""));
        // MTP draft weights + its own KV/compute buffers (measured: ~1.35 GB on top of the file)
        if (read_gguf(model).arch == read_gguf(mtp).arch) {
            compute_bytes += (double)file_size(mtp) + 1.35e9;
        }
    }
    // measured Vulkan/RDNA4 allocation overhead over raw weights+KV (logits, graphs, shards)
    double margin = app.cfg.get("backend", "vulkan") == "cpu" ? 0.4e9 : 1.3e9;
    std::string mq = app.cfg.get("model", "");
    double out_bytes = 0.0;
    if (g.n_vocab > 0 && g.n_embd > 0) {
        double be = kv_bytes_per_elem(mq.find("Q8") != std::string::npos ? "q8_0"
                     : mq.find("Q4_0") != std::string::npos ? "q4_0"
                     : mq.find("Q5") != std::string::npos ? "q5_k"
                     : mq.find("Q6") != std::string::npos ? "q6_k"
                     : mq.find("f16") != std::string::npos || mq.find("bf16") != std::string::npos ? "f16"
                     : "iq4_nl");
        out_bytes = (double)g.n_vocab * g.n_embd * be;
    }
    double limit_gb = app.cfg.getf("vram_limit_gb", 15.0);
    double free_min_gb = app.cfg.getf("vram_free_min_gb", 1.3);
    uint64_t total = amd_vram_total();
    uint64_t used = amd_vram_used();
    double allowed_d = (double)std::min<uint64_t>(
        (uint64_t)(limit_gb * 1024.0 * 1024.0 * 1024.0),
        total > used ? total - used : 0);
    double total_free = total > used ? (double)(total - used) : 0.0;
    allowed_d = std::min(allowed_d, std::max(0.0, total_free - free_min_gb * 1024.0 * 1024.0 * 1024.0));
    p.allowed_bytes = (uint64_t)std::max(0.0, allowed_d);    int n_layer = g.n_layer;

    // FULL-GPU shortcut: if the entire model + KV + compute fits the budget, put
    // everything on the GPU with no -ot tricks. On hybrid archs any CPU-resident
    // tensor (even FFN) streams RAM every token; total-GPU is a different speed class.
    {
        double est_all = (double)model_size + kv_bytes + compute_bytes + margin * 0.5;
        if (est_all <= (double)p.allowed_bytes) {
            p.ngl = n_layer + 1;
            p.est_bytes = (uint64_t)est_all;
            p.full_gpu = true;
            p.n_expert_layers = 0;
            return p;
        }
    }
    // --- smart mode: keep ALL attention layers + KV on GPU, fill the rest of the
    //     budget with the heaviest layers' weights, offload the rest via -ot regex ---
    if (app.cfg.get("ot_mode", "layers") == "smart" && !g.layers.empty() && !g.is_moe) {
        double fixed_gpu = kv_bytes + compute_bytes + margin + g.nonblk_bytes;
        // the -ot regex only offloads ffn_ tensors, so EVERY layer's non-FFN weights
        // (attention + linear-attention/ssm/gating) stay on GPU and must be budgeted
        for (auto& kv : g.layers) fixed_gpu += kv.second.keep;
        double budget = (double)p.allowed_bytes - fixed_gpu;
        if (budget > 0) {
            std::vector<std::pair<double,int>> heavy; // (offloadable bytes, layer id)
            for (auto& kv : g.layers) heavy.push_back({kv.second.total - kv.second.keep, kv.first});
            std::sort(heavy.begin(), heavy.end(), [](auto& a, auto& b){ return a.first > b.first; });
            double acc = 0; std::vector<int> cpu_layers;
            const double slack = budget * 0.02; // float rounding headroom so an exact fit stays on GPU
            for (auto& h : heavy) {
                if (acc + h.first <= budget + slack) { acc += h.first; }
                else cpu_layers.push_back(h.second);
            }
            if (cpu_layers.empty()) {
                // the entire model fits on the GPU: zero CPU streaming, max speed
                p.ngl = n_layer + 1;
                p.est_bytes = (uint64_t)(fixed_gpu + acc);
                p.full_gpu = true;
                p.n_expert_layers = 0;
                return p;
            }
            if (!cpu_layers.empty()) {
                std::sort(cpu_layers.begin(), cpu_layers.end());
                std::string ids;
                for (size_t i = 0; i < cpu_layers.size(); i++) {
                    if (i) ids += "|";
                    ids += std::to_string(cpu_layers[i]);
                }
                // offload ONLY feed-forward weights: ssm_/attn_ tensors run per-token with
                // sequential dependencies and would serialize the whole graph on every step
                p.ot_regex = "blk\\.(" + ids + ")\\.ffn_";
                p.ngl = n_layer + 1; // everything nominally on GPU; -ot moves heavy tensors to CPU
                p.est_bytes = (uint64_t)(fixed_gpu + acc);
                p.n_expert_layers = 0;
                return p;
            }
        }
        // budget too small for smart mode: fall through to classic layer split
    }
    if (g.is_moe && g.n_expert_layers > 0) {
        // `other` = everything that is not expert weights. The giant per-layer tables
        // inside it stream from RAM, so only nonblk_gpu_bytes counts against VRAM.
        double nonblk_cpu = g.nonblk_bytes - g.nonblk_gpu_bytes;
        double other = (double)model_size - g.expert_bytes_total - nonblk_cpu;
        double epl = g.expert_bytes_total / g.n_expert_layers;
        // Fixed GPU cost = everything that stays on VRAM no matter how many expert
        // layers we place: attention/ssm core, output head, KV cache, compute buffers.
        // Analytic: other + kv + compute + margin. A measured floor is far better:
        // run once with all experts offloaded (K=0) and put the engine's VRAM usage
        // (GiB) into vram_fixed_gpu_gb — the planner then only spends what's REALLY left.
        double fixed_gpu = other + kv_bytes + compute_bytes + margin;
        double floor_gb = app.cfg.getf("vram_fixed_gpu_gb", 0.0);
        if (floor_gb > 0) fixed_gpu = floor_gb * 1024.0 * 1024.0 * 1024.0;
        double est_safety = app.cfg.getf("est_safety_pct", 1.10); // per-layer headroom
        double budget_left = (double)p.allowed_bytes - fixed_gpu;
        int K = budget_left > 0 ? (int)(budget_left / (epl * est_safety)) : 0;
        if (K < 0) K = 0;
        if (K > g.n_expert_layers) K = g.n_expert_layers;
        p.ngl = n_layer + 1;
        p.expert_cpu_from = K;
        p.n_expert_layers = g.n_expert_layers;
        p.est_bytes = (uint64_t)(fixed_gpu + (double)K * epl);
        p.full_gpu = (K >= g.n_expert_layers && nonblk_cpu == 0);
        // combined -ot: giant tables first, then the CPU-resident expert layers
        std::string re;
        if (!g.nonblk_cpu_regex.empty()) re += g.nonblk_cpu_regex;
        if (K < g.n_expert_layers) {
            std::string layers;
            for (int i = K; i < g.n_expert_layers; i++) {
                if (!layers.empty()) layers += "|";
                layers += std::to_string(i);
            }
            if (!re.empty()) re += "|";
            re += "blk\\.(" + layers + ")\\..*exps";
        }
        if (!re.empty()) p.ot_regex = re;
        return p;
    }
    p.ngl = 0;
    for (int ngl = n_layer + 1; ngl >= 0; ngl--) {
        double layers_frac = (double)std::min(ngl, n_layer) / n_layer;
        double est = (model_size - out_bytes) * layers_frac
                   + (ngl > n_layer ? out_bytes : 0.0)
                   + kv_bytes * layers_frac
                   + compute_bytes + margin;
        if (est <= (double)p.allowed_bytes) {
            p.ngl = ngl;
            p.est_bytes = (uint64_t)est;
            p.full_gpu = (ngl >= n_layer + 1);
            break;
        }
    }
    return p;
}

static void report_plan(App& app, const std::string& model, const VramPlan& p) {
    GgufInfo g = read_gguf(model);
    uint64_t total = amd_vram_total();
    uint64_t used = amd_vram_used();
    std::cerr << "[vram-governor] total " << total / 1024 / 1024 << " MiB, in use "
              << used / 1024 / 1024 << " MiB, budget " << p.allowed_bytes / 1024 / 1024
              << " MiB, est. need " << p.est_bytes / 1024 / 1024 << " MiB -> ngl "
              << p.ngl << "/" << g.n_layer
              << (p.full_gpu ? "  [FULL-GPU: zero RAM streaming]" : "") << "\n";
    if (g.is_moe && p.n_expert_layers > 0) {
        std::cerr << "[vram-governor] MoE: expert layers 0.." << p.expert_cpu_from - 1
                  << " on GPU, " << p.expert_cpu_from << ".." << p.n_expert_layers - 1
                  << " on CPU (of " << p.n_expert_layers << ")\n";
    }
    if (p.ngl == 0) std::cerr << "[vram-governor] WARNING: model will run on CPU/RAM, too slow!\n";
}

// -ot regex that sends planned expert layers to CPU. expert_cpu_from==0 means ALL
// expert layers stream from RAM (the 80B-class playbook) — the old code emitted NO
// regex in that case, which told the engine to load every expert onto the GPU.
static std::string expert_offload_regex(const VramPlan& p) {
    if (p.expert_cpu_from <= 0) return "blk\\..*exps";
    std::string layers;
    for (int i = p.expert_cpu_from; i < p.n_expert_layers; i++) {
        if (i > p.expert_cpu_from) layers += "|";
        layers += std::to_string(i);
    }
    return "blk\\.(" + layers + ")\\..*exps";
}

static std::vector<char*> to_argv(const std::vector<std::string>& in) {
    std::vector<char*> out;
    for (auto& s : in) out.push_back(const_cast<char*>(s.c_str()));
    out.push_back(nullptr);
    return out;
}

static pid_t run(const std::vector<std::string>& args, bool wait = true) {
    auto av = to_argv(args);
    pid_t pid = fork();
    if (pid == 0) {
        execv(args[0].c_str(), av.data());
        std::cerr << "failed to exec " << args[0] << ": " << strerror(errno) << "\n";
        _exit(127);
    }
    if (wait) {
        int st = 0;
        waitpid(pid, &st, 0);
        if (WIFEXITED(st) && WEXITSTATUS(st) != 0) exit(WEXITSTATUS(st));
    }
    return pid;
}

static void gpu_args(App& app, const std::string& model, std::vector<std::string>& v) {
    std::string ngl;
    if (app.cfg.get("backend", "vulkan") == "cpu") {
        ngl = "0";
    } else if (app.cfg.getb("full_gpu", false)) {
        // explicit full-GPU mode: all weights+KV on the GPU. Fastest possible config for
        // models that fit; llama.cpp aborts safely (no sysmem fallback) if they don't.
        VramPlan probe = plan_vram(app, model);
        // chokepoint: NO plan may exceed the budget. Every branch of plan_vram is
        // supposed to guarantee this; if a future arch breaks that assumption we
        // fail safe (all weights in RAM) instead of OOM-killing the desktop.
        if (probe.est_bytes > probe.allowed_bytes + probe.allowed_bytes * 0.02) {
            std::cerr << "[vram-governor] SAFETY: plan est " << probe.est_bytes / 1024 / 1024
                      << " MiB > budget " << probe.allowed_bytes / 1024 / 1024
                      << " MiB — failing safe to ngl=0\n";
            probe.ngl = 0;
            probe.ot_regex.clear();
            probe.n_expert_layers = 0;
            probe.full_gpu = false;
            probe.est_bytes = 0; // ngl=0: nothing on GPU -> RAM must hold the whole model
            ngl = "0";
            if (!ram_feasible_or_die(app, model, probe.est_bytes)) exit(1);
        } else if (probe.full_gpu) {
            // 2% slack: compute/MTP estimates are deliberately padded; empirical loads land
            // ~0.5GB under estimate. Budget itself already enforces the hard VRAM limit.
            ngl = "999";
            std::cerr << "[vram-governor] full_gpu: whole model on GPU, est "
                      << probe.est_bytes / 1024 / 1024 << " MiB (budget "
                      << probe.allowed_bytes / 1024 / 1024 << " MiB)\n";
        } else {
            std::cerr << "[vram-governor] full_gpu=1 but model+KV est "
                      << probe.est_bytes / 1024 / 1024 << " MiB > budget "
                      << probe.allowed_bytes / 1024 / 1024 << " MiB -> governor split\n";
            report_plan(app, model, probe);
            if (!ram_feasible_or_die(app, model, probe.est_bytes)) exit(1);
            ngl = std::to_string(probe.ngl);
            if (!probe.ot_regex.empty()) {
                std::cerr << "[vram-governor] smart -ot offloading " << probe.ot_regex.substr(0, 60) << "... =CPU\n";
                v.push_back("-ot");
                v.push_back(probe.ot_regex + "=CPU");
            } else if (probe.n_expert_layers > 0 && probe.expert_cpu_from < probe.n_expert_layers) {
                std::string re = expert_offload_regex(probe);
                std::cerr << "[vram-governor] -ot \"" << re << "=CPU\"\n";
                v.push_back("-ot");
                v.push_back(re + "=CPU");
            }
        }
    } else if (!app.cfg.getb("auto_vram", true)) {
        ngl = std::to_string(app.cfg.geti("gpu_layers", 999));
    } else {
        VramPlan p2 = plan_vram(app, model);
        if (p2.est_bytes > p2.allowed_bytes + p2.allowed_bytes * 0.02) {
            std::cerr << "[vram-governor] SAFETY: plan est " << p2.est_bytes / 1024 / 1024
                      << " MiB > budget " << p2.allowed_bytes / 1024 / 1024
                      << " MiB — failing safe to ngl=0\n";
            p2.ngl = 0;
            p2.ot_regex.clear();
            p2.n_expert_layers = 0;
            p2.est_bytes = 0; // ngl=0: RAM must hold the whole model
            if (!ram_feasible_or_die(app, model, p2.est_bytes)) exit(1);
        }
        report_plan(app, model, p2);
        if (!ram_feasible_or_die(app, model, p2.est_bytes)) exit(1);
        ngl = std::to_string(p2.ngl);
        if (!p2.ot_regex.empty()) {
            std::cerr << "[vram-governor] smart -ot offloading " << p2.ot_regex.substr(0, 60) << "... =CPU\n";
            v.push_back("-ot");
            v.push_back(p2.ot_regex + "=CPU");
        } else if (p2.n_expert_layers > 0 && p2.expert_cpu_from < p2.n_expert_layers) {
            std::string re = expert_offload_regex(p2);
            std::cerr << "[vram-governor] -ot \"" << re << "=CPU\"\n";
            v.push_back("-ot");
            v.push_back(re + "=CPU");
        }
    }
    v.push_back("-ngl");
    v.push_back(ngl);
}

static std::vector<std::string> model_args(App& app, bool want_draft) {
    std::vector<std::string> v;
    std::string m = app.main_model();
    v.push_back("-m");
    v.push_back(m);
    gpu_args(app, m, v);
    v.push_back("-t");
    v.push_back(std::to_string(app.cfg.geti("threads", 16)));
    v.push_back("-tb");
    v.push_back(std::to_string(app.cfg.geti("threads_batch", 16)));
    v.push_back("-b");
    v.push_back(std::to_string(app.cfg.geti("batch", 2048)));
    v.push_back("-ub");
    v.push_back(std::to_string(app.cfg.geti("ubatch", 512)));
    v.push_back("-c");
    v.push_back(std::to_string(app.cfg.geti("context", 40960)));
    if (app.cfg.getb("flash_attn", true)) { v.push_back("-fa"); v.push_back("on"); }
    std::string kv = app.cfg.get("kv_cache", "q8_0");
    if (kv != "f16" && app.cfg.getb("flash_attn", true)) {
        v.push_back("-ctk"); v.push_back(kv);
        v.push_back("-ctv"); v.push_back(kv);
    }
    // speculative decoding: spec_type = draft (small model) or draft-mtp (MTP sidecar)
    std::string spec_type = app.cfg.get("spec_type", "");
    if (spec_type.empty()) spec_type = app.cfg.getb("speculative", false) ? "draft" : "off";
    if (spec_type == "draft-mtp") {
        std::string mtp = app.model_path(app.cfg.get("mtp_model", ""));
        // MTP sidecar must share the main model's architecture, else llama.cpp aborts
        bool arch_ok = read_gguf(m).arch == read_gguf(mtp).arch;
        if (file_exists(mtp) && arch_ok) {
            v.push_back("--model-draft");
            v.push_back(mtp);
            v.push_back("--spec-type");
            v.push_back("draft-mtp");
            v.push_back("--spec-draft-n-max");
            v.push_back(std::to_string(app.cfg.geti("draft_max", 6)));
            v.push_back("--spec-draft-n-min");
            v.push_back(std::to_string(app.cfg.geti("draft_min", 1)));
        } else {
            std::cerr << "[fastollama] mtp_model missing or arch mismatch (" << mtp << "), running without MTP\n";
        }
    } else if (spec_type == "draft" && want_draft && file_exists(app.draft_model())) {
        v.push_back("--model-draft");
        v.push_back(app.draft_model());
        v.push_back("-ngld");
        v.push_back(std::to_string(app.cfg.geti("gpu_layers_draft", 999)));
        v.push_back("--spec-draft-n-max");
        v.push_back(std::to_string(app.cfg.geti("draft_max", 24)));
        v.push_back("--spec-draft-n-min");
        v.push_back(std::to_string(app.cfg.geti("draft_min", 4)));
        v.push_back("--spec-draft-p-min");
        v.push_back(std::to_string(app.cfg.getf("draft_p_min", 0.75f)));
    }
    if (app.cfg.getb("yarn", false)) {
        v.push_back("--rope-scaling"); v.push_back("yarn");
        v.push_back("--rope-scale"); v.push_back(std::to_string(app.cfg.getf("yarn_scale", 4.0f)));
        v.push_back("--rope-freq-base"); v.push_back("1000000");
        if (app.cfg.geti("context", 40960) > 40960) {
            v.push_back("--yarn-orig-ctx"); v.push_back("40960");
        }
    }
    if (app.cfg.getb("mlock", false)) v.push_back("--mlock");
    if (app.cfg.getb("no_mmap", false)) v.push_back("--no-mmap");
    // defrag: deprecated upstream (auto-defrag now); kept as no-op for old settings
    if (app.cfg.getf("defrag", 0.1f) <= 0.0f) { /* 0 = off, nothing to do */ }
    if (app.cfg.getb("kv_unified", false)) v.push_back("--kv-unified");
    int cre = app.cfg.geti("cache_reuse", 0);
    if (cre > 0) { v.push_back("--cache-reuse"); v.push_back(std::to_string(cre)); }
    if (app.cfg.getb("no_context_shift", false)) v.push_back("--no-context-shift");
    return v;
}

static std::vector<std::string> sampler_args(App& app) {
    std::vector<std::string> v;
    v.push_back("--temp"); v.push_back(std::to_string(app.cfg.getf("temp", 0.6f)));
    v.push_back("--top-p"); v.push_back(std::to_string(app.cfg.getf("top_p", 0.95f)));
    v.push_back("--top-k"); v.push_back(std::to_string(app.cfg.geti("top_k", 20)));
    v.push_back("--min-p"); v.push_back(std::to_string(app.cfg.getf("min_p", 0.0f)));
    return v;
}

static void thinking_args(App& app, std::vector<std::string>& v) {
    if (app.cfg.getb("jinja", true)) v.push_back("--jinja");
    if (!app.cfg.getb("thinking", true)) {
        v.push_back("--reasoning-budget");
        v.push_back("0");
    }
}

static void cmd_serve(App& app, const std::vector<std::string>& extra) {
    std::string m = app.main_model();
    if (!file_exists(m)) {
        std::cerr << "model not found: " << m << "\nrun: fastollama pull qwen3-8b\n";
        exit(1);
    }
    app.apply_env();
    std::vector<std::string> v;
    v.push_back(app.llama_server());
    auto ma = model_args(app, true);
    v.insert(v.end(), ma.begin(), ma.end());
    v.push_back("--host"); v.push_back(app.cfg.get("host", "127.0.0.1"));
    v.push_back("--port"); v.push_back(std::to_string(app.cfg.geti("port", 8080)));
    v.push_back("-np"); v.push_back(std::to_string(app.cfg.geti("parallel", 1)));
    thinking_args(app, v);
    v.insert(v.end(), extra.begin(), extra.end());
    std::cerr << "fastollama serving " << m << " -> http://" << app.cfg.get("host", "127.0.0.1")
              << ":" << app.cfg.geti("port", 8080) << "\n";
    pid_t srv = run(v);
    // VRAM guard dog: llama-server has no --vram-limit flag, so we enforce it from the
    // outside. If VRAM use crosses the emergency line (default 99.5% of total) we stop
    // the server cleanly instead of letting the driver thrash/freeze the desktop.
    // Budget-side enforcement happens in plan_vram (vram_limit_gb); this is the last resort.
    if (srv > 0 && app.cfg.getb("vram_guard", true)) {
        double em = app.cfg.getf("vram_emergency_pct", 0.95f); // 95%: the 0.2s poll can't outrun a load spike; the plan-side budget is the real enforcement
        pid_t guard = fork();
        if (guard == 0) {
            uint64_t total = amd_vram_total();
            while (true) {
                usleep(200 * 1000);
                if (kill(srv, 0) != 0) _exit(0); // server already gone
                uint64_t used = amd_vram_used();
                if (total > 0 && used >= (uint64_t)((double)total * em)) {
                    std::cerr << "\n[fastollama] VRAM GUARD: " << used / 1024 / 1024 << "/"
                              << total / 1024 / 1024 << " MiB used — stopping server to protect the desktop\n";
                    kill(srv, SIGTERM);
                    usleep(3000000);
                    kill(srv, SIGKILL);
                    _exit(42);
                }
            }
        }
    }
}

static void cmd_chat(App& app, const std::vector<std::string>& extra) {
    app.apply_env();
    std::vector<std::string> v;
    v.push_back(app.llama_cli());
    auto ma = model_args(app, true);
    v.insert(v.end(), ma.begin(), ma.end());
    auto sa = sampler_args(app);
    v.insert(v.end(), sa.begin(), sa.end());
    v.push_back("-cnv");
    thinking_args(app, v);
    v.insert(v.end(), extra.begin(), extra.end());
    run(v);
}

static void cmd_ask(App& app, const std::string& prompt, const std::vector<std::string>& extra) {
    app.apply_env();
    std::vector<std::string> v;
    v.push_back(app.llama_cli());
    auto ma = model_args(app, false);
    v.insert(v.end(), ma.begin(), ma.end());
    auto sa = sampler_args(app);
    v.insert(v.end(), sa.begin(), sa.end());
    v.push_back("-st");
    v.push_back("--no-display-prompt");
    v.push_back("-p"); v.push_back(prompt);
    thinking_args(app, v);
    v.insert(v.end(), extra.begin(), extra.end());
    run(v);
}

static void cmd_bench(App& app, const std::vector<std::string>& extra) {
    app.apply_env();
    std::string m = app.main_model();
    std::vector<std::string> v;
    v.push_back(app.llama_bench());
    v.push_back("-m"); v.push_back(m);
    gpu_args(app, m, v);
    v.push_back("-t"); v.push_back(std::to_string(app.cfg.geti("threads", 16)));
    v.push_back("-fa"); v.push_back(app.cfg.getb("flash_attn", true) ? "on" : "off");
    std::string kv = app.cfg.get("kv_cache", "q8_0");
    if (kv != "f16" && app.cfg.getb("flash_attn", true)) {
        v.push_back("-ctk"); v.push_back(kv);
        v.push_back("-ctv"); v.push_back(kv);
    }
    v.push_back("-n"); v.push_back(std::to_string(app.cfg.geti("bench_tokens", 256)));
    v.push_back("-r"); v.push_back(std::to_string(app.cfg.geti("bench_reps", 3)));
    v.push_back("-p"); v.push_back("0");
    v.insert(v.end(), extra.begin(), extra.end());
    run(v);
}

static void settings_set_kv(const std::string& base, const std::string& key, const std::string& val) {
    std::string path = base + "/settings.txt";
    std::ifstream in(path);
    if (!in.is_open()) { std::cerr << "no settings.txt at " << path << "\n"; return; }
    std::vector<std::string> lines;
    std::string line;
    bool wrote = false;
    while (std::getline(in, line)) {
        std::string t = trim(line);
        if (!t.empty() && t[0] != '#') {
            size_t e = t.find('=');
            if (e != std::string::npos && trim(t.substr(0, e)) == key) {
                lines.push_back(key + " = " + val);
                wrote = true;
                continue;
            }
        }
        lines.push_back(line);
    }
    in.close();
    if (!wrote) lines.push_back(key + " = " + val);
    std::ofstream out(path);
    for (auto& l : lines) out << l << "\n";
}

static void cmd_pull(App& app, const std::string& name, bool set_flag) {
    std::map<std::string, std::string> known = {
        {"qwen3-8b", "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"},
        {"qwen3-8b-q4", "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"},
        {"qwen3-8b-q5", "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q5_K_M.gguf"},
        {"qwen3-8b-q6", "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q6_K.gguf"},
        {"qwen3-8b-q8", "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf"},
        {"qwen3-14b", "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"},
        {"qwen3-14b-q4", "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"},
        {"qwen3-30b", "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"},
        {"qwen3.8-27b", "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf"},
        {"qwen3.8-27b-iq4", "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ4_XS.gguf"},
        {"qwen3.8-27b-iq2s", "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ2_S.gguf"},
        {"qwen3.8-27b-iq1m", "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ1_M.gguf"},
        {"qwen3-next-80b", "https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/Qwen3-Next-80B-A3B-Instruct-UD-IQ2_XXS.gguf"},
        {"qwen3.8-27b-mtp", "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"},
        {"qwen3-0.6b", "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"},
        {"qwen3-0.6b-q4", "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"},
        {"qwen3-1.7b", "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf"},
        {"qwen3-4b", "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf"},
        {"qwen3-4b-q3", "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q3_K_S.gguf"},
        {"qwen3-4b-iq4", "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-IQ4_XS.gguf"},
        {"qwen3-4b-2507", "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-IQ4_XS.gguf"},
        {"qwen3-8b-iq4", "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-IQ4_XS.gguf"},
        // --- non-Qwen families: anything llama.cpp supports works here ---
        {"llama3.1-8b", "https://huggingface.co/unsloth/Llama-3.1-8B-GGUF/resolve/main/Llama-3.1-8B-Q4_K_M.gguf"},
        {"llama3.2-3b", "https://huggingface.co/unsloth/Llama-3.2-3B-GGUF/resolve/main/Llama-3.2-3B-Q4_K_M.gguf"},
        {"gemma3-4b", "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"},
        {"gemma3-12b", "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf"},
        {"gemma3-27b", "https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-Q4_K_M.gguf"},
        {"mistral-7b", "https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3-Q4_K_M.gguf"},
        {"mistral-nemo", "https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"},
        {"phi4", "https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf"},
        {"deepseek-r1-8b", "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"},
        // --- newest Qwen: Qwen4-preview arch (~177B, n-gram embeddings; multi-shard: pull shards 1..3) ---
        {"flash-next-iq1s", "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"},
        {"flash-next-iq1m", "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"},
        {"flash-next-mtp", "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"},
    };
    std::string url;
    std::string out;
    if (name.rfind("http", 0) == 0) {
        url = name;
        size_t s = name.find_last_of('/');
        out = s == std::string::npos ? "model.gguf" : name.substr(s + 1);
        size_t q = out.find('?');
        if (q != std::string::npos) out = out.substr(0, q);
        size_t dot = out.find(".gguf");
        if (dot != std::string::npos) out = out.substr(0, dot);
    } else {
        auto it = known.find(name);
        if (it == known.end()) {
            std::cerr << "unknown model '" << name << "'. known: ";
            for (auto& k : known) std::cerr << k.first << " ";
            std::cerr << "\nor pass a full https://...gguf url\n";
            exit(1);
        }
        url = it->second;
        size_t s = url.find_last_of('/');
        out = url.substr(s + 1);
        size_t dot = out.find(".gguf");
        out = out.substr(0, dot);
    }
    std::string dst = app.base + "/models/" + out + ".gguf";
    if (file_exists(dst)) {
        std::cerr << "already have " << dst << "\n";
        if (set_flag) settings_set_kv(app.base, out.rfind("mtp-", 0) == 0 ? "draft_model" : "model", out);
        return;
    }

    // disk-space guard: model downloads are multi-GB, refuse to fill the disk
    struct statvfs vfs;
    if (statvfs(app.base.c_str(), &vfs) == 0) {
        double free_gb = (double)vfs.f_bavail * vfs.f_frsize / (1024.0 * 1024.0 * 1024.0);
        if (free_gb < 10.0) {
            std::cerr << "only " << (int)free_gb << " GB free on " << app.base
                      << " — models need 7-27 GB. free some space first\n";
            exit(1);
        }
        std::cerr << "(" << (int)free_gb << " GB free on disk)\n";
    }

    // download helper with resume + retry (shared by all shards)
    auto remote_bytes = [](const std::string& u) -> uint64_t {
        std::string cmd = "curl -sIL '" + u + "' 2>/dev/null | tr -d '\\r' | "
                          "awk 'tolower($1)==\"content-length:\" {v=$2} END{print v+0}'";
        FILE* pp = popen(cmd.c_str(), "r");
        if (!pp) return 0;
        unsigned long long v = 0;
        if (fscanf(pp, "%llu", &v) != 1) v = 0;
        pclose(pp);
        return v;
    };
    auto fetch = [&](const std::string& u, const std::string& d) -> bool {
        // refuse to start a download that cannot fit: wasted hours are worse than errors
        uint64_t rb = remote_bytes(u);
        if (rb > 0) {
            struct statvfs v2;
            if (statvfs(app.base.c_str(), &v2) == 0) {
                double free_b = (double)v2.f_bavail * v2.f_frsize;
                if (free_b < (double)rb * 1.05) {
                    std::cerr << "not enough disk: need " << rb / 1024 / 1024 / 1024
                              << " GB, have " << (uint64_t)free_b / 1024 / 1024 / 1024 << " GB\n";
                    exit(1);
                }
            }
        }
        std::cerr << "pulling " << u << "\ninto   " << d << "\n";
        fflush(stderr);
        pid_t pid = fork();
        if (pid == 0) {
            execlp("curl", "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
                   "-C", "-", "-o", d.c_str(), u.c_str(), (char*)nullptr);
            std::cerr << "curl exec failed (is curl installed?)\n";
            _exit(127);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        return WIFEXITED(st) && WEXITSTATUS(st) == 0;
    };

    // llama.cpp multi-shard models: file is X-00001-of-0000M.gguf and llama-server loads
    // the whole model from shard 1 — so pull must fetch ALL M shards.
    int n_shards = 1;
    size_t sh = out.find("-00001-of-");
    if (sh != std::string::npos) n_shards = atoi(out.c_str() + sh + 10);
    if (n_shards > 1) std::cerr << "multi-shard model: " << n_shards << " shards\n";

    for (int i = 1; i <= n_shards; i++) {
        std::string name_i = out, url_i = url;
        if (n_shards > 1) {
            char tail[48];
            snprintf(tail, sizeof(tail), "-0000%d-of-0000%d.gguf", i, n_shards);
            name_i = out.substr(0, sh) + tail;
            url_i = url.substr(0, url.find_last_of('/') + 1) + name_i;
        }
        std::string dst_i = app.base + "/models/" + name_i + ".gguf";
        if (file_exists(dst_i)) {
            std::cerr << "already have shard " << i << "/" << n_shards << "\n";
            continue;
        }
        if (!fetch(url_i, dst_i)) {
            std::cerr << "download failed — partial file kept at " << dst_i
                      << ", re-run pull to resume\n";
            exit(1);
        }
        if (n_shards > 1) std::cerr << "shard " << i << "/" << n_shards << " done\n";
    }
    std::cerr << "done: " << dst << "\n";
    // settings value: for multi-shard models use the clean base name (model_path
    // resolves it to shard 1 automatically)
    std::string settings_name = out;
    size_t sh2 = out.find("-00001-of-");
    if (sh2 != std::string::npos) settings_name = out.substr(0, sh2);
    if (out.rfind("mtp-", 0) == 0) {
        if (set_flag) settings_set_kv(app.base, "draft_model", settings_name);
        std::cerr << "tip: set  spec_type = draft-mtp  in settings.txt\n";
    } else {
        if (set_flag) settings_set_kv(app.base, "model", settings_name);
        else std::cerr << "tip: set  model = " << settings_name << "  in settings.txt (or re-run with --set)\n";
    }
}

static void cmd_quantize(App& app, const std::string& src_name, const std::string& qtype,
                         const std::vector<std::string>& extra) {
    std::string src = app.model_path(src_name);
    if (!file_exists(src)) {
        std::cerr << "source model not found: " << src << "\n";
        exit(1);
    }
    std::string dst = src.substr(0, src.find(".gguf")) + "-" + qtype + ".gguf";
    std::cerr << "quantizing " << src << " -> " << dst << " (" << qtype << ")\n";
    std::vector<std::string> v;
    v.push_back(app.llama_quantize());
    v.insert(v.end(), extra.begin(), extra.end());
    v.push_back(src);
    v.push_back(dst);
    v.push_back(qtype);
    run(v);
}

static void cmd_list(App& app) {
    std::string dir = app.base + "/models";
    std::string cmd = "ls -lh " + dir + " 2>/dev/null | awk 'NR>1 {print $5, $9}'";
    int rc = system(cmd.c_str());
    (void)rc;
}

static void cmd_plan(App& app, const std::string& model_override) {
    std::string m = model_override.empty() ? app.main_model() : app.model_path(model_override);
    if (!file_exists(m)) { std::cerr << "model not found: " << m << "\n"; exit(1); }
    VramPlan p = plan_vram(app, m);
    GgufInfo g = read_gguf(m);
    std::cerr << "model         : " << m << "\n";
    std::cerr << "arch          : " << g.arch << ", layers " << g.n_layer
              << " (attn " << g.n_kv_layers << "), kv-heads " << g.n_head_kv
              << ", head_dim " << g.head_dim
              << ", trained ctx " << g.ctx_train << "\n";
    if (g.is_moe) {
        std::cerr << "moe           : " << g.n_expert_layers << " expert layers, "
                  << (uint64_t)(g.expert_bytes_total / 1024 / 1024) << " MiB expert weights\n";
    }
    std::cerr << "context       : " << app.cfg.geti("context", 40960) << "\n";
    std::cerr << "kv cache      : " << app.cfg.get("kv_cache", "q8_0") << "\n";
    std::cerr << "vram budget   : " << p.allowed_bytes / 1024 / 1024 << " MiB\n";
    std::cerr << "est. usage    : " << p.est_bytes / 1024 / 1024 << " MiB\n";
    std::cerr << "gpu layers    : " << p.ngl << " / " << g.n_layer << "\n";
    if (g.is_moe && p.n_expert_layers > 0) {
        std::cerr << "expert split  : GPU 0.." << p.expert_cpu_from - 1 << ", CPU "
                  << p.expert_cpu_from << ".." << p.n_expert_layers - 1 << "\n";
    }
}

static void usage() {
    std::cerr <<
        "fastollama - tuned llama.cpp runner for AMD gfx1201\n"
        "usage: fastollama <command>\n"
        "  serve              start API server (OpenAI compatible) using settings.txt\n"
        "  chat               interactive terminal chat using settings.txt\n"
        "  ask \"prompt\"       one-shot answer to stdout\n"
        "  bench              run llama-bench with tuned settings\n"
        "  plan               show vram-governor plan for current settings\n"
        "  pull <name>        download model (qwen3.8-27b-iq2s [default], qwen3.8-27b, qwen3.8-27b-iq1m, qwen3-next-80b, qwen3.8-27b-mtp, qwen3-30b, qwen3-8b, qwen3-0.6b, ... or full url); add --set to wire into settings.txt\n"
        "  quantize <src> <q> create a quantized copy (e.g. quantize Qwen3-8B-Q8_0 Q4_0)\n"
        "  list               list downloaded models\n"
        "  stop               kill running llama-server\n"
        "settings: ./settings.txt (key = value, lines with # are comments)\n"
        "extra:    args after -- are passed raw to llama.cpp binary\n";
}

int main(int argc, char** argv) {
    App app;
    app.cfg = load_config();
    app.base = exe_dir();
    if (app.base.size() > 3 && app.base.substr(app.base.size() - 3) == "bin") {
        app.base = app.base.substr(0, app.base.size() - 4);
    }
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    std::vector<std::string> extra;
    bool after_dashdash = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--") { after_dashdash = true; continue; }
        if (after_dashdash) extra.push_back(argv[i]);
    }
    if (cmd == "serve") cmd_serve(app, extra);
    else if (cmd == "chat") cmd_chat(app, extra);
    else if (cmd == "ask") {
        if (argc < 3) { std::cerr << "usage: fastollama ask \"prompt\"\n"; return 1; }
        cmd_ask(app, argv[2], extra);
    }
    else if (cmd == "bench") cmd_bench(app, extra);
    else if (cmd == "plan") cmd_plan(app, argc >= 3 ? argv[2] : "");
    else if (cmd == "pull") {
        if (argc < 3) { std::cerr << "usage: fastollama pull <name> [--set]\n"; return 1; }
        bool set_flag = false;
        for (int i = 3; i < argc; i++) if (std::string(argv[i]) == "--set") set_flag = true;
        cmd_pull(app, argv[2], set_flag);
    }
    else if (cmd == "quantize") {
        if (argc < 4) { std::cerr << "usage: fastollama quantize <model-name> <QTYPE>\n"; return 1; }
        cmd_quantize(app, argv[2], argv[3], extra);
    }
    else if (cmd == "list") cmd_list(app);
    else if (cmd == "stop") {
        int rc = system("pkill -f llama-server && echo stopped || echo nothing running");
        (void)rc;
    }
    else { usage(); return 1; }
    return 0;
}
