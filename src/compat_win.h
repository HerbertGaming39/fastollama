// compat_win.h — Windows shims so fastollama.cpp compiles and runs unmodified.
// Strategy: keep every Linux path intact under #ifndef _WIN32 and provide
// Windows equivalents here (DXGI for VRAM, GlobalMemoryStatusEx for RAM,
// CreateProcess for fork/exec). The governors behave identically.
#pragma once

#ifdef _MSC_VER
#pragma comment(lib, "dxgi")
#pragma comment(lib, "user32")
#endif

#ifdef _WIN32

#include <windows.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#define popen  _popen
#define pclose _pclose

static inline int setenv(const char* k, const char* v, int) { return _putenv_s(k, v); }

// ---- VRAM via DXGI (largest non-software adapter; used = physical budget) ----
static inline int dxgi_adapter_count() {
    static int cached = -1;
    if (cached >= 0) return cached;
    cached = 0;
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory))) return 0;
    UINT i = 0; IDXGIAdapter1* ad = nullptr;
    while (factory->EnumAdapters1(i++, &ad) != DXGI_ERROR_NOT_FOUND) { ad->Release(); cached++; }
    factory->Release();
    return cached;
}

// adapter with the largest dedicated VRAM (skips software/WARP)
static inline int dxgi_best_adapter() {
    int best = -1; UINT64 best_mem = 0;
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory))) return -1;
    UINT i = 0; IDXGIAdapter1* ad = nullptr;
    while (factory->EnumAdapters1(i++, &ad) == S_OK) {
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(ad->GetDesc1(&d)) &&
            !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            d.DedicatedVideoMemory > best_mem) {
            best_mem = d.DedicatedVideoMemory; best = (int)(i - 1);
        }
        ad->Release();
    }
    factory->Release();
    return best;
}

static inline IDXGIAdapter1* dxgi_get(int idx) {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&factory))) return nullptr;
    IDXGIAdapter1* ad = nullptr;
    factory->EnumAdapters1((UINT)idx, &ad);
    factory->Release();
    return ad; // caller releases
}

static inline uint64_t amd_vram_total_dxgi() {
    int c = dxgi_best_adapter();
    if (c < 0) return 0;
    IDXGIAdapter1* ad = dxgi_get(c);
    if (!ad) return 0;
    DXGI_ADAPTER_DESC1 d{};
    UINT64 total = SUCCEEDED(ad->GetDesc1(&d)) ? d.DedicatedVideoMemory : 0;
    ad->Release();
    return total;
}

static inline uint64_t amd_vram_used_dxgi() {
    int c = dxgi_best_adapter();
    if (c < 0) return 0;
    IDXGIAdapter1* ad = dxgi_get(c);
    if (!ad) return 0;
    UINT64 used = 0;
    IDXGIAdapter3* a3 = nullptr;
    if (SUCCEEDED(ad->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        // LOCAL segment = the adapter's dedicated VRAM on discrete cards
        if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            used = info.CurrentUsage;
        a3->Release();
    }
    ad->Release();
    return used;
}

// ---- RAM ----
static inline uint64_t win_ram_available_bytes() {
    MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) return 0;
    return (uint64_t)m.ullAvailPhys;
}
static inline uint64_t win_ram_total_bytes() {
    MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) return 0;
    return (uint64_t)m.ullTotalPhys;
}

// ---- disk ----
static inline bool win_disk_free_bytes(const char* path, uint64_t* out) {
    ULARGE_INTEGER q{};
    if (!GetDiskFreeSpaceExA(path, &q, nullptr, nullptr)) return false;
    *out = q.QuadPart;
    return true;
}

// ---- process helpers ----
struct WinProc { HANDLE h = nullptr; DWORD pid = 0; };

static inline std::string win_quote(const std::string& s) {
    if (s.find(' ') == std::string::npos && s.find('\t') == std::string::npos) return s;
    std::string r = "\"";
    for (char ch : s) { if (ch == '"') r += "\\\""; else r += ch; }
    return r + "\"";
}

// spawn without waiting (returns handle); caller closes it
static inline WinProc win_spawn(const std::vector<std::string>& args) {
    std::string cmd;
    for (size_t i = 0; i < args.size(); i++) {
        if (i) cmd += " ";
        cmd += win_quote(args[i]);
    }
    STARTUPINFOA si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    WinProc wp;
    if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0,
                       nullptr, nullptr, &si, &pi)) {
        wp.h = pi.hProcess; wp.pid = pi.dwProcessId;
        CloseHandle(pi.hThread);
    }
    return wp;
}

// spawn + wait; returns exit code (127 = spawn failed)
static inline int win_run(const std::vector<std::string>& args) {
    WinProc wp = win_spawn(args);
    if (!wp.h) return 127;
    DWORD code = 127;
    WaitForSingleObject(wp.h, INFINITE);
    GetExitCodeProcess(wp.h, &code);
    CloseHandle(wp.h);
    return (int)code;
}

#endif // _WIN32
