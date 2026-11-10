// 最小 HTTP 服务：把"外部请求"接到控制面上（回答"怎么被调用"）。
//
// 不是生产服务器：单线程、一次一个请求、只支持 POST /v1/completions，
// 目的是证明控制面可以被真实客户端驱动，而不是只从 main 里构造 Request。
//
// 用法：
//   ./agentinferkv_http [port]        # 默认 18080
//   curl -s -X POST localhost:18080/v1/completions -d '{"prompt":"hello","max_tokens":8}'
//
// 返回：{"completed":1,"ttft_ms":...,"tokens":...,"text":"..."}
// 注意：这里用 SimulatedExecutor（不跑 GPU），因为它只验证"接入 + 控制面"链路。

#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "ModelConfig.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

using namespace agentinferkv;

namespace
{
    // 从 JSON 里粗暴取字符串字段（够用即可，不引 JSON 库）
    std::string FindString(const std::string& body, const char* key)
    {
        const std::string needle = std::string("\"") + key + "\"";
        auto p = body.find(needle);
        if (p == std::string::npos) return "";
        p = body.find(':', p);
        if (p == std::string::npos) return "";
        p = body.find('"', p);
        if (p == std::string::npos) return "";
        auto q = body.find('"', p + 1);
        if (q == std::string::npos) return "";
        return body.substr(p + 1, q - p - 1);
    }

    int FindInt(const std::string& body, const char* key, int fallback)
    {
        const std::string needle = std::string("\"") + key + "\"";
        auto p = body.find(needle);
        if (p == std::string::npos) return fallback;
        p = body.find(':', p);
        if (p == std::string::npos) return fallback;
        return std::atoi(body.c_str() + p + 1);
    }

    std::string Handle(const std::string& body, int n_ctx)
    {
        const std::string prompt = FindString(body, "prompt");
        const int max_tokens = FindInt(body, "max_tokens", 8);

        ModelConfig config;
        config.block_size = 16;
        config.num_blocks = 64;

        FCFSScheduler sched;
        PagedBlockManager mgr(config);
        PrefixCache cache(&mgr, 0);
        SimulatedExecutor exec;

        // 用模拟执行器：这里验证的是"HTTP → 控制面"链路，不跑 GPU。
        // LlamaCppExecutor 需要模型文件，作为可选项留给 main_real。
        SimulationLoop loop(config, sched, mgr, exec, &cache);

        std::vector<Request> reqs;
        Request r;
        r.id = 1;
        r.arrival_time = 0.0;
        r.max_new_tokens = max_tokens;
        r.prompt_text = prompt;
        // 模拟执行器不看 token 内容，但控制面要用 PromptLen 估块，所以按字符数近似
        r.prompt_tokens.resize(static_cast<std::size_t>(prompt.size() / 4 + 1));
        reqs.push_back(std::move(r));

        auto result = loop.Run(std::move(reqs));
        double ttft = 0.0;
        int tokens = 0;
        bool done = false;
        for (const auto& c : result.completed)
        {
            if (c.state == RequestState::kDone)
            {
                done = true;
                ttft = c.Ttft();
                tokens = c.generated_tokens;
            }
        }

        char out[512];
        std::snprintf(out, sizeof(out),
            "{\"completed\":%d,\"ttft_ms\":%.3f,\"tokens\":%d,\"n_ctx\":%d}\n",
            done ? 1 : 0, ttft * 1000.0, tokens, n_ctx);
        return out;
    }
}

int main(int argc, char** argv)
{
    const int port = (argc > 1) ? std::atoi(argv[1]) : 18080;
    const int n_ctx = 2048;

    const int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { std::perror("socket"); return 1; }

    int one = 1;
    ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        std::perror("bind");
        ::close(listen_fd);
        return 1;
    }
    ::listen(listen_fd, 8);
    std::fprintf(stderr, "[http] listening on 127.0.0.1:%d\n", port);

    for (;;)
    {
        const int client = ::accept(listen_fd, nullptr, nullptr);
        if (client < 0) continue;

        // 读头部直到 \r\n\r\n
        std::string raw;
        char buf[4096];
        while (raw.find("\r\n\r\n") == std::string::npos)
        {
            const ssize_t n = ::read(client, buf, sizeof(buf));
            if (n <= 0) break;
            raw.append(buf, static_cast<std::size_t>(n));
            if (raw.size() > 1 << 20) break;
        }

        const auto head_end = raw.find("\r\n\r\n");
        const std::string head = (head_end == std::string::npos) ? raw : raw.substr(0, head_end);
        std::string body = (head_end == std::string::npos) ? "" : raw.substr(head_end + 4);

        // Content-Length 补齐
        int content_length = 0;
        const auto cl = head.find("Content-Length:");
        if (cl != std::string::npos) content_length = std::atoi(head.c_str() + cl + 15);
        while (static_cast<int>(body.size()) < content_length)
        {
            const ssize_t n = ::read(client, buf, sizeof(buf));
            if (n <= 0) break;
            body.append(buf, static_cast<std::size_t>(n));
        }

        const bool is_post = head.rfind("POST /v1/completions", 0) == 0;
        std::string payload;
        std::string status = "200 OK";
        if (!is_post)
        {
            status = "404 Not Found";
            payload = "{\"error\":\"use POST /v1/completions\"}\n";
        }
        else
        {
            payload = Handle(body, n_ctx);
        }

        const std::string resp =
            "HTTP/1.1 " + status + "\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(payload.size()) + "\r\n"
            "Connection: close\r\n\r\n" + payload;

        (void)!::write(client, resp.data(), resp.size());
        ::close(client);
    }

    ::close(listen_fd);
    return 0;
}
