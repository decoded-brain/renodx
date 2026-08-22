/*
 * Copyright (C) 2026 Carlos Lopez
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <string>
#include <thread>

#include "src/utils/ipc/ipc.hpp"

namespace {

bool StopsPromptly(bool connect_client) {
  renodx::utils::ipc::Server server;
  const auto pipe_name = std::wstring(L"renodx-ipc-stop-test-")
                         + std::to_wstring(GetCurrentProcessId())
                         + (connect_client ? L"-connected" : L"-listener");
  if (!server.Start(
          renodx::utils::ipc::ServerConfig{
              .pipe_name = pipe_name,
              .max_instances = 4,
          },
          [](const renodx::utils::ipc::Message&, renodx::utils::ipc::Server&) {})) {
    return false;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  renodx::utils::ipc::Client client;
  if (connect_client && !client.Connect(pipe_name, 1000)) return false;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  const auto started = std::chrono::steady_clock::now();
  server.Stop();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  client.Disconnect();
  return elapsed < std::chrono::seconds(2);
}

}  // namespace

int main() {
  if (!StopsPromptly(false)) return 1;
  if (!StopsPromptly(true)) return 2;
  return 0;
}
