import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "../../api";
import { ModelSettingsPanel } from "./ModelSettingsPanel";

const settings = {
  main: { provider: "openai", model: "main-model", base_url: "", configured: true, key_configured: true },
  judge: { provider: "openai", model: "main-model", base_url: "", configured: true, key_configured: true },
  feedback: { provider: "", model: "", base_url: "", configured: false, key_configured: false },
};
afterEach(() => { vi.restoreAllMocks(); document.body.innerHTML = ""; });

it("sends a replacement key only on save and clears it after saving", async () => {
  vi.spyOn(api, "modelSettings").mockResolvedValue(settings);
  const save = vi.spyOn(api, "saveModelSettings").mockResolvedValue(settings);
  const container = document.createElement("div"); document.body.append(container);
  const root = createRoot(container);
  await act(async () => root.render(<ModelSettingsPanel onClose={() => {}} />));
  const keyInput = container.querySelector<HTMLInputElement>("#model-key")!;
  expect(keyInput.type).toBe("password"); expect(keyInput.value).toBe("");
  keyInput.value = "replacement-key";
  await act(async () => container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(save).toHaveBeenCalledWith("main", expect.objectContaining({ api_key: "replacement-key" }));
  expect(container.querySelector<HTMLInputElement>("#model-key")!.value).toBe("");
  expect(container.textContent).toContain("已保存到工作区 .env");
  await act(async () => root.unmount());
});

it("retains an existing key when the password input is blank", async () => {
  vi.spyOn(api, "modelSettings").mockResolvedValue(settings);
  const save = vi.spyOn(api, "saveModelSettings").mockResolvedValue(settings);
  const container = document.createElement("div"); document.body.append(container);
  const root = createRoot(container);
  await act(async () => root.render(<ModelSettingsPanel onClose={() => {}} />));
  await act(async () => container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(save.mock.calls[0][1]).not.toHaveProperty("api_key");
  await act(async () => root.unmount());
});
