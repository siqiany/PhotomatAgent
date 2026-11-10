import { useEffect, useState, type FormEvent } from "react";
import { api, type ModelSettings, type ModelRole as Role } from "../../api";

const labels: Record<Role, string> = { main: "主智能体", judge: "独立审查 judge", feedback: "反馈编译器" };

export function ModelSettingsPanel({ onClose }: { onClose: () => void }) {
  const [settings, setSettings] = useState<ModelSettings | null>(null);
  const [role, setRole] = useState<Role>("main");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  useEffect(() => { void api.modelSettings().then(setSettings).catch((e: unknown) => setError(e instanceof Error ? e.message : "读取配置失败")); }, []);
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const key = String(data.get("api_key") ?? "").trim();
    setSaving(true); setError(""); setMessage("");
    try {
      const result = await api.saveModelSettings(role, {
        provider: String(data.get("provider")), model: String(data.get("model")).trim(),
        base_url: String(data.get("base_url") ?? "").trim(), ...(key ? { api_key: key } : {}),
      });
      setSettings(result); form.reset();
      setMessage("已保存到工作区 .env，后续运行使用新配置；正在运行的任务保持原配置。");
    } catch (e) { setError(e instanceof Error ? e.message : "保存失败"); }
    finally { setSaving(false); }
  }
  const current = settings?.[role];
  return <div className="modal-backdrop">
    <section className="modal-card" role="dialog" aria-modal="true" aria-labelledby="model-settings-title">
      <div className="modal-topline"><h2 id="model-settings-title">模型配置</h2><button className="icon-button" onClick={onClose} aria-label="关闭模型配置">×</button></div>
      <p className="muted">各角色分别配置 API key，允许使用相同模型。审查意见为辅助信息，科学判定由确定性 checker 完成。</p>
      <label htmlFor="model-role">角色</label>
      <select id="model-role" value={role} disabled={saving} onChange={e => { setRole(e.target.value as Role); setError(""); setMessage(""); }}>
        {Object.entries(labels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select>
      {current && <form key={`${role}-${JSON.stringify(current)}`} onSubmit={e => void save(e)}>
        <label htmlFor="model-provider">提供商</label><select id="model-provider" name="provider" defaultValue={current.provider || "openai"} disabled={saving}>
          <option value="openai">OpenAI / 兼容 API</option><option value="anthropic">Anthropic</option><option value="fake">Fake（本地演示）</option>
        </select>
        <label htmlFor="model-name">模型名称</label><input id="model-name" name="model" required maxLength={200} defaultValue={current.model} disabled={saving} />
        <label htmlFor="model-url">Base URL（留空使用官方服务）</label><input id="model-url" name="base_url" type="url" defaultValue={current.base_url} disabled={saving} />
        <label htmlFor="model-key">API key（{current.key_configured ? "已配置，留空保留" : "尚未配置"}）</label><input id="model-key" name="api_key" type="password" autoComplete="new-password" disabled={saving} />
        <div className="modal-actions"><button className="button button-primary" disabled={saving}>{saving ? "正在保存…" : "保存配置"}</button></div>
      </form>}
      {error && <p role="alert" className="inline-error">{error}</p>}
      {message && <p role="status">{message}</p>}
    </section>
  </div>;
}
