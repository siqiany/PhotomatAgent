import { useState } from "react";

import type { Artifact } from "../../api";

interface Props {
  report: string;
  artifacts: Artifact[];
  taskId: string;
  runId: string | null;
  status: string;
  error: string | null;
}

export function ArtifactPanel({ report, artifacts, taskId, runId, status, error }: Props) {
  const currentArtifacts = artifacts.filter((artifact) => artifact.run_id === runId);
  const attachments = artifacts.filter((artifact) => artifact.run_id === null);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [previewErrors, setPreviewErrors] = useState<Record<string, string>>({});
  const [loadingPreview, setLoadingPreview] = useState<string | null>(null);

  const togglePreview = async (artifact: Artifact) => {
    if (artifact.artifact_id in previews) {
      setPreviews((current) => {
        const next = { ...current };
        delete next[artifact.artifact_id];
        return next;
      });
      return;
    }
    setLoadingPreview(artifact.artifact_id);
    try {
      const response = await fetch(`/api/artifacts/${encodeURIComponent(artifact.artifact_id)}/content`);
      if (!response.ok || !response.body) throw new Error("资料预览读取失败，请下载后查看。");
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let size = 0;
      try {
        while (size < 32 * 1024) {
          const result = await reader.read();
          if (result.done) break;
          const remaining = 32 * 1024 - size;
          const chunk = result.value.slice(0, remaining);
          chunks.push(chunk);
          size += chunk.byteLength;
          if (result.value.byteLength > remaining) {
            await reader.cancel();
            break;
          }
        }
      } finally {
        reader.releaseLock();
      }
      const bytes = new Uint8Array(size);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
      const suffix = size === 32 * 1024 ? "\n\n…（预览限于前 32 KiB）" : "";
      setPreviews((current) => ({ ...current, [artifact.artifact_id]: new TextDecoder().decode(bytes) + suffix }));
      setPreviewErrors((current) => ({ ...current, [artifact.artifact_id]: "" }));
    } catch (reason) {
      setPreviewErrors((current) => ({
        ...current,
        [artifact.artifact_id]: reason instanceof Error ? reason.message : "资料预览读取失败。",
      }));
    } finally {
      setLoadingPreview(null);
    }
  };

  return (
    <aside className="workspace-inspector" data-workbench-panel="details">
      <section className="inspector-card report-card" id="research-report" aria-label="本轮报告" role="region">
        <div className="card-heading">
          <div>
            <span className="eyebrow">RESEARCH OUTPUT</span>
            <h2>本轮报告</h2>
          </div>
          <span className="report-symbol" aria-hidden="true">↗</span>
        </div>
        {error ? <div className="inline-error" role="alert">{error}</div> : null}
        {report ? (
          <div className="report-content" data-testid="research-report-content">{report}</div>
        ) : (
          <div className="report-empty">
            <span className="empty-orbit" aria-hidden="true">◌</span>
            <strong>{status === "FAILED" ? "本轮没有生成报告" : "研究结果将在这里呈现"}</strong>
            <span>运行结束后，报告和可下载产物会保存在当前任务中。</span>
          </div>
        )}
        {report ? (
          <div className="evidence-note">
            <span className="evidence-mark" aria-hidden="true">i</span>
            智能体输出不等于科学验证结论，请结合证据与检查结果判断。
          </div>
        ) : null}
      </section>

      <section className="inspector-card artifact-card" aria-label="本轮产物">
        <div className="card-heading compact-heading">
          <div>
            <span className="eyebrow">FILES & EVIDENCE</span>
            <h2>本轮产物</h2>
          </div>
          <span className="pill-count">{currentArtifacts.length}</span>
        </div>
        {currentArtifacts.length === 0 ? (
          <p className="muted small-copy">当前运行还没有登记文件产物。</p>
        ) : (
          <ul className="artifact-list">
            {currentArtifacts.map((artifact) => (
              <li key={artifact.artifact_id}>
                <span className="file-icon" aria-hidden="true">▤</span>
                <span className="file-copy">
                  <strong>{artifact.filename}</strong>
                  <span>{formatBytes(artifact.size_bytes)} · {artifact.media_type}</span>
                </span>
                <a
                  className="icon-link"
                  href={`/api/artifacts/${encodeURIComponent(artifact.artifact_id)}/content`}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`打开 ${artifact.filename}`}
                >↗</a>
              </li>
            ))}
          </ul>
        )}
        <p className="artifact-scope">任务 {taskId.slice(-8)} · 文件按来源和摘要关联。</p>
      </section>

      <section className="inspector-card attachment-card" aria-label="任务资料">
        <div className="card-heading compact-heading">
          <div>
            <span className="eyebrow">TASK MATERIALS</span>
            <h2>任务资料</h2>
          </div>
          <span className="pill-count">{attachments.length}</span>
        </div>
        {attachments.length === 0 ? (
          <p className="muted small-copy">上传的参考资料会持续关联到此任务。</p>
        ) : (
          <ul className="artifact-list">
            {attachments.map((attachment) => (
              <li key={attachment.artifact_id}>
                <span className="file-icon" aria-hidden="true">▤</span>
                <span className="file-copy">
                  <strong>{attachment.filename}</strong>
                  <span>{formatBytes(attachment.size_bytes)}</span>
                </span>
                <a
                  className="icon-link"
                  href={`/api/artifacts/${encodeURIComponent(attachment.artifact_id)}/content`}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`下载 ${attachment.filename}`}
                >↓</a>
              </li>
            ))}
          </ul>
        )}
        {attachments.map((attachment) => attachment.media_type.startsWith("image/") ? (
          <a
            className="attachment-preview-image"
            href={`/api/artifacts/${encodeURIComponent(attachment.artifact_id)}/content`}
            target="_blank"
            rel="noreferrer"
            key={`preview-${attachment.artifact_id}`}
          >
            <img
              src={`/api/artifacts/${encodeURIComponent(attachment.artifact_id)}/content`}
              alt={`资料预览：${attachment.filename}`}
              loading="lazy"
            />
          </a>
        ) : (
          <div className="text-preview-wrap" key={`preview-${attachment.artifact_id}`}>
            <button
              className="text-button preview-button"
              onClick={() => void togglePreview(attachment)}
              disabled={loadingPreview === attachment.artifact_id}
            >
              {loadingPreview === attachment.artifact_id
                ? "正在读取预览…"
                : attachment.artifact_id in previews ? "收起文字预览" : "预览前 32 KiB"}
            </button>
            {previewErrors[attachment.artifact_id] ? (
              <p className="inline-error preview-error" role="alert">{previewErrors[attachment.artifact_id]}</p>
            ) : null}
            {attachment.artifact_id in previews ? (
              <pre className="text-preview">{previews[attachment.artifact_id]}</pre>
            ) : null}
          </div>
        ))}
      </section>
    </aside>
  );
}

function formatBytes(size: number): string {
  return size < 1024 * 1024 ? `${Math.max(1, Math.round(size / 1024))} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`;
}
