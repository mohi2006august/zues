import { useEffect, useRef, useState } from "react";
import { ApiError, api, downloads } from "../api";
import { SourceBadge } from "../components/Badges";
import { Icon, type IconName } from "../components/Icon";
import { addDays } from "../lib/format";
import { load, save } from "../lib/storage";
import { go } from "../router";
import { useApp } from "../state";

type Mode = "upload" | "live" | "emulate";
const MODES: { id: Mode; label: string; icon: IconName }[] = [
  { id: "upload", label: "Upload", icon: "upload" },
  { id: "live", label: "Live NWP", icon: "radio" },
  { id: "emulate", label: "Emulate", icon: "flask" },
];
const CSV_COLUMNS = "block_lgd, issue_date, valid_date, rain_mm, tmax_c, tmin_c, rh_max_pct, rh_min_pct, wind_kmph, wind_dir_deg, cloud_okta";

export default function RunsPage() {
  const { region, regionId, runs, runId, setRunId, refreshRuns } = useApp();
  const [mode, setMode] = useState<Mode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [issue, setIssue] = useState("");
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<{ message: string; details: string[] } | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState(() => load<string>("apiKey", ""));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (region?.data_period && !issue) {
      // Default to a monsoon day inside the data period (emulation needs 5 days of data after it).
      const last = region.data_period[1];
      const candidate = `${last.slice(0, 4)}-07-10`;
      setIssue(candidate < last ? candidate : region.data_period[0]);
    }
  }, [region, issue]);

  // Poll while any run is still being processed.
  useEffect(() => {
    if (!runs.some((r) => r.status === "queued" || r.status === "running")) return;
    const t = window.setInterval(() => refreshRuns(), 1500);
    return () => window.clearInterval(t);
  }, [runs, refreshRuns]);

  const saveKey = (k: string) => {
    setApiKey(k);
    save("apiKey", k);
  };

  const handle = async (fn: () => Promise<{ run_id: string; warnings?: string[] }>) => {
    if (!regionId) return;
    setBusy(true);
    setErrors(null);
    setOk(null);
    try {
      const res = await fn();
      setOk(`Run ${res.run_id.slice(0, 8)} started.${res.warnings?.length ? ` Warnings: ${res.warnings.join("; ")}` : ""}`);
      const poll = async (tries: number): Promise<void> => {
        const r = (await refreshRuns()).find((x) => x.run_id === res.run_id);
        if (r?.status === "done") {
          setRunId(res.run_id);
          setOk(`Run ${res.run_id.slice(0, 8)} is ready and is now the selected bulletin.`);
        } else if (r?.status === "failed") {
          setErrors({ message: "The run failed", details: [r.notes ?? ""] });
        } else if (tries > 0) {
          await new Promise((done) => setTimeout(done, 1200));
          return poll(tries - 1);
        }
      };
      await poll(40);
    } catch (e) {
      const err = e as ApiError;
      setErrors({ message: err.message, details: err.details ?? [] });
    } finally {
      setBusy(false);
    }
  };

  if (!region) return <div className="empty"><span className="spinner" /></div>;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Forecast runs</h1>
          <p>Each run takes a 5-day block forecast, downscales every variable to every panchayat and drafts the advisories.</p>
        </div>
      </div>

      <div className="runs-grid">
        <div className="card">
          <div className="card-head"><h2>New run</h2></div>
          <div className="card-pad stack" style={{ gap: 14 }}>
            <div className="seg full" role="tablist" aria-label="Forecast source">
              {MODES.map((m) => (
                <button key={m.id} role="tab" aria-selected={mode === m.id} className={mode === m.id ? "on" : ""} onClick={() => setMode(m.id)}>
                  <span className="row" style={{ gap: 6, justifyContent: "center" }}><Icon name={m.icon} size={14} />{m.label}</span>
                </button>
              ))}
            </div>

            {mode === "upload" && (
              <>
                <p className="small muted">The official IMD block forecast as CSV, one row per block and day.</p>
                <div
                  className={`dropzone ${over ? "over" : ""}`}
                  onClick={() => inputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setOver(true); }}
                  onDragLeave={() => setOver(false)}
                  onDrop={(e) => { e.preventDefault(); setOver(false); setFile(e.dataTransfer.files[0] ?? null); }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
                >
                  <Icon name={file ? "file" : "upload"} size={22} />
                  <b>{file ? file.name : "Drop a CSV here or choose a file"}</b>
                  <span className="small">{file ? `${(file.size / 1024).toFixed(1)} KB` : "Columns: see below"}</span>
                  <input ref={inputRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
                </div>
                <p className="mono faint" style={{ fontSize: 11, lineHeight: 1.6 }}>{CSV_COLUMNS}</p>
                <button className="btn primary block" disabled={!file || busy} onClick={() => file && handle(() => api.upload(region.region_id, file))}>
                  {busy ? <span className="spinner" /> : "Downscale this forecast"}
                </button>
              </>
            )}

            {mode === "live" && (
              <>
                <p className="small muted">
                  Fetches today's 5-day numerical weather forecast (Open-Meteo: GFS / ECMWF) at every block centroid and downscales it.
                  Useful when no IMD bulletin is at hand. Needs internet on the server.
                </p>
                <button className="btn primary block" disabled={busy} onClick={() => handle(() => api.fetchLive(region.region_id))}>
                  {busy ? <span className="spinner" /> : <><Icon name="radio" size={15} /> Fetch live forecast</>}
                </button>
              </>
            )}

            {mode === "emulate" && (
              <>
                <p className="small muted">
                  Builds a block forecast from this district's historical data plus realistic forecast error. For demos and testing only.
                </p>
                <label className="field">
                  <span className="label">Issue date</span>
                  <input className="input" type="date" value={issue} min={region.data_period?.[0]}
                    max={region.data_period ? addDays(region.data_period[1], -5) : undefined} onChange={(e) => setIssue(e.target.value)} />
                  {region.data_period && (
                    <span className="small faint">History available {region.data_period[0]} to {region.data_period[1]}</span>
                  )}
                </label>
                <button className="btn primary block" disabled={!issue || busy} onClick={() => handle(() => api.emulate(region.region_id, issue))}>
                  {busy ? <span className="spinner" /> : "Create emulated run"}
                </button>
              </>
            )}

            {errors && (
              <div className="notice error">
                <Icon name="alert" />
                <div>
                  <b>{errors.message}</b>
                  {errors.details.length > 0 && <ul>{errors.details.map((d) => <li key={d}>{d}</li>)}</ul>}
                </div>
              </div>
            )}
            {ok && <div className="notice ok"><Icon name="check" /><span>{ok}</span></div>}

            <details className="small">
              <summary className="muted" style={{ cursor: "pointer" }}>Server API key</summary>
              <p className="faint" style={{ margin: "6px 0" }}>Only needed if the server sets <span className="mono">PCAST_API_KEY</span>. Stored in this browser.</p>
              <input className="input" style={{ width: "100%" }} type="password" value={apiKey} onChange={(e) => saveKey(e.target.value)} placeholder="X-API-Key" autoComplete="off" />
            </details>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <h2>History</h2>
            <span className="small muted">{runs.length} runs</span>
          </div>
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr><th>Issued</th><th>Source</th><th>Model</th><th>Status</th><th>Exports</th><th /></tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} className={r.run_id === runId ? "current" : ""}>
                    <td>
                      <b className="num" style={{ fontWeight: 600 }}>{new Date(`${r.issue_date}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}</b>
                      <div className="mono faint" style={{ fontSize: 11 }}>{r.run_id.slice(0, 8)}</div>
                    </td>
                    <td><SourceBadge source={r.source} /></td>
                    <td className="small">{r.model_id}<div className="faint mono" style={{ fontSize: 11 }}>{r.model_version}</div></td>
                    <td>
                      <span className={`status ${r.status}`} title={r.notes ?? undefined}>{r.status}</span>
                      {r.advisory_counts && r.status === "done" && (
                        <div className="small faint num">{Object.values(r.advisory_counts).reduce((a, b) => a + (b ?? 0), 0)} advisories</div>
                      )}
                    </td>
                    <td>
                      {r.status === "done" && (
                        <div className="dl">
                          <a href={downloads.csv(r.run_id)}>CSV</a>
                          <a href={downloads.geojson(r.run_id)}>GeoJSON</a>
                          <a href={downloads.pdf(r.run_id)} target="_blank" rel="noreferrer">PDF</a>
                          <a href={downloads.sms(r.run_id)}>SMS</a>
                        </div>
                      )}
                    </td>
                    <td className="r">
                      {r.status === "done" && (r.run_id === runId ? (
                        <span className="chip accent">Selected</span>
                      ) : (
                        <button className="btn sm" onClick={() => { setRunId(r.run_id); go("/"); }}>Open <Icon name="arrow" size={13} /></button>
                      ))}
                    </td>
                  </tr>
                ))}
                {!runs.length && <tr><td colSpan={6}><div className="empty" style={{ padding: 32 }}>No runs yet. Create one on the left.</div></td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
