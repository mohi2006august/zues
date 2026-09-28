import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "./api";
import { load, save } from "./lib/storage";
import type { FeatureCollection, MapProps, RegionDetail, RegionSummary, Run } from "./types";

interface AppState {
  regions: RegionSummary[];
  regionId: string | null;
  region: RegionDetail | null;
  runs: Run[];
  runId: string | null;
  run: Run | null;
  blocks: FeatureCollection<MapProps> | null;
  gpNames: Map<number, string>;
  blockNames: Map<number, string>;
  error: string | null;
  setRegionId: (id: string) => void;
  setRunId: (id: string) => void;
  refreshRuns: () => Promise<Run[]>;
  dark: boolean;
  toggleTheme: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside provider");
  return v;
}

function prefersDark(): boolean {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    return false;
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [regions, setRegions] = useState<RegionSummary[]>([]);
  const [regionId, setRegionIdState] = useState<string | null>(null);
  const [region, setRegion] = useState<RegionDetail | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunIdState] = useState<string | null>(null);
  const [blocks, setBlocks] = useState<FeatureCollection<MapProps> | null>(null);
  const [gpNames, setGpNames] = useState<Map<number, string>>(new Map());
  const [error, setError] = useState<string | null>(null);
  const [dark, setDark] = useState<boolean>(() => load<string | null>("theme", null) === "dark" || (load<string | null>("theme", null) === null && prefersDark()));

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }, [dark]);

  const toggleTheme = useCallback(() => {
    setDark((d) => {
      save("theme", d ? "light" : "dark");
      return !d;
    });
  }, []);

  useEffect(() => {
    api
      .regions()
      .then((rs) => {
        setRegions(rs);
        const saved = load<string | null>("region", null);
        // First visit: prefer a real district over the synthetic demo.
        const pick = rs.find((r) => r.region_id === saved) ?? rs.find((r) => r.data_source === "real") ?? rs[0];
        setRegionIdState(pick ? pick.region_id : null);
        if (!rs.length) setError("No regions are built yet. Run `pcast demo` on the server.");
      })
      .catch((e) => setError(`Cannot reach the API: ${e.message}. Is \`pcast serve\` running?`));
  }, []);

  const refreshRuns = useCallback(async () => {
    if (!regionId) return [];
    const rs = await api.runs(regionId);
    setRuns(rs);
    return rs;
  }, [regionId]);

  useEffect(() => {
    if (!regionId) return;
    save("region", regionId);
    setRegion(null);
    setRuns([]);
    setBlocks(null);
    api.region(regionId).then(setRegion).catch((e) => setError(e.message));
    api.blocks(regionId).then(setBlocks).catch(() => undefined);
    api
      .panchayats(regionId)
      .then((fc) => setGpNames(new Map(fc.features.map((f) => [f.properties.gp_lgd as number, f.properties.gp_name as string]))))
      .catch(() => undefined);
    refreshRuns().then((rs) => {
      const saved = load<string | null>(`run.${regionId}`, null);
      const done = rs.filter((r) => r.status === "done");
      const pick = done.find((r) => r.run_id === saved) ?? done[0];
      setRunIdState(pick ? pick.run_id : null);
    });
  }, [regionId, refreshRuns]);

  const setRegionId = useCallback((id: string) => setRegionIdState(id), []);
  const setRunId = useCallback(
    (id: string) => {
      setRunIdState(id);
      if (regionId) save(`run.${regionId}`, id);
    },
    [regionId],
  );

  const blockNames = useMemo(
    () => new Map((blocks?.features ?? []).map((f) => [f.properties.block_lgd, f.properties.block_name])),
    [blocks],
  );
  const run = runs.find((r) => r.run_id === runId) ?? null;

  const value: AppState = {
    regions, regionId, region, runs, runId, run, blocks, gpNames, blockNames, error,
    setRegionId, setRunId, refreshRuns, dark, toggleTheme,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
