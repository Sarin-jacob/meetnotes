export interface DeviceInfo {
  model?: string;
  os?: string;
  browser: string;
  ramGB?: number;
  cpuCores: number;
  crossOriginIsolated: boolean;
  webgpu: 'yes' | 'no-api' | 'no-adapter';
  gpu?: string;
  shaderF16: boolean;
  maxBufferMB?: number;
  storageFreeGB?: number;
  wakeLock: boolean;
  vibrate: boolean;
}

interface UAData {
  brands?: Array<{ brand: string; version: string }>;
  getHighEntropyValues?(hints: string[]): Promise<Record<string, string>>;
}

export async function getDeviceInfo(): Promise<DeviceInfo> {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: UAData };
  const brand = nav.userAgentData?.brands?.find((b) => /Chrome|Edge|Samsung|Chromium/i.test(b.brand));
  const info: DeviceInfo = {
    browser: brand ? `${brand.brand} ${brand.version}` : navigator.userAgent,
    ramGB: nav.deviceMemory,
    cpuCores: navigator.hardwareConcurrency,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: 'no-api',
    shaderF16: false,
    wakeLock: 'wakeLock' in navigator,
    vibrate: 'vibrate' in navigator,
  };

  try {
    const he = await nav.userAgentData?.getHighEntropyValues?.(['model', 'platform', 'platformVersion']);
    info.model = he?.model || undefined;
    info.os = he?.platform ? `${he.platform} ${he.platformVersion ?? ''}`.trim() : undefined;
  } catch {
    /* not available */
  }

  try {
    const est = await navigator.storage.estimate();
    if (est.quota) info.storageFreeGB = +(((est.quota - (est.usage ?? 0)) / 1e9).toFixed(1));
  } catch {
    /* not available */
  }

  if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (!adapter) info.webgpu = 'no-adapter';
    else {
      info.webgpu = 'yes';
      const a = adapter.info;
      info.gpu = [a.vendor, a.architecture, a.description].filter(Boolean).join(' · ') || 'unknown';
      info.shaderF16 = adapter.features.has('shader-f16');
      info.maxBufferMB = Math.round(adapter.limits.maxBufferSize / 2 ** 20);
    }
  }
  return info;
}
