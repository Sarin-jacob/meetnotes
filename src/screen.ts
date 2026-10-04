import { formatTime } from './audio.ts';

/**
 * "Dark recording screen": a full-screen black overlay plus a screen wake lock.
 * On OLED phones black pixels are off, so this is nearly as good as dimming,
 * and keeping the page visible stops Chrome from suspending the recording.
 */
export class DarkScreen {
  private lock: WakeLockSentinel | null = null;
  private timer = 0;
  private lastTap = 0;
  private overlay = document.getElementById('dark-overlay')!;
  private clock = document.getElementById('dark-time')!;
  private onTap = () => {
    const now = performance.now();
    if (now - this.lastTap < 400) void this.exit();
    this.lastTap = now;
  };
  private onVisible = async () => {
    // Wake locks are released when the page is hidden; re-acquire on return.
    if (!document.hidden && !this.overlay.hidden) await this.acquire();
  };
  onExit?: () => void;

  get wakeLockActive(): boolean {
    return !!this.lock && !this.lock.released;
  }

  async enter(elapsed: () => number): Promise<boolean> {
    this.overlay.hidden = false;
    this.overlay.addEventListener('pointerup', this.onTap);
    document.addEventListener('visibilitychange', this.onVisible);
    this.timer = window.setInterval(() => (this.clock.textContent = formatTime(elapsed())), 500);
    try {
      await document.documentElement.requestFullscreen?.({ navigationUI: 'hide' });
    } catch {
      /* fullscreen is optional */
    }
    return this.acquire();
  }

  async exit(): Promise<void> {
    this.overlay.hidden = true;
    this.overlay.removeEventListener('pointerup', this.onTap);
    document.removeEventListener('visibilitychange', this.onVisible);
    clearInterval(this.timer);
    await this.lock?.release().catch(() => {});
    this.lock = null;
    if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
    this.onExit?.();
  }

  private async acquire(): Promise<boolean> {
    try {
      this.lock = await navigator.wakeLock.request('screen');
      return true;
    } catch {
      return false;
    }
  }
}

/** Loud two-tone alarm + vibration, used to warn that the 1-hour handover is due. */
export async function playHandoverAlarm(seconds = 4): Promise<{ vibrated: boolean }> {
  const ctx = new AudioContext();
  const gain = ctx.createGain();
  gain.gain.value = 0.9;
  gain.connect(ctx.destination);
  const beep = 0.25;
  for (let i = 0; i < seconds / (beep * 2); i++) {
    const osc = ctx.createOscillator();
    osc.type = 'square';
    osc.frequency.value = i % 2 ? 880 : 1320;
    osc.connect(gain);
    osc.start(ctx.currentTime + i * beep * 2);
    osc.stop(ctx.currentTime + i * beep * 2 + beep);
  }
  const vibrated = navigator.vibrate?.([400, 200, 400, 200, 800]) ?? false;
  setTimeout(() => void ctx.close(), seconds * 1000 + 200);
  return { vibrated };
}
