"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  CheckCircle2,
  Database,
  FileSearch,
  GitBranch,
  Layers,
  MessageSquare,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { checkBackendHealth } from "@/lib/api";

const navItems = [
  { href: "/", label: "Overview", icon: Layers },
  { href: "/playground", label: "QA Playground", icon: MessageSquare },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/retrieval", label: "Retrieval", icon: GitBranch },
  { href: "/analytics", label: "Latency & Resources", icon: Activity },
  { href: "/routing", label: "Adaptive Routing", icon: FileSearch },
  { href: "/citations", label: "Citations", icon: ShieldCheck },
  { href: "/reproducibility", label: "Reproducibility", icon: Terminal },
];

export function Navbar() {
  const pathname = usePathname();
  const [backendStatus, setBackendStatus] = useState<{
    ok: boolean;
    message: string;
  }>({
    ok: false,
    message: "Checking FastAPI backend...",
  });

  useEffect(() => {
    checkBackendHealth().then(setBackendStatus);
    const interval = setInterval(() => {
      checkBackendHealth().then(setBackendStatus);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-neutral-800 bg-neutral-950/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-emerald-500 font-bold text-white shadow-md shadow-indigo-500/20 group-hover:scale-105 transition-transform">
              RS
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold tracking-tight text-white sm:text-lg">
                  RAGScaleBench
                </span>
                <span className="rounded-full bg-emerald-950/80 px-2 py-0.5 text-[10px] font-semibold text-emerald-400 border border-emerald-800/60">
                  v1.0 Validated
                </span>
              </div>
              <p className="text-[11px] text-neutral-400">
                Rigorous Systems Evaluation & Adaptive RAG
              </p>
            </div>
          </Link>
        </div>

        {/* Desktop Navigation Links */}
        <nav className="hidden xl:flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive =
              pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  isActive
                    ? "bg-neutral-800 text-white font-semibold shadow-inner"
                    : "text-neutral-300 hover:bg-neutral-900 hover:text-white"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Live Backend Telemetry Pill */}
        <div className="flex items-center gap-2">
          <div
            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium border ${
              backendStatus.ok
                ? "bg-emerald-950/50 text-emerald-400 border-emerald-800/60"
                : "bg-amber-950/50 text-amber-400 border-amber-800/60"
            }`}
            title={backendStatus.message}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendStatus.ok
                  ? "bg-emerald-400 animate-pulse"
                  : "bg-amber-400"
              }`}
            />
            <span className="hidden sm:inline">
              {backendStatus.ok ? "FastAPI :8000 Live" : "FastAPI Local Mode"}
            </span>
          </div>
        </div>
      </div>

      {/* Sub-nav for tablets / smaller screens */}
      <div className="xl:hidden border-t border-neutral-900 overflow-x-auto px-4 py-2 flex items-center gap-1 scrollbar-none">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex shrink-0 items-center gap-1.5 px-2.5 py-1 rounded text-xs transition-colors ${
                isActive
                  ? "bg-neutral-800 text-white font-semibold"
                  : "text-neutral-400 hover:bg-neutral-900 hover:text-white"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {item.label}
            </Link>
          );
        })}
      </div>
    </header>
  );
}
