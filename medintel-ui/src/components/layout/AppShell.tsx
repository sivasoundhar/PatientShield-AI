import { Outlet } from "react-router-dom"

import { Sidebar } from "@/components/layout/Sidebar"

/** Wraps every routed page with the persistent sidebar (CLAUDE.md Day 3: React Router + sidebar nav). */
export function AppShell() {
  return (
    <div className="flex min-h-screen bg-muted/40">
      <Sidebar />
      <main className="min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl px-8 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
