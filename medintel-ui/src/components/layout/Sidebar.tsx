import { NavLink } from "react-router-dom"
import { ShieldCheck, LayoutDashboard, Activity, FolderOpen, MessageSquare, ListChecks } from "lucide-react"

import { useCurrentDocument } from "@/context/DocumentContext"
import { cn } from "@/lib/utils"

/**
 * Persistent left-hand navigation, present on every page (CLAUDE.md section
 * 16 Day 3: "sidebar with links to all 5 pages"). Processing/Workspace/Chat
 * links follow the "current" document from context so a returning user
 * lands where they left off; falls back to a route with no id, which each
 * of those pages renders as a friendly empty state (see their own files)
 * rather than crashing on a missing param.
 */
const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true, needsDoc: false },
  { to: "/processing", label: "Processing", icon: Activity, end: false, needsDoc: true },
  { to: "/workspace", label: "Workspace", icon: FolderOpen, end: false, needsDoc: true },
  { to: "/chat", label: "AI Chat", icon: MessageSquare, end: false, needsDoc: true },
  { to: "/audit", label: "Audit Trail", icon: ListChecks, end: false, needsDoc: false },
] as const

export function Sidebar() {
  const { currentDocumentId } = useCurrentDocument()

  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2.5 border-b border-sidebar-border px-5 py-5">
        <div className="flex size-9 items-center justify-center rounded-lg bg-primary/20 text-primary">
          <ShieldCheck className="size-5" />
        </div>
        <div>
          <p className="text-sm font-semibold text-white">PatientShield AI</p>
          <p className="text-xs text-sidebar-muted">Clinical Document Intelligence</p>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-1 p-3">
        {navItems.map(({ to, label, icon: Icon, end, needsDoc }) => {
          const href = needsDoc && currentDocumentId ? `${to}/${currentDocumentId}` : to
          return (
            <NavLink
              key={to}
              to={href}
              end={end}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-active-bg text-sidebar-active-fg"
                    : "text-sidebar-foreground hover:bg-sidebar-hover-bg",
                )
              }
            >
              <Icon className="size-4 shrink-0" />
              {label}
            </NavLink>
          )
        })}
      </nav>

      <div className="border-t border-sidebar-border p-4">
        <div className="rounded-lg border border-sidebar-border bg-sidebar-hover-bg p-3">
          <p className="text-xs font-semibold text-white">HIPAA-Aware by Design</p>
          <p className="mt-1 text-[11px] leading-relaxed text-sidebar-muted">
            PHI is detected and masked before anything reaches the knowledge index. Every agent decision is logged.
          </p>
        </div>
      </div>
    </aside>
  )
}
