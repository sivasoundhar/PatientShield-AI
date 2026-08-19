import { BrowserRouter, Routes, Route } from "react-router-dom"

import { DocumentProvider } from "@/context/DocumentContext"
import { AppShell } from "@/components/layout/AppShell"
import { Dashboard } from "@/pages/Dashboard"
import { Processing } from "@/pages/Processing"
import { Workspace } from "@/pages/Workspace"
import { MedicalChat } from "@/pages/MedicalChat"
import { AuditTrail } from "@/pages/AuditTrail"

/**
 * Routing per CLAUDE.md Day 3: React Router, sidebar nav to all 5 pages.
 * Processing/Workspace/Chat/Audit all take an optional :documentId — each
 * page renders its own empty state when it's missing rather than crashing,
 * since there's no single "current document" the router can assume.
 */
export default function App() {
  return (
    <DocumentProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/processing" element={<Processing />} />
            <Route path="/processing/:documentId" element={<Processing />} />
            <Route path="/workspace" element={<Workspace />} />
            <Route path="/workspace/:documentId" element={<Workspace />} />
            <Route path="/chat" element={<MedicalChat />} />
            <Route path="/chat/:documentId" element={<MedicalChat />} />
            <Route path="/audit" element={<AuditTrail />} />
            <Route path="/audit/:documentId" element={<AuditTrail />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </DocumentProvider>
  )
}
