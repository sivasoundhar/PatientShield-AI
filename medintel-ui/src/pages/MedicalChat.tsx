import { Link, useParams } from "react-router-dom"
import { FileWarning } from "lucide-react"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ChatPanel } from "@/components/ChatPanel"

/**
 * Standalone full-page Q&A view (CLAUDE.md Day 3 file list). Same
 * ChatPanel as Workspace.tsx's embedded Chat tab — see ChatPanel.tsx for
 * why the behavior is shared — just given the whole page instead of a
 * tab's worth of space, for a focused Q&A session.
 */
export function MedicalChat() {
  const { documentId } = useParams<{ documentId: string }>()

  if (!documentId) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border py-24 text-center">
        <FileWarning className="size-8 text-muted-foreground/60" />
        <p className="text-sm text-muted-foreground">Select a document from the Dashboard to start a chat session.</p>
        <Button asChild variant="outline" size="sm">
          <Link to="/">Go to Dashboard</Link>
        </Button>
      </div>
    )
  }

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">AI Chat</h1>
        <p className="mt-1 text-sm text-muted-foreground">Ask questions grounded in this document's indexed content.</p>
      </div>
      <Card className="flex-1 overflow-hidden">
        <ChatPanel documentId={documentId} />
      </Card>
    </div>
  )
}
