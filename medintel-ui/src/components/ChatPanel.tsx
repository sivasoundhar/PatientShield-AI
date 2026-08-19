import { useState, useRef, useEffect } from "react"
import { Send, Bot, User, Loader2 } from "lucide-react"

import { api, getErrorMessage } from "@/services/api"
import { formatScore } from "@/lib/format"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface ChatMessage {
  role: "user" | "assistant"
  text: string
  sourceCitation?: string | null
  confidence?: number
  foundInDocument?: boolean
}

/**
 * The Q&A chat UI, shared between Workspace.tsx's "Chat" tab (embedded,
 * compact) and the standalone MedicalChat.tsx page (full page). One
 * implementation avoids the two drifting out of sync — CLAUDE.md lists them
 * as separate files, but the chat *behavior* (ask, render citation,
 * loading/error states) is identical between an embedded panel and a full
 * page, so only the surrounding layout differs per call site.
 */
export function ChatPanel({ documentId }: { documentId: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState("")
  const [isAsking, setIsAsking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault()
    const q = question.trim()
    if (!q || isAsking) return

    setMessages((prev) => [...prev, { role: "user", text: q }])
    setQuestion("")
    setIsAsking(true)
    setError(null)

    try {
      const response = await api.askQuestion(documentId, q)
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: response.answer,
          sourceCitation: response.source_citation,
          confidence: response.confidence,
          foundInDocument: response.found_in_document,
        },
      ])
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="py-8 text-center text-sm text-muted-foreground">
            Ask a question about this document to get started.
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div
              className={`flex size-7 shrink-0 items-center justify-center rounded-full ${
                msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-accent text-primary"
              }`}
            >
              {msg.role === "user" ? <User className="size-4" /> : <Bot className="size-4" />}
            </div>
            <div className={`max-w-[80%] rounded-xl px-3.5 py-2.5 text-sm ${
              msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
            }`}>
              <p>{msg.text}</p>
              {msg.role === "assistant" && (
                <div className="mt-2 flex flex-wrap items-center gap-2 text-xs opacity-80">
                  {msg.sourceCitation && <span>Source: {msg.sourceCitation}</span>}
                  {msg.confidence !== undefined && <span>Confidence: {formatScore(msg.confidence)}</span>}
                  {msg.foundInDocument === false && <span className="text-status-warning">Not found in document</span>}
                </div>
              )}
            </div>
          </div>
        ))}
        {isAsking && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Thinking…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <p className="px-4 pb-1 text-xs text-status-danger">{error}</p>}

      <form onSubmit={handleAsk} className="flex items-center gap-2 border-t border-border p-3">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a medical question about this document…"
          disabled={isAsking}
        />
        <Button type="submit" size="icon" disabled={isAsking || !question.trim()}>
          <Send className="size-4" />
        </Button>
      </form>
      <p className="px-4 pb-3 text-[11px] text-muted-foreground">AI can make mistakes. Please verify important information.</p>
    </div>
  )
}
