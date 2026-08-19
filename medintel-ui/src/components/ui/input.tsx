import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "flex h-9 w-full min-w-0 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm outline-none transition-[color,box-shadow] placeholder:text-muted-foreground disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50",
        "focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30",
        className,
      )}
      {...props}
    />
  )
}

export { Input }
