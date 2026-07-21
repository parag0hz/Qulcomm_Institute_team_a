import { FormEvent, useState } from "react";
import { api } from "../api";
import type { DesignParameters } from "../types";

type ChatMessage = { role: "user" | "assistant"; content: string; meta?: string; error?: boolean };

export function CopilotPanel({ design }: { design: DesignParameters }) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "Ask about the current design, sensitivity, confidence, or a target Cd." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const prompts = [
    ["Explain current Cd", "Why is the current Cd high?"],
    ["Review top drivers", "Which three parameters should I review first and why?"],
    ["Target Cd 0.240", "Find a direction toward Cd 0.240 and explain the tradeoffs."],
  ];

  const ask = async (message: string) => {
    if (!message.trim() || busy) return;
    const history = messages.slice(-6).map(({ role, content }) => ({
      role,
      content: content.slice(0, 2000),
    }));
    setMessages((current) => [...current, { role: "user", content: message }]);
    setInput("");
    setBusy(true);
    try {
      const response = await api.copilot({ message, parameters: design, history });
      setMessages((current) => [...current, { role: "assistant", content: response.answer, meta: `${response.provider}${response.model ? ` · ${response.model}` : ""}` }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "assistant", content: error instanceof Error ? error.message : "Copilot request failed.", error: true }]);
    } finally {
      setBusy(false);
    }
  };

  const submit = (event: FormEvent) => { event.preventDefault(); void ask(input); };

  return (
    <section className="copilot-card">
      <div className="copilot-heading"><span className="copilot-mark">AI</span><div><strong>Paragon Design Copilot</strong><small>Grounded design assistant</small></div></div>
      <p className="helper-copy">Cd calculations come from Paragon. The LLM explains evidence and never replaces CFD validation.</p>
      <div className="copilot-prompts">{prompts.map(([label, prompt]) => <button key={label} type="button" onClick={() => void ask(prompt)}>{label}</button>)}</div>
      <div className="copilot-messages">
        {messages.map((message, index) => <div key={`${message.role}-${index}`} className={`copilot-message ${message.role} ${message.error ? "error" : ""}`}><p>{message.content}</p>{message.meta && <small>{message.meta}</small>}</div>)}
        {busy && <div className="copilot-message assistant busy-copy">Analyzing the current design…</div>}
      </div>
      <form className="copilot-form" onSubmit={submit}>
        <textarea rows={3} maxLength={2000} value={input} onChange={(event) => setInput(event.target.value)} placeholder="예: 현재 Cd가 높은 이유와 먼저 바꿀 3개 항목을 알려줘" />
        <button className="primary-button" type="submit" disabled={busy}>{busy ? "Analyzing design…" : "Ask Copilot"}</button>
      </form>
    </section>
  );
}
