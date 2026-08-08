// redline-conscious-loop — LEAN (rewritten after bloated fan-out failed ROI)
//
// Reality check: parent agent ships P12b code + rebuild + A/B faster than
// 14-agent domain theater. This file is now a thin optional assist, not a
// second bureaucracy.
//
// Max agents per run: 2 (or 1).
//   mode=fire  → 1 senior-dev does one concrete slice + docs
//   mode=qr    → 1 quality-reviewer checks uncommitted / named path only
//
// Args:
//   mode: "fire" | "qr"     default fire
//   focus: string           required-ish for fire (what to do this slice)
//
// Hard bans still apply. Clear Thought MCP once per agent, not five panels.
//

let meta = #{
    name: "redline-lean",
    description: "LEAN Redline slice: one implementer (or one QR). No domain theater.",
    when_to_use: "Optional assist for one Redline code/measure slice — prefer parent agent for speed",
    phases: [
        #{ title: "Fire", detail: "one agent does the work" },
    ],
};

let fire_schema = #{
    "type": "object",
    "required": ["ok", "summary", "changed", "next", "clear_thought_used"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "changed": #{ "type": "boolean" },
        "files": #{ "type": "array", "items": #{ "type": "string" } },
        "next": #{ "type": "string" },
        "clear_thought_used": #{ "type": "boolean" },
    },
};

let mode = "fire";
let focus = "P12c: profile OWN_RMSNORM host path; cut pre-stream-sync redundancy if safe";
if args != () {
    if args.mode != () { mode = args.mode; }
    if args.focus != () { focus = args.focus; }
}
if mode != "fire" && mode != "qr" {
    mode = "fire";
}

log("LEAN redline loop mode=" + mode + " focus=" + focus);

phase("Fire");

let p = "";
p += "Repo: /home/antmi/lemon-mlx-engine branch exp/redline-kernel-launch.\n";
p += "Hub: docs/experiments/redline-kernel-launch/. Code: src/common/redline_decode_session.cpp.\n";
p += "FACTS: floor micro WINS; gen B1 OWN_RMSNORM ~−2.6%; P12b POST_SYNC=off ≈ device ";
p += "(post-sync NOT primary tax). Default ON forbidden. No fake gen t/s. No force-push.\n";
p += "MANDATORY: search_tool+use_tool clear-thought__sequentialthinking once, then act.\n";
p += "FOCUS: " + focus + "\n";

if mode == "qr" {
    p += "ROLE: single quality-reviewer. Read git diff / recent P12B docs. ";
    p += "Verdict PASS|FAIL on bans + honesty. Do not re-explore whole roadmap.\n";
} else {
    p += "ROLE: single senior-dev. ONE slice only. Prefer small code+doc change. ";
    p += "Rebuild chat if code changes. Skip multi-rep gen A/B unless focus demands it ";
    p += "and GPU is free (else PENDING_BENCH). Write short notes under ";
    p += "docs/experiments/redline-kernel-launch/ if useful.\n";
}

let r = agent(p, #{
    label: if mode == "qr" { "lean-qr" } else { "lean-fire" },
    agent_type: if mode == "qr" { "quality-reviewer" } else { "senior-developer" },
    capability_mode: "all",
    output_schema: fire_schema,
});

let summary = "agent failed";
let next_step = "parent continues";
let ok = false;
if r != () && r.success && r.output != () {
    ok = r.output.ok == true;
    if r.output.summary != () { summary = r.output.summary; }
    if r.output.next != () { next_step = r.output.next; }
    if r.output.clear_thought_used != true {
        log("WARN clear_thought_used=false");
    }
}

let scratch = "# lean redline\n\n" + summary + "\n\nnext: " + next_step + "\n";
let sp = write_scratch_file("redline-lean.md", scratch);
complete(#{
    ok: ok,
    mode: mode,
    summary: summary,
    next: next_step,
    product_default_on: false,
    scratch: sp,
    note: "Lean 1-agent workflow. Prefer parent for multi-step critical work.",
});
