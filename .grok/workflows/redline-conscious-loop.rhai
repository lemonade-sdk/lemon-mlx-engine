// redline-conscious-loop — continuous Redline product-ownership development
//
// Conscious critical loop for lemonade-sdk/lemon-mlx-engine branch
// exp/redline-kernel-launch. Orchestrates domain subagents with mandatory
// Clear Thought MCP on every agent. Does NOT invent gen t/s wins.
//
// Flow:
//   Ground (facts) → Domain panel (parallel specialists + Clear Thought)
//   → Strategy (plan) → Implement (senior-dev, optional) → Quintuple review
//   → Synthesize next-loop docs
//
// Args:
//   mode: "assess" | "p12b" | "full"   (default "full")
//   skip_implement: bool               (default false when mode full/p12b)
//   skip_bench: bool                   (default true — agents plan benches;
//                                        parent/host runs GPU benches)
//   focus: string                      optional extra focus (e.g. "sync-tax")
//
// Hard bans (every agent prompt):
//   - No LoopBrake / fake gen t/s
//   - No product default ON until measured ≥2% B1 win
//   - XOR pure-graph stay fail-closed
//   - No force-push
//   - PENDING_BENCH if GPU busy / dual writers
//   - Microbench µs ≠ gen t/s
//   - Slogan: replace product launches, don't pile additive flags
//
// Each agent MUST:
//   1) search_tool for clear-thought, then use_tool at least once
//      (sequentialthinking and/or mentalmodel and/or debuggingapproach)
//   2) set clear_thought_used=true only if those calls succeeded
//   3) read real files under docs/experiments/redline-kernel-launch/ and src/
//

let meta = #{
    name: "redline-conscious-loop",
    description: "Clear Thought + domain subagents loop for Redline product ownership (P12b/sync tax, residual ownership, honest B0/B1/B2)",
    when_to_use: "Continuous Redline kernel-launch work on exp/redline-kernel-launch: assess, plan P12b, implement, quintuple review",
    phases: [
        #{ title: "Ground", detail: "roadmap + latest gen A/B facts" },
        #{ title: "DomainPanel", detail: "parallel domain specialists + Clear Thought MCP" },
        #{ title: "Strategy", detail: "plan next slice (P12b or residual)" },
        #{ title: "Implement", detail: "senior-dev code or doc-only" },
        #{ title: "Quintuple", detail: "explore/plan/impl/QR/supervisor" },
        #{ title: "Synthesize", detail: "loop report + next fire" },
    ],
};

let ground_schema = #{
    "type": "object",
    "required": ["ok", "summary", "wins", "losses", "next_code", "clear_thought_used"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "wins": #{ "type": "array", "items": #{ "type": "string" } },
        "losses": #{ "type": "array", "items": #{ "type": "string" } },
        "next_code": #{ "type": "string" },
        "gpu_note": #{ "type": "string" },
        "clear_thought_used": #{ "type": "boolean" },
    },
};

let domain_schema = #{
    "type": "object",
    "required": ["ok", "domain", "summary", "findings", "risks", "clear_thought_used"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "domain": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "findings": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 12 },
        "risks": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 8 },
        "recommended_action": #{ "type": "string" },
        "clear_thought_used": #{ "type": "boolean" },
        "artifact_path": #{ "type": "string" },
    },
};

let strategy_schema = #{
    "type": "object",
    "required": ["ok", "decision", "rationale", "steps", "success_metric", "clear_thought_used"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "decision": #{ "type": "string" },
        "rationale": #{ "type": "string" },
        "steps": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 12 },
        "success_metric": #{ "type": "string" },
        "files_to_touch": #{ "type": "array", "items": #{ "type": "string" } },
        "default_on_allowed": #{ "type": "boolean" },
        "clear_thought_used": #{ "type": "boolean" },
    },
};

let implement_schema = #{
    "type": "object",
    "required": ["ok", "changed", "summary", "files", "build_ok", "clear_thought_used"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "changed": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "files": #{ "type": "array", "items": #{ "type": "string" } },
        "build_ok": #{ "type": "boolean" },
        "bench_status": #{ "type": "string" },
        "clear_thought_used": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let review_schema = #{
    "type": "object",
    "required": ["ok", "role", "verdict", "issues", "clear_thought_used"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "role": #{ "type": "string" },
        "verdict": #{ "type": "string" },
        "issues": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 10 },
        "bans_ok": #{ "type": "boolean" },
        "clear_thought_used": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let report_schema = #{
    "type": "object",
    "required": ["ok", "report_path", "summary", "next_fire", "clear_thought_used"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "report_path": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "next_fire": #{ "type": "string" },
        "product_default_on": #{ "type": "boolean" },
        "clear_thought_used": #{ "type": "boolean" },
    },
};

// --- Clear Thought preamble (shared; every agent must obey) ---
fn clear_thought_preamble() {
    let s = "";
    s += "MANDATORY CLEAR THOUGHT MCP: Before concluding, call search_tool with query ";
    s += "'clear-thought sequentialthinking' (or mentalmodel/debuggingapproach), then ";
    s += "use_tool to invoke clear-thought__sequentialthinking AND at least one of ";
    s += "clear-thought__mentalmodel (first_principles or pareto_principle) or ";
    s += "clear-thought__debuggingapproach. Set clear_thought_used=true ONLY if those ";
    s += "MCP calls actually ran. Capability mode includes MCP via search_tool/use_tool.\n";
    s += "HARD BANS: no fake gen t/s; no product default ON until ≥2% B1 win measured; ";
    s += "XOR pure-graph fail-closed; no force-push; microbench ≠ gen t/s; do not pile ";
    s += "additive SMALL_OP/SIDECAR as speed. Slogan: replace product launches.\n";
    s += "Repo root: /home/antmi/lemon-mlx-engine. Branch: exp/redline-kernel-launch.\n";
    s += "Hub: docs/experiments/redline-kernel-launch/ (ROADMAP.md MASTER.md P12_OWN_RMSNORM.md ";
    s += "GEN_AB_*.md). Code: src/common/redline_decode_session.cpp/.h graph_decode.cpp.\n";
    s += "Known facts: E1 ~1.9× launch floor WIN; P10 retained glue ~300× oneshot WIN; ";
    s += "gen B1 OWN_RMSNORM ~−2–5% LOSS (sync tax); B2 all-flags LOSS; OWN_GLUE gen TIE. ";
    s += "Next code priority: P12b cut mid-eval hipDeviceSynchronize tax on OWN_RMSNORM/OWN_GLUE.\n";
    s
}

// --- args ---
let mode = "full";
let skip_implement = false;
let skip_bench = true;
let focus = "";

if args != () {
    if args.mode != () { mode = args.mode; }
    if args.skip_implement != () { skip_implement = args.skip_implement; }
    if args.skip_bench != () { skip_bench = args.skip_bench; }
    if args.focus != () { focus = args.focus; }
}

if mode != "assess" && mode != "p12b" && mode != "full" {
    mode = "full";
}
if mode == "assess" {
    skip_implement = true;
}
if mode == "p12b" {
    focus = "P12b OWN_RMSNORM/OWN_GLUE mid-eval sync tax";
}

log("redline-conscious-loop start mode=" + mode
    + " skip_implement=" + skip_implement.to_string()
    + " focus=" + focus);

// =============================================================================
// Phase: Ground
// =============================================================================
phase("Ground");

let ground_prompt = clear_thought_preamble();
ground_prompt += "ROLE: ground-facts explorer (read-only).\n";
ground_prompt += "TASK: Read ROADMAP.md, MASTER.md, GEN_AB_RETRY2_20260808.md, ";
ground_prompt += "GEN_AB_CLEAN_20260808.md, P12_OWN_RMSNORM.md, and skim ";
ground_prompt += "redline_try_own_rmsnorm_packed in src/common/redline_decode_session.cpp ";
ground_prompt += "(hipStreamSynchronize / hipDeviceSynchronize sites). ";
ground_prompt += "Optionally run: rocm-smi --showuse --showmeminfo vram; note lemonade contention.\n";
ground_prompt += "OUTPUT schema fields: ok, summary (≤400 chars), wins[], losses[], ";
ground_prompt += "next_code (one primary next code slice), gpu_note, clear_thought_used.\n";
ground_prompt += "Use tools; do not invent numbers — quote from docs.\n";
if focus != "" {
    ground_prompt += "Extra focus: " + focus + "\n";
}

let ground = agent(ground_prompt, #{
    label: "ground-facts",
    agent_type: "explore",
    capability_mode: "all",
    output_schema: ground_schema,
});

let ground_ok = ground != () && ground.success && ground.output.ok == true;
if !ground_ok {
    log("ground failed or incomplete — continuing with known defaults");
}
let next_code_hint = "P12b cut OWN_RMSNORM mid-eval device sync tax";
if ground_ok && ground.output.next_code != () && ground.output.next_code != "" {
    next_code_hint = ground.output.next_code;
}
log("next_code_hint=" + next_code_hint);

// =============================================================================
// Phase: DomainPanel (parallel specialists)
// =============================================================================
phase("DomainPanel");

let domains = [
    #{
        id: "dispatch-floor",
        title: "AQL/PM4 dispatch floor & retained IB",
        brief: "E1/E2/P2/P10 floors; when floor wins transfer to gen; retained vs oneshot",
    },
    #{
        id: "product-ownership",
        title: "Product path ownership (glue/RMSNorm residual)",
        brief: "OWN_GLUE OWN_RMSNORM inventory residual strided CustomKernel; replace not add",
    },
    #{
        id: "sync-ordering",
        title: "Mid-eval sync tax & dual-queue ordering",
        brief: "hipDeviceSynchronize sites in OWN_*; pre/post sync; events vs device fence; P12b design",
    },
    #{
        id: "gen-measure",
        title: "Gen A/B discipline B0/B1/B2",
        brief: "honest measure protocol; when PENDING_BENCH; never claim micro as gen",
    },
    #{
        id: "rocm-hsa",
        title: "ROCm/HSA/gfx1150 integration hazards",
        brief: "RUNPATH counted_queue_acquire; dual VRAM lemonade; gfx1150 PM4 path",
    },
];

let djobs = [];
for d in domains {
    let dp = clear_thought_preamble();
    dp += "ROLE: domain specialist — " + d.title + " (id=" + d.id + ").\n";
    dp += "DOMAIN BRIEF: " + d.brief + "\n";
    dp += "TASK: Investigate this domain only. Read relevant docs/code with tools. ";
    dp += "Apply Clear Thought MCP first_principles or debuggingapproach. ";
    dp += "Return findings (concrete, file-backed), risks, recommended_action for THIS loop.\n";
    dp += "Primary next_code_hint from ground: " + next_code_hint + "\n";
    if focus != "" {
        dp += "Focus: " + focus + "\n";
    }
    dp += "domain field must equal \"" + d.id + "\". clear_thought_used must reflect MCP use.\n";
    djobs.push(#{
        prompt: dp,
        label: "domain:" + d.id,
        agent_type: "explore",
        capability_mode: "all",
        output_schema: domain_schema,
    });
}
let dresults = parallel(djobs);

let domain_summaries = [];
let domain_actions = [];
let di = 0;
for dr in dresults {
    if dr != () && dr.success && dr.output != () {
        let sline = dr.output.domain + ": " + dr.output.summary;
        domain_summaries.push(sline);
        if dr.output.recommended_action != () && dr.output.recommended_action != "" {
            domain_actions.push(dr.output.recommended_action);
        }
        if dr.output.clear_thought_used != true {
            log("WARN domain " + dr.output.domain + " clear_thought_used=false");
        }
    } else {
        log("WARN domain panel slot " + di.to_string() + " failed");
    }
    di += 1;
}
log("domain panel collected " + domain_summaries.len().to_string() + " summaries");

// =============================================================================
// Phase: Strategy
// =============================================================================
phase("Strategy");

let strat_prompt = clear_thought_preamble();
strat_prompt += "ROLE: planning-analysis-strategist / plan agent.\n";
strat_prompt += "TASK: Choose ONE next implementation slice for this fire. Prefer P12b ";
strat_prompt += "(reduce OWN_RMSNORM/OWN_GLUE hipDeviceSynchronize tax without races) ";
strat_prompt += "unless domain evidence says a different residual ownership is higher leverage.\n";
strat_prompt += "Use Clear Thought decisionframework-style reasoning via sequentialthinking MCP.\n";
strat_prompt += "Ground next_code_hint: " + next_code_hint + "\n";
strat_prompt += "Domain actions (JSON-ish list):\n";
for a in domain_actions {
    strat_prompt += " - " + a + "\n";
}
strat_prompt += "Domain summaries:\n";
for s in domain_summaries {
    strat_prompt += " - " + s + "\n";
}
strat_prompt += "decision: short slug (e.g. p12b-stream-sync, own-strided-rms, measure-only).\n";
strat_prompt += "default_on_allowed MUST be false unless citing a measured ≥2% B1 gen win (there is none).\n";
strat_prompt += "success_metric: how we know this slice worked (smoke + optional gen B0/B1).\n";
strat_prompt += "steps: ordered implementation steps. files_to_touch: paths.\n";
if mode == "assess" {
    strat_prompt += "MODE=assess: plan only; decision may be measure-only or p12b design without code.\n";
}
if mode == "p12b" {
    strat_prompt += "MODE=p12b: decision MUST be a P12b sync-tax reduction plan with concrete code steps.\n";
}

let strategy = agent(strat_prompt, #{
    label: "strategy-plan",
    agent_type: "plan",
    capability_mode: "all",
    output_schema: strategy_schema,
});

let decision = "p12b-stream-sync";
let steps_blob = "";
let files_blob = "";
let strategy_ok = strategy != () && strategy.success && strategy.output.ok == true;
if strategy_ok {
    if strategy.output.decision != () && strategy.output.decision != "" {
        decision = strategy.output.decision;
    }
    if strategy.output.steps != () {
        for st in strategy.output.steps {
            steps_blob += " - " + st + "\n";
        }
    }
    if strategy.output.files_to_touch != () {
        for f in strategy.output.files_to_touch {
            files_blob += f + " ";
        }
    }
    if strategy.output.default_on_allowed == true {
        log("WARN strategy set default_on_allowed=true — FORBIDDEN without ≥2% B1; treat as false");
    }
}
log("strategy decision=" + decision);

// =============================================================================
// Phase: Implement
// =============================================================================
phase("Implement");

let impl_ok = false;
let impl_changed = false;
let impl_summary = "skipped";
let impl_files = [];

if skip_implement {
    log("skip_implement=true — no code phase");
    impl_summary = "implement skipped (assess or flag)";
} else {
    let impl_prompt = clear_thought_preamble();
    impl_prompt += "ROLE: senior-developer for Redline P12b / ownership.\n";
    impl_prompt += "DECISION from plan: " + decision + "\n";
    impl_prompt += "PLANNED STEPS:\n" + steps_blob + "\n";
    impl_prompt += "FILES HINT: " + files_blob + "\n";
    impl_prompt += "TASK:\n";
    impl_prompt += "1) Clear Thought first_principles on whether each hipDeviceSynchronize is ";
    impl_prompt += "required for dual-queue correctness or can be thinned (stream-only, remove ";
    impl_prompt += "redundant post-sync if consumers already fence, optional env for aggressive path).\n";
    impl_prompt += "2) Implement the SMALLEST safe change for this decision in the parent workspace ";
    impl_prompt += "(not a fake stub). Prefer: document P12b design + optional env ";
    impl_prompt += "MLX_REDLINE_OWN_RMSNORM_LAZY_SYNC=1 that skips post hipDeviceSynchronize only when ";
    impl_prompt += "safe experiments allow, OR replace device-wide post-sync with hipStreamSynchronize ";
    impl_prompt += "on the product stream when hip_stream is non-null (still a wait, but not whole device).\n";
    impl_prompt += "3) Keep default OFF; never flip product default ON.\n";
    impl_prompt += "4) Rebuild chat if code changed: cmake --build build -j --target chat ";
    impl_prompt += "(or ninja -C build chat). If build fails, report build_ok=false.\n";
    impl_prompt += "5) Write/update docs: docs/experiments/redline-kernel-launch/P12B_SYNC_TAX.md ";
    impl_prompt += "with honesty (NOT gen t/s claim until B0/B1 remeasured).\n";
    impl_prompt += "6) Do NOT git push unless explicitly needed; prefer local files. No force-push.\n";
    if skip_bench {
        impl_prompt += "skip_bench=true: do NOT run long gen A/B; note bench_status=PENDING_BENCH.\n";
    } else {
        impl_prompt += "If GPU free (rocm-smi), optional short 0.8B B0 vs B1 only; else PENDING_BENCH.\n";
    }
    impl_prompt += "OUTPUT: ok, changed, summary, files[], build_ok, bench_status, notes, clear_thought_used.\n";

    let impl = agent(impl_prompt, #{
        label: "senior-dev-implement",
        agent_type: "senior-developer",
        capability_mode: "all",
        output_schema: implement_schema,
    });

    if impl != () && impl.success && impl.output != () {
        impl_ok = impl.output.ok == true;
        impl_changed = impl.output.changed == true;
        if impl.output.summary != () {
            impl_summary = impl.output.summary;
        }
        if impl.output.files != () {
            impl_files = impl.output.files;
        }
        log("implement ok=" + impl_ok.to_string()
            + " changed=" + impl_changed.to_string()
            + " build_ok=" + impl.output.build_ok.to_string());
    } else {
        log("implement agent failed");
        impl_summary = "implement agent failed";
    }
}

// =============================================================================
// Phase: Quintuple review
// =============================================================================
phase("Quintuple");

let review_roles = [
    #{ id: "facts", title: "explore/facts", agent_type: "explore" },
    #{ id: "strategy", title: "plan/strategy-check", agent_type: "plan" },
    #{ id: "impl", title: "implement-coherence", agent_type: "quality-reviewer" },
    #{ id: "quality", title: "quality-reviewer", agent_type: "quality-reviewer" },
    #{ id: "supervisor", title: "supervisor-second-QR", agent_type: "quality-reviewer" },
];

let rjobs = [];
for role in review_roles {
    let rp = clear_thought_preamble();
    rp += "ROLE: quintuple check — " + role.title + " (id=" + role.id + ").\n";
    rp += "DECISION: " + decision + "\n";
    rp += "IMPLEMENT SUMMARY: " + impl_summary + "\n";
    rp += "TASK: Independently verify this fire using tools. Read any files claimed changed. ";
    rp += "Use Clear Thought metacognitivemonitoring or sequentialthinking via MCP.\n";
    rp += "verdict: PASS | PASS_WITH_NOTES | FAIL | SKIP (if nothing to review).\n";
    rp += "issues: concrete problems only. bans_ok: true only if hard bans held.\n";
    rp += "role field must equal \"" + role.id + "\".\n";
    rp += "Fail closed on fake gen claims or default ON without measure.\n";
    rjobs.push(#{
        prompt: rp,
        label: "qr:" + role.id,
        agent_type: role.agent_type,
        capability_mode: "all",
        output_schema: review_schema,
    });
}
let rresults = parallel(rjobs);

let review_lines = [];
let any_fail = false;
let ri = 0;
for rr in rresults {
    if rr != () && rr.success && rr.output != () {
        let line = rr.output.role + "=" + rr.output.verdict;
        review_lines.push(line);
        if rr.output.verdict == "FAIL" {
            any_fail = true;
        }
        if rr.output.clear_thought_used != true {
            log("WARN review " + rr.output.role + " clear_thought_used=false");
        }
    } else {
        review_lines.push("slot" + ri.to_string() + "=MISSING");
        any_fail = true;
    }
    ri += 1;
}
log("quintuple: " + review_lines.len().to_string() + " roles; any_fail=" + any_fail.to_string());

// =============================================================================
// Phase: Synthesize
// =============================================================================
phase("Synthesize");

let syn_prompt = clear_thought_preamble();
syn_prompt += "ROLE: software-program-manager / loop synthesizer.\n";
syn_prompt += "TASK: Write the loop report to disk and return schema.\n";
syn_prompt += "Create/update: docs/experiments/redline-kernel-launch/LOOP_REPORT_LATEST.md ";
syn_prompt += "AND docs/experiments/redline-kernel-launch/LOOP_REPORT_WORKFLOW.md with:\n";
syn_prompt += "- mode, decision, implement summary\n";
syn_prompt += "- domain panel bullets\n";
syn_prompt += "- quintuple verdicts\n";
syn_prompt += "- honest win/loss (no fake TPS)\n";
syn_prompt += "- next_fire: concrete next agent/human step\n";
syn_prompt += "- product_default_on: false\n";
syn_prompt += "Also lightly touch ROADMAP.md living section if needed (1–3 lines).\n";
syn_prompt += "Use Clear Thought sequentialthinking to critically challenge whether this fire ";
syn_prompt += "advanced product ownership or only docs.\n";
syn_prompt += "DECISION: " + decision + "\n";
syn_prompt += "IMPL: " + impl_summary + "\n";
syn_prompt += "REVIEWS:\n";
for rl in review_lines {
    syn_prompt += " - " + rl + "\n";
}
syn_prompt += "DOMAINS:\n";
for s in domain_summaries {
    syn_prompt += " - " + s + "\n";
}
syn_prompt += "report_path must be the markdown path written. clear_thought_used required.\n";

let report = agent(syn_prompt, #{
    label: "synthesize-report",
    agent_type: "general-purpose",
    capability_mode: "all",
    output_schema: report_schema,
});

let report_path = "docs/experiments/redline-kernel-launch/LOOP_REPORT_WORKFLOW.md";
let report_summary = "workflow complete";
let next_fire = "parent continues P12b + optional B0/B1 when GPU free";
if report != () && report.success && report.output != () {
    if report.output.report_path != () && report.output.report_path != "" {
        report_path = report.output.report_path;
    }
    if report.output.summary != () {
        report_summary = report.output.summary;
    }
    if report.output.next_fire != () {
        next_fire = report.output.next_fire;
    }
}

// Scratch summary for UI
let scratch = "";
scratch += "# redline-conscious-loop result\n\n";
scratch += "- mode: " + mode + "\n";
scratch += "- decision: " + decision + "\n";
scratch += "- implement: " + impl_summary + "\n";
scratch += "- report: " + report_path + "\n";
scratch += "- next_fire: " + next_fire + "\n";
scratch += "- any_quintuple_fail: " + any_fail.to_string() + "\n";
scratch += "- product_default_on: false\n";
let sp = write_scratch_file("redline-loop-summary.md", scratch);

complete(#{
    ok: true,
    mode: mode,
    decision: decision,
    implement_summary: impl_summary,
    implement_changed: impl_changed,
    report_path: report_path,
    report_summary: report_summary,
    next_fire: next_fire,
    quintuple_fail: any_fail,
    product_default_on: false,
    scratch: sp,
});
