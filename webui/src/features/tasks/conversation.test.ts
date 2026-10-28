import { expect, it } from "vitest";
import type { WorkbenchEvent } from "../../api";
import { repliesForRun, stopExplanation } from "./conversation";
const e = (type: string, payload: Record<string, unknown>, run_id = "r"): WorkbenchEvent => ({ seq: 1, type, payload, run_id });
it("separates tool-call progress from the final answer and isolates runs", () => {
  const events = [e("model_stream_started", {}), e("text_delta", {text: "Checking"}), e("model_response_completed", {tool_call_count: 1, finish_reason: "tool_calls"}), e("text_delta", {text: "other"}, "old"), e("model_stream_started", {}), e("text_delta", {text: "Result"}), e("model_response_completed", {tool_call_count: 0, finish_reason: "stop"}), e("loop_completed", {reason: "final_response"})];
  expect(repliesForRun(events, "r")).toEqual([{text: "Checking", final: false}, {text: "Result", final: true}]);
});
it("explains a no-progress stop with the actual failed tool", () => {
  const events = [e("tool_failed", {tool_name: "generation.register_hypothesis", error: "unknown evidence id"}), e("loop_completed", {reason: "no_progress"})];
  expect(stopExplanation(events, "r")).toContain("generation.register_hypothesis");
  expect(stopExplanation(events, "r")).toContain("unknown evidence id");
  expect(stopExplanation(events, "other")).toBeNull();
});

it("does not promote provider truncation to a final reply", () => {
  for (const reason of ["max_tokens", "unknown", "error", "cancelled"]) {
    expect(repliesForRun([e("text_delta", {text: "incomplete"}), e("model_response_completed", {tool_call_count: 0, finish_reason: reason})], "r")[0].final).toBe(false);
  }
});

it("keeps a normal reply provisional when runtime stops at the call limit", () => {
  expect(repliesForRun([e("text_delta", {text: "provisional"}), e("model_response_completed", {tool_call_count: 0, finish_reason: "stop"}), e("loop_completed", {reason: "max_iterations"})], "r")[0].final).toBe(false);
});

it("demotes the first maker answer when automatic repair starts", () => {
  const events = [e("text_delta", { text: "draft" }), e("model_response_completed", { tool_call_count: 0, finish_reason: "stop" }),
    e("loop_completed", { reason: "final_response" }), e("closed_loop_round_started", { round_index: 2, phase: "repair" }),
    e("model_stream_started", {}), e("text_delta", { text: "repaired" }),
    e("model_response_completed", { tool_call_count: 0, finish_reason: "stop" }), e("loop_completed", { reason: "final_response" }),
    e("closed_loop_completed", { delivery_status: "PASS", scientific_status: "UNKNOWN" })];
  const replies = repliesForRun(events, "r");
  expect(replies.map(r => r.final)).toEqual([false, true]);
});
